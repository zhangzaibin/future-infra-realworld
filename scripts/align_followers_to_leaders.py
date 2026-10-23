#!/usr/bin/env python
"""一次性将 ttyACM0/2 从臂平滑对齐到 ttyACM1/3 主臂的当前姿态。"""

import argparse
import logging
import math
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("必须是有限的正数")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-follower-port", default="/dev/ttyACM0")
    parser.add_argument("--left-leader-port", default="/dev/ttyACM1")
    parser.add_argument("--right-follower-port", default="/dev/ttyACM2")
    parser.add_argument("--right-leader-port", default="/dev/ttyACM3")
    parser.add_argument("--robot-id", default="bimanual_follower", help="从臂校准 ID 前缀")
    parser.add_argument("--teleop-id", default="bimanual_leader", help="主臂校准 ID 前缀")
    parser.add_argument("--duration", type=positive_float, default=5.0, help="渐进移动时间，秒（默认 5）")
    parser.add_argument("--timeout", type=positive_float, default=20.0, help="到位超时，秒（默认 20）")
    parser.add_argument("--fps", type=positive_float, default=30.0)
    parser.add_argument("--max-step", type=positive_float, default=2.0, help="每周期最大归一化位置增量")
    parser.add_argument("--tolerance", type=positive_float, default=2.0, help="到位容差，归一化位置单位")
    parser.add_argument("--dry-run", action="store_true", help="只读取并显示位置，不写入电机")
    parser.add_argument("--release-on-exit", action="store_true", help="成功后释放从臂扭矩；默认保持姿态")
    args = parser.parse_args(argv)
    ports = [
        args.left_follower_port, args.left_leader_port,
        args.right_follower_port, args.right_leader_port,
    ]
    if len({Path(port).resolve() for port in ports}) != 4:
        parser.error("四个机械臂必须使用不同的串口")
    if args.timeout <= args.duration:
        parser.error("--timeout 必须大于 --duration")
    return args


def make_pairs(args):
    # 延迟导入，让 --help 无需安装硬件依赖即可使用。
    from lerobot.robots.so100_follower import SO100Follower, SO100FollowerConfig
    from lerobot.teleoperators.so100_leader import SO100Leader, SO100LeaderConfig

    return [
        (
            SO100Follower(SO100FollowerConfig(
                port=getattr(args, f"{side}_follower_port"),
                id=f"{args.robot_id}_{side}", cameras={},
            )),
            SO100Leader(SO100LeaderConfig(
                port=getattr(args, f"{side}_leader_port"), id=f"{args.teleop_id}_{side}",
            )),
        )
        for side in ("left", "right")
    ]


def read_positions(arm):
    positions = arm.bus.sync_read("Present_Position")
    if positions.keys() != arm.bus.motors.keys():
        raise ValueError(f"{arm.id}: 关节读数不完整")
    for name, value in positions.items():
        lower = 0 if name == "gripper" else -100
        if not math.isfinite(value) or not lower <= value <= 100:
            raise ValueError(f"{arm.id}: {name} 位置无效: {value}")
    return positions


def move_to_targets(pairs, targets, args):
    starts = [read_positions(follower) for follower, _ in pairs]
    previous = [dict(start) for start in starts]
    start_time = time.monotonic()
    stable_frames = 0
    while True:
        tick = time.monotonic()
        elapsed = tick - start_time
        observed = [read_positions(follower) for follower, _ in pairs]
        error = max(
            abs(target[name] - current[name])
            for target, current in zip(targets, observed, strict=True)
            for name in target
        )
        if elapsed >= args.duration and error <= args.tolerance:
            stable_frames += 1
            if stable_frames >= 3:
                logger.info("两组从臂均已到位，最大误差 %.3f", error)
                return
        else:
            stable_frames = 0
        if elapsed >= args.timeout:
            raise TimeoutError(f"对齐超时，最大位置误差 {error:.3f}；请检查阻挡、负载或校准")

        ratio = min(elapsed / args.duration, 1.0)
        blend = ratio * ratio * (3 - 2 * ratio)
        for index, (follower, _) in enumerate(pairs):
            command = {}
            for name, target in targets[index].items():
                desired = starts[index][name] + blend * (target - starts[index][name])
                # 同时约束相对上一指令和实际位置的偏差，避免阻挡后积累大幅目标。
                current = observed[index][name]
                last = previous[index][name]
                lower = max(last - args.max_step, current - args.max_step)
                upper = min(last + args.max_step, current + args.max_step)
                if lower > upper:
                    raise RuntimeError(f"{follower.id}/{name}: 实际位置偏离指令过大，停止对齐")
                command[name] = max(lower, min(upper, desired))
            follower.bus.sync_write("Goal_Position", command)
            previous[index] = command
        time.sleep(max(0.0, 1 / args.fps - (time.monotonic() - tick)))


def align(args, pairs):
    arms = [arm for pair in pairs for arm in pair]
    activated = []
    success = False
    try:
        # 在任何电机写入前检查全部四个臂。只连接总线，避免 connect() 自动使能或校准。
        for arm in arms:
            if arm.calibration.keys() != arm.bus.motors.keys():
                raise ValueError(f"{arm.id}: 缺少完整校准，请先按照 README 校准: {arm.calibration_fpath}")
            for name, motor in arm.bus.motors.items():
                cal = arm.calibration[name]
                if cal.id != motor.id or cal.range_min >= cal.range_max:
                    raise ValueError(f"{arm.id}/{name}: 校准 ID 或范围无效")
        for arm in arms:
            arm.bus.connect()
            if not arm.is_calibrated:
                raise ValueError(f"{arm.id}: 电机校准与文件不一致，请先运行 lerobot-calibrate")

        targets = []
        for follower, leader in pairs:
            current = read_positions(follower)
            target = read_positions(leader)
            if target.keys() != current.keys():
                raise ValueError("主从臂关节名称不一致")
            targets.append(target)
            logger.info("从臂 %s ← 主臂 %s", follower.config.port, leader.config.port)
            for name in target:
                logger.info("  %-14s %8.2f -> %8.2f", name, current[name], target[name])

        if args.dry_run:
            logger.info("只读预览完成，未写入电机。位置为校准后的归一化值，不是角度。")
            return

        for follower, _ in pairs:
            activated.append(follower)
            follower.bus.disable_torque()
            # configure() 会使能扭矩；先覆盖旧目标，避免上电后追逐上次运行的目标。
            present = follower.bus.sync_read("Present_Position", normalize=False)
            follower.bus.sync_write("Goal_Position", present, normalize=False)
            follower.configure()
        move_to_targets(pairs, targets, args)
        success = True
    finally:
        cleanup_errors = []
        for arm in reversed(arms):
            if not arm.bus.is_connected:
                continue
            disable = arm in activated and (not success or args.release_on_exit)
            try:
                arm.bus.disconnect(disable_torque=disable)
            except Exception as exc:
                logger.exception("断开 %s 失败", arm.id)
                cleanup_errors.append(exc)
                # 释放扭矩失败时仍尝试关闭串口，并继续清理其他臂。
                try:
                    arm.bus.disconnect(disable_torque=False)
                except Exception:
                    logger.exception("关闭 %s 串口失败", arm.id)
        if cleanup_errors and sys.exc_info()[0] is None:
            raise RuntimeError("断开机械臂失败，请检查电机状态") from cleanup_errors[0]
    if success:
        logger.info("从臂扭矩已释放。" if args.release_on_exit else "从臂保持目标姿态，串口已关闭，扭矩仍开启。")


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        align(args, make_pairs(args))
    except KeyboardInterrupt:
        logger.warning("已中断，已尝试释放从臂扭矩。")
        return 130
    except Exception as exc:
        logger.error("对齐失败: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
