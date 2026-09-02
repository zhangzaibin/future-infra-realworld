#!/usr/bin/env python3
"""
使用远程 VLA (Vision-Language-Action) API 进行真机推理

此脚本通过 HTTP 请求与远程 VLA 服务器（如 OpenPI、Pi0）通信，获取动作并控制真实机器人。

使用方法 (双摄像头: global + wrist):
    python -m lerobot.scripts.infer_with_vla_api \
        --robot.type=so101_follower \
        --robot.port=/dev/ttyACM0 \
        --robot.id=my_awesome_follower_arm \
        --robot.cameras="{ global: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
        --vla_server_url=http://192.168.1.100:20019 \
        --task="pick up the cup" \
        --camera_key=global \
        --wrist_camera_key=wrist \
        --fps=30 \
        --duration=60 \
        --use_action_chunk=true \
        --chunk_execution_horizon=50
"""

import logging
import signal
import sys
import time
from dataclasses import dataclass, field
from threading import Event, Lock
from typing import Any

import numpy as np

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    koch_follower,
    so100_follower,
    so101_follower,
)
from lerobot.robots.utils import make_robot_from_config
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging
from lerobot.vla_client import (
    VLAActionAdapter,
    VLAActionChunkExecutor,
    VLAActionLimiter,
    VLAClient,
    VLAObservationAdapter,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RobotWrapper:
    """线程安全的机器人包装器"""

    def __init__(self, robot: Robot):
        self.robot = robot
        self._lock = Lock()

    def get_observation(self) -> dict[str, Any]:
        with self._lock:
            return self.robot.get_observation()

    def send_action(self, action: dict[str, Any]):
        with self._lock:
            self.robot.send_action(action)


@dataclass
class VLAInferConfig:
    """VLA 推理配置"""

    # 机器人配置
    robot: RobotConfig = None

    # VLA 服务器配置
    vla_server_url: str = "http://localhost:20019"
    vla_timeout: float = 120.0

    # 任务描述
    task: str = "完成操作任务"

    # 相机配置
    camera_key: str = "global"  # 主相机（全局视角）
    wrist_camera_key: str | None = "wrist"  # 腕部相机

    # 关节顺序（如果需要自定义）
    joint_order: list[str] | None = None

    # 图像配置
    target_image_height: int | None = None
    target_image_width: int | None = None

    # 推理配置
    fps: float = 30.0
    duration: float = 60.0

    # Action chunk 配置
    use_action_chunk: bool = True
    chunk_execution_horizon: int = 50  # 与 DP 保持一致

    # 动作限幅配置
    enable_action_limit: bool = True
    max_action_delta: float = 5.0

    # 显示配置
    display_data: bool = False

    # 日志级别
    log_level: str = "INFO"

    def __post_init__(self):
        if self.robot is None:
            raise ValueError("必须提供机器人配置 (--robot.type=...)")
        
        # 默认关节顺序
        if self.joint_order is None:
            self.joint_order = [
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
                "gripper",
            ]


# 全局停止事件
shutdown_event = Event()


def signal_handler(signum, frame):
    """信号处理函数"""
    logger.info("收到停止信号，正在停止推理...")
    shutdown_event.set()


@parser.wrap()
def main(cfg: VLAInferConfig):
    """主函数"""
    init_logging()

    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, cfg.log_level.upper()))

    logger.info("=" * 60)
    logger.info("使用远程 VLA API 进行真机推理")
    logger.info("=" * 60)
    logger.info(f"VLA 服务器: {cfg.vla_server_url}")
    logger.info(f"任务: {cfg.task}")
    logger.info(f"相机: {cfg.camera_key}")
    if cfg.wrist_camera_key:
        logger.info(f"腕部相机: {cfg.wrist_camera_key}")
    logger.info(f"FPS: {cfg.fps}")
    logger.info(f"运行时长: {cfg.duration} 秒")
    logger.info(f"Action Chunk: {cfg.use_action_chunk} (horizon={cfg.chunk_execution_horizon})")
    logger.info(f"动作限幅: {cfg.enable_action_limit}")
    logger.info("=" * 60)

    vla_client = None

    try:
        # 1. 初始化机器人
        logger.info(f"初始化机器人: {cfg.robot.type}")
        robot = make_robot_from_config(cfg.robot)
        robot.connect()
        robot_wrapper = RobotWrapper(robot)
        logger.info("机器人连接成功")

        # 2. 初始化 VLA 客户端
        logger.info(f"连接到 VLA 服务器: {cfg.vla_server_url}")
        vla_client = VLAClient(server_url=cfg.vla_server_url, timeout=cfg.vla_timeout)
        logger.info("VLA 客户端连接成功")

        # 3. 初始化适配器
        target_size = None
        if cfg.target_image_height and cfg.target_image_width:
            target_size = (cfg.target_image_height, cfg.target_image_width)

        obs_adapter = VLAObservationAdapter(
            joint_order=cfg.joint_order,
            front_camera_key=cfg.camera_key,
            wrist_camera_key=cfg.wrist_camera_key,
            target_image_size=target_size,
            default_task=cfg.task,
        )

        action_adapter = VLAActionAdapter(
            joint_order=cfg.joint_order,
            chunk_index=0,
        )

        # 4. 初始化动作限幅器
        action_limiter = None
        if cfg.enable_action_limit:
            action_limiter = VLAActionLimiter(
                joint_order=cfg.joint_order,
                max_delta=cfg.max_action_delta,
            )
            # 获取机器人当前位置作为初始位置
            initial_obs = robot_wrapper.get_observation()
            initial_pos = {k: v for k, v in initial_obs.items() if k.endswith(".pos")}
            action_limiter.set_last_action(initial_pos)
            logger.info(f"动作限幅器初始化，当前位置: {initial_pos}")

        # 5. 开始推理循环
        logger.info("开始推理循环...")
        logger.info("按 Ctrl+C 停止")

        if cfg.use_action_chunk:
            run_inference_loop_with_chunk(
                robot=robot_wrapper,
                vla_client=vla_client,
                obs_adapter=obs_adapter,
                action_adapter=action_adapter,
                action_limiter=action_limiter,
                shutdown_event=shutdown_event,
                task=cfg.task,
                fps=cfg.fps,
                duration=cfg.duration,
                chunk_execution_horizon=cfg.chunk_execution_horizon,
            )
        else:
            run_inference_loop(
                robot=robot_wrapper,
                vla_client=vla_client,
                obs_adapter=obs_adapter,
                action_adapter=action_adapter,
                action_limiter=action_limiter,
                shutdown_event=shutdown_event,
                task=cfg.task,
                fps=cfg.fps,
                duration=cfg.duration,
            )

    except Exception as e:
        logger.error(f"推理过程出错: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    finally:
        # 清理资源
        if "robot" in locals():
            try:
                robot.disconnect()
                logger.info("机器人已断开连接")
            except Exception as e:
                logger.error(f"断开机器人连接时出错: {e}")

        logger.info("推理结束")


def run_inference_loop(
    robot: RobotWrapper,
    vla_client: VLAClient,
    obs_adapter: VLAObservationAdapter,
    action_adapter: VLAActionAdapter,
    action_limiter: VLAActionLimiter | None,
    shutdown_event: Event,
    task: str,
    fps: float,
    duration: float,
):
    """
    单步动作推理循环
    每个时间步都获取新的观察并推理动作
    """
    step_count = 0
    start_time = time.perf_counter()
    action_interval = 1.0 / fps

    while not shutdown_event.is_set():
        loop_start = time.perf_counter()

        # 检查是否超时
        elapsed = loop_start - start_time
        if elapsed >= duration:
            logger.info(f"达到运行时长 {duration} 秒，停止推理")
            break

        try:
            # 1. 获取机器人观察
            lerobot_obs = robot.get_observation()

            # 2. 转换为 VLA 格式
            vla_example = obs_adapter.convert(lerobot_obs, task=task)

            # 3. 调用 VLA 服务器获取动作
            vla_actions = vla_client.get_actions(vla_example)

            # 4. 转换为 LeRobot 格式
            lerobot_action = action_adapter.convert(vla_actions)

            # 5. 应用动作限幅
            if action_limiter is not None:
                current_pos = {k: v for k, v in lerobot_obs.items() if k.endswith(".pos")}
                lerobot_action = action_limiter.limit(lerobot_action, current_pos)

            # 6. 发送动作到机器人
            robot.send_action(lerobot_action)

            step_count += 1

            # 日志输出（每 30 步）
            if step_count % 30 == 0:
                logger.info(
                    f"步骤 {step_count} | 已运行 {elapsed:.1f}s | "
                    f"动作: gripper={lerobot_action.get('gripper.pos', 0):.2f}"
                )

        except Exception as e:
            logger.error(f"推理步骤出错: {e}")
            import traceback

            traceback.print_exc()
            continue

        # 等待下一个时间步
        dt = time.perf_counter() - loop_start
        if dt < action_interval:
            precise_sleep(action_interval - dt)

    logger.info(f"推理循环结束，共执行 {step_count} 步")


def run_inference_loop_with_chunk(
    robot: RobotWrapper,
    vla_client: VLAClient,
    obs_adapter: VLAObservationAdapter,
    action_adapter: VLAActionAdapter,
    action_limiter: VLAActionLimiter | None,
    shutdown_event: Event,
    task: str,
    fps: float,
    duration: float,
    chunk_execution_horizon: int,
):
    """
    Action Chunk 推理循环
    每隔 chunk_execution_horizon 步获取新的 action chunk，中间使用缓存的动作
    """
    step_count = 0
    chunk_count = 0
    start_time = time.perf_counter()
    action_interval = 1.0 / fps

    current_chunk: np.ndarray | None = None
    chunk_step = 0
    last_lerobot_obs: dict[str, Any] | None = None

    while not shutdown_event.is_set():
        loop_start = time.perf_counter()

        # 检查是否超时
        elapsed = loop_start - start_time
        if elapsed >= duration:
            logger.info(f"达到运行时长 {duration} 秒，停止推理")
            break

        try:
            # 检查是否需要获取新的 action chunk
            need_new_chunk = (
                current_chunk is None
                or chunk_step >= chunk_execution_horizon
                or chunk_step >= len(current_chunk)
            )

            if need_new_chunk:
                # 1. 获取机器人观察
                lerobot_obs = robot.get_observation()
                last_lerobot_obs = lerobot_obs

                # 2. 转换为 VLA 格式
                vla_example = obs_adapter.convert(lerobot_obs, task=task)

                # 3. 调用 VLA 服务器获取 action chunk
                vla_actions = vla_client.get_actions(vla_example)

                # 确保是 2D 数组
                if len(vla_actions.shape) == 1:
                    vla_actions = vla_actions.reshape(1, -1)

                current_chunk = vla_actions
                chunk_step = 0
                chunk_count += 1
                
                # 计算实际执行步数
                actual_horizon = min(chunk_execution_horizon, len(current_chunk))
                logger.info(
                    f"获取 Chunk #{chunk_count}: 共 {len(current_chunk)} 步, "
                    f"将执行 {actual_horizon} 步 (horizon={chunk_execution_horizon})"
                )

            # 4. 从 chunk 中获取当前步的动作
            action_values = current_chunk[chunk_step]
            action_adapter.chunk_index = 0  # 重置索引
            lerobot_action = action_adapter.convert(action_values)
            chunk_step += 1

            # 5. 应用动作限幅
            if action_limiter is not None:
                if last_lerobot_obs is not None:
                    current_pos = {
                        k: v for k, v in last_lerobot_obs.items() if k.endswith(".pos")
                    }
                else:
                    current_pos = None
                lerobot_action = action_limiter.limit(lerobot_action, current_pos)

            # 6. 发送动作到机器人
            robot.send_action(lerobot_action)

            step_count += 1

            # 日志输出（每个 chunk 执行完或每 50 步）
            if chunk_step >= min(chunk_execution_horizon, len(current_chunk)) or step_count % 50 == 0:
                logger.info(
                    f"步骤 {step_count} | Chunk #{chunk_count} 完成 {chunk_step}/{len(current_chunk)} 步 | "
                    f"已运行 {elapsed:.1f}s | "
                    f"gripper={lerobot_action.get('gripper.pos', 0):.2f}"
                )

        except Exception as e:
            logger.error(f"推理步骤出错: {e}")
            import traceback

            traceback.print_exc()
            continue

        # 等待下一个时间步
        dt = time.perf_counter() - loop_start
        if dt < action_interval:
            precise_sleep(action_interval - dt)

    logger.info(f"推理循环结束，共执行 {step_count} 步，{chunk_count} 个 chunks")


if __name__ == "__main__":
    main()

