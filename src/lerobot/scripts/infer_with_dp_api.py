#!/usr/bin/env python3
"""
使用远程 Diffusion Policy API 进行真机推理

此脚本通过 HTTP 请求与远程 DP 服务器通信，获取动作并控制真实机器人。

使用方法:
    python -m lerobot.scripts.infer_with_dp_api \
        --robot.type=so101_follower \
        --robot.port=/dev/ttyACM0 \
        --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
        --robot.id=my_robot \
        --dp_server_url=http://192.168.1.100:20020 \
        --agent_id=0 \
        --camera_key=front \
        --fps=30 \
        --duration=60
"""

import logging
import time
import signal
import sys
from dataclasses import dataclass, field
from threading import Event, Lock, Thread
from typing import Any, Dict, Optional, List

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
from lerobot.dp_client import DPClient, ObservationAdapter, ActionAdapter, ActionLimiter
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RobotWrapper:
    """线程安全的机器人包装器"""

    def __init__(self, robot: Robot):
        self.robot = robot
        self.lock = Lock()

    def get_observation(self) -> Dict[str, Any]:
        with self.lock:
            return self.robot.get_observation()

    def send_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            return self.robot.send_action(action)

    @property
    def name(self) -> str:
        return self.robot.name

    def connect(self):
        self.robot.connect()

    def disconnect(self):
        self.robot.disconnect()


@dataclass
class DPInferConfig:
    """DP 远程推理配置"""

    # 机器人配置
    robot: RobotConfig = None

    # DP 服务器配置
    dp_server_url: str = "http://localhost:20020"
    agent_id: int = 0

    # 相机配置
    camera_key: str = "front"  # LeRobot 中的相机名称
    camera_key_in_dp: str = "head_cam"  # DP 服务器期望的相机名称

    # 关节配置（如果需要自定义顺序）
    joint_order: Optional[List[str]] = None

    # 图像配置
    target_image_height: Optional[int] = None  # 目标图像高度，None 表示不调整
    target_image_width: Optional[int] = None  # 目标图像宽度，None 表示不调整

    # 推理配置
    fps: float = 30.0  # 控制频率 (Hz)
    duration: float = 60.0  # 运行时长 (秒)

    # Action chunk 配置
    use_action_chunk: bool = False  # 是否使用 action chunk 模式
    chunk_execution_horizon: int = 8  # 每个 chunk 执行多少步

    # 限幅配置
    enable_action_limit: bool = True  # 是否启用动作限幅

    # 显示配置
    display_data: bool = False  # 是否显示数据

    # 日志级别
    log_level: str = "INFO"

    def __post_init__(self):
        if self.robot is None:
            raise ValueError("必须提供机器人配置 (--robot.type=...)")


@parser.wrap()
def main(cfg: DPInferConfig):
    """主函数"""
    init_logging()
    
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, cfg.log_level.upper()))
    
    logger.info("=" * 60)
    logger.info("使用远程 DP API 进行真机推理")
    logger.info("=" * 60)
    logger.info(f"DP 服务器: {cfg.dp_server_url}")
    logger.info(f"Agent ID: {cfg.agent_id}")
    logger.info(f"相机: {cfg.camera_key} -> {cfg.camera_key_in_dp}")
    logger.info(f"FPS: {cfg.fps}")
    logger.info(f"运行时长: {cfg.duration} 秒")
    logger.info(f"动作限幅: {cfg.enable_action_limit}")
    logger.info("=" * 60)

    # 创建关闭事件
    shutdown_event = Event()

    def signal_handler(signum, frame):
        logger.info("收到中断信号，正在关闭...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    robot = None
    dp_client = None

    try:
        # 1. 初始化机器人
        logger.info(f"初始化机器人: {cfg.robot.type}")
        robot = make_robot_from_config(cfg.robot)
        robot.connect()
        robot_wrapper = RobotWrapper(robot)
        logger.info("机器人连接成功")

        # 2. 初始化 DP 客户端
        logger.info(f"连接到 DP 服务器: {cfg.dp_server_url}")
        dp_client = DPClient(
            agent_id=cfg.agent_id,
            server_url=cfg.dp_server_url,
        )
        logger.info("DP 客户端连接成功")

        # 3. 初始化适配器
        target_size = None
        if cfg.target_image_height and cfg.target_image_width:
            target_size = (cfg.target_image_height, cfg.target_image_width)

        obs_adapter = ObservationAdapter(
            camera_key=cfg.camera_key,
            joint_order=cfg.joint_order,
            target_image_size=target_size,
            image_key_in_dp=cfg.camera_key_in_dp,
        )

        action_adapter = ActionAdapter(
            joint_order=cfg.joint_order,
            use_action_chunk=cfg.use_action_chunk,
        )

        # 4. 初始化动作限幅器
        action_limiter = None
        if cfg.enable_action_limit:
            action_limiter = ActionLimiter(
                joint_order=cfg.joint_order,
            )
            # 获取机器人当前位置作为初始位置
            initial_obs = robot_wrapper.get_observation()
            initial_pos = {
                k: v for k, v in initial_obs.items() if k.endswith('.pos')
            }
            action_limiter.set_last_action(initial_pos)
            logger.info(f"动作限幅器初始化，当前位置: {initial_pos}")

        # 5. 重置 DP 服务器的观察历史
        logger.info("重置 DP 服务器观察历史...")
        dp_client.reset_obs()

        # 6. 开始推理循环
        logger.info("开始推理循环...")
        logger.info("按 Ctrl+C 停止")
        
        if cfg.use_action_chunk:
            run_inference_loop_with_chunk(
                robot=robot_wrapper,
                dp_client=dp_client,
                obs_adapter=obs_adapter,
                action_adapter=action_adapter,
                action_limiter=action_limiter,
                shutdown_event=shutdown_event,
                fps=cfg.fps,
                duration=cfg.duration,
                chunk_execution_horizon=cfg.chunk_execution_horizon,
            )
        else:
            run_inference_loop(
                robot=robot_wrapper,
                dp_client=dp_client,
                obs_adapter=obs_adapter,
                action_adapter=action_adapter,
                action_limiter=action_limiter,
                shutdown_event=shutdown_event,
                fps=cfg.fps,
                duration=cfg.duration,
            )

    except KeyboardInterrupt:
        logger.info("用户中断")
    except Exception as e:
        logger.error(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理
        logger.info("清理资源...")
        if robot is not None:
            try:
                robot.disconnect()
                logger.info("机器人断开连接")
            except Exception as e:
                logger.error(f"断开机器人连接时出错: {e}")

        logger.info("推理结束")


def run_inference_loop(
    robot: RobotWrapper,
    dp_client: DPClient,
    obs_adapter: ObservationAdapter,
    action_adapter: ActionAdapter,
    action_limiter: Optional[ActionLimiter],
    shutdown_event: Event,
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

            # 2. 转换为 DP 格式
            dp_obs = obs_adapter.convert(lerobot_obs)

            # 3. 调用 DP 服务器获取动作
            dp_action = dp_client.get_action(dp_obs)

            # 4. 转换为 LeRobot 格式
            lerobot_action = action_adapter.convert(dp_action)

            # 5. 应用动作限幅
            if action_limiter is not None:
                # 获取当前机器人位置
                current_pos = {k: v for k, v in lerobot_obs.items() if k.endswith('.pos')}
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
        sleep_time = action_interval - dt
        if sleep_time > 0:
            precise_sleep(sleep_time)
        else:
            logger.warning(f"推理耗时 {dt*1000:.1f}ms 超过了目标 {action_interval*1000:.1f}ms")

    logger.info(f"推理循环结束，共执行 {step_count} 步")


def run_inference_loop_with_chunk(
    robot: RobotWrapper,
    dp_client: DPClient,
    obs_adapter: ObservationAdapter,
    action_adapter: ActionAdapter,
    action_limiter: Optional[ActionLimiter],
    shutdown_event: Event,
    fps: float,
    duration: float,
    chunk_execution_horizon: int,
):
    """
    Action Chunk 推理循环
    获取一个 action chunk，执行多步后再获取新的 chunk
    """
    step_count = 0
    chunk_count = 0
    start_time = time.perf_counter()
    action_interval = 1.0 / fps

    current_chunk: Optional[np.ndarray] = None
    chunk_step = 0
    last_lerobot_obs: Optional[Dict[str, Any]] = None

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
                current_chunk is None or 
                chunk_step >= chunk_execution_horizon or
                chunk_step >= len(current_chunk)
            )

            if need_new_chunk:
                # 1. 获取机器人观察
                lerobot_obs = robot.get_observation()
                last_lerobot_obs = lerobot_obs

                # 2. 转换为 DP 格式
                dp_obs = obs_adapter.convert(lerobot_obs)

                # 3. 调用 DP 服务器获取 action chunk
                dp_action = dp_client.get_action(dp_obs)

                # 确保是 2D 数组
                if len(dp_action.shape) == 1:
                    dp_action = dp_action.reshape(1, -1)

                current_chunk = dp_action
                chunk_step = 0
                chunk_count += 1
                logger.debug(f"获取新的 action chunk #{chunk_count}, shape={current_chunk.shape}")

            # 4. 从 chunk 中获取当前步的动作
            action_values = current_chunk[chunk_step]
            lerobot_action = action_adapter.convert(action_values)
            chunk_step += 1

            # 5. 应用动作限幅
            if action_limiter is not None:
                # 获取当前机器人位置（如果刚获取了新 chunk 就用那个，否则用缓存的）
                if last_lerobot_obs is not None:
                    current_pos = {k: v for k, v in last_lerobot_obs.items() if k.endswith('.pos')}
                else:
                    current_pos = None
                lerobot_action = action_limiter.limit(lerobot_action, current_pos)

            # 6. 发送动作到机器人
            robot.send_action(lerobot_action)

            step_count += 1

            # 日志输出（每 30 步）
            if step_count % 30 == 0:
                logger.info(
                    f"步骤 {step_count} | Chunk #{chunk_count} 步 {chunk_step}/{len(current_chunk)} | "
                    f"已运行 {elapsed:.1f}s"
                )

        except Exception as e:
            logger.error(f"推理步骤出错: {e}")
            import traceback
            traceback.print_exc()
            # 重置 chunk 状态
            current_chunk = None
            chunk_step = 0
            continue

        # 等待下一个时间步
        dt = time.perf_counter() - loop_start
        sleep_time = action_interval - dt
        if sleep_time > 0:
            precise_sleep(sleep_time)
        else:
            logger.warning(f"推理耗时 {dt*1000:.1f}ms 超过了目标 {action_interval*1000:.1f}ms")

    logger.info(f"推理循环结束，共执行 {step_count} 步，{chunk_count} 个 chunks")


if __name__ == "__main__":
    main()

