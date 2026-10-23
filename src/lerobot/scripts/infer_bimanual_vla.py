#!/usr/bin/env python3
"""
使用远程 VLA (Vision-Language-Action) API 进行双臂真机推理

此脚本通过 HTTP 请求与远程 VLA 服务器（如 OpenPI、Pi0）通信，获取动作并控制双臂机器人。
Pi0 返回 32 维动作向量，取前 12 维 (left 6 + right 6)。

使用方法:
    python -m lerobot.scripts.infer_bimanual_vla \
        --robot.type=bi_so101_follower \
        --robot.left_arm_port=/dev/ttyACM0 \
        --robot.right_arm_port=/dev/ttyACM1 \
        --robot.id=bimanual_follower \
        --robot.cameras="{ global: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}, left_wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: MJPG}, right_wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
        --vla_server_url=http://10.8.131.8:20011 \
        --task="pick up the cup" \
        --fps=30 \
        --duration=60 \
        --use_action_chunk=true \
        --chunk_execution_horizon=50

人工接管（VLA 出错后按 Space 切到主臂遥操，数据按 lerobot-record 格式写入 dataset）:
    python -m lerobot.scripts.infer_bimanual_vla \
        --robot.type=bi_so100_follower \
        ... \
        --teleop.type=bi_so100_leader \
        --teleop.left_arm_port=/dev/ttyACM1 \
        --teleop.right_arm_port=/dev/ttyACM3 \
        --teleop.id=bimanual_leader \
        --dataset.repo_id=robodata/fun \
        --dataset.root=./datasets/fun \
        --dataset.single_task="fold the towels"
"""

import logging
import signal
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock
from typing import Any

import numpy as np

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.orbbec_astra.configuration_orbbec_astra import OrbbecAstraCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so100_follower,
    koch_follower,
    so100_follower,
    so101_follower,
)
from lerobot.robots.bi_so100_follower import BiSO100Follower, BiSO100FollowerConfig  # noqa: F401
from lerobot.robots.bi_so101_follower import BiSO101Follower, BiSO101FollowerConfig  # noqa: F401
from lerobot.robots.utils import make_robot_from_config
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_so100_leader,
    make_teleoperator_from_config,
)
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot.vla_client import (
    BimanualVLAActionAdapter,
    BimanualVLAActionLimiter,
    BimanualVLAObservationAdapter,
    VLAClient,
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
class TakeoverDatasetConfig:
    """人工接管后写入的 LeRobot dataset 配置，字段与 lerobot-record 对齐。"""

    repo_id: str
    single_task: str | None = None
    root: str | Path | None = None
    fps: int = 30
    num_episodes: int = 200
    video: bool = True
    push_to_hub: bool = False
    private: bool = False
    tags: list[str] | None = None
    num_image_writer_processes: int = 0
    num_image_writer_threads_per_camera: int = 4
    video_encoding_batch_size: int = 1


@dataclass
class BimanualVLAInferConfig:
    """双臂 VLA 推理配置"""

    # 机器人配置
    robot: RobotConfig = None

    # VLA 服务器配置
    vla_server_url: str = "http://10.8.131.8:20011"
    vla_timeout: float = 120.0

    # 任务描述
    task: str = "完成操作任务"

    # 相机配置
    global_camera_key: str = "global"
    left_wrist_camera_key: str = "left_wrist"
    right_wrist_camera_key: str = "right_wrist"

    # 单臂关节顺序
    single_arm_joint_order: list[str] | None = None

    # 动作维度 (从 32 维截取)
    action_dim: int = 12

    # 图像配置
    target_image_height: int | None = None
    target_image_width: int | None = None

    # 推理配置
    fps: float = 30.0
    duration: float = 6000.0

    # Action chunk 配置
    use_action_chunk: bool = True
    chunk_execution_horizon: int = 50

    # 动作限幅配置
    enable_action_limit: bool = True
    max_action_delta: float = 5.0

    # 显示配置
    display_data: bool = False

    # 人工接管：同时提供 teleop 与 dataset 后，推理中按 Space 切到主臂遥操并落盘
    teleop: TeleoperatorConfig | None = None
    dataset: TakeoverDatasetConfig | None = None
    resume: bool = True
    play_sounds: bool = False

    # 人工接管时：先让从臂静止，按 A 让主臂电机自动对齐从臂关节，再按 Enter 开始遥操录制
    enable_leader_align: bool = False
    leader_align_ramp_duration_s: float = 1.5

    # 日志级别
    log_level: str = "INFO"

    def __post_init__(self):
        if self.robot is None:
            raise ValueError("必须提供机器人配置 (--robot.type=bi_so101_follower)")

        if (self.teleop is None) != (self.dataset is None):
            raise ValueError(
                "人工接管需要同时提供 --teleop.type=... 和 --dataset.repo_id=...。"
                "只配其中一项无法启动接管。"
            )

        if self.dataset is not None:
            if not self.dataset.single_task:
                self.dataset.single_task = self.task
            self.dataset.fps = int(self.fps)

        # 默认单臂关节顺序
        if self.single_arm_joint_order is None:
            self.single_arm_joint_order = [
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
                "gripper",
            ]

    @property
    def enable_takeover(self) -> bool:
        return self.teleop is not None and self.dataset is not None


# 全局停止事件
shutdown_event = Event()

STOP_SHUTDOWN = "shutdown"
STOP_DURATION = "duration"
STOP_USER = "stop"
STOP_TAKEOVER = "takeover"


def signal_handler(signum, frame):
    """信号处理函数"""
    logger.info("收到停止信号，正在停止推理...")
    shutdown_event.set()


def inference_stop_reason(
    shutdown_event: Event,
    events: dict[str, Any] | None,
    start_time: float,
    duration: float,
) -> str | None:
    """Return why the VLA loop should stop, or None to keep running."""
    if shutdown_event.is_set():
        return STOP_SHUTDOWN
    if events is not None:
        if events.get("stop_recording"):
            return STOP_USER
        if events.get("takeover"):
            return STOP_TAKEOVER
    if time.perf_counter() - start_time >= duration:
        return STOP_DURATION
    return None


def dataset_meta_exists(root: str | Path | None) -> bool:
    return root is not None and (Path(root) / "meta" / "info.json").exists()


def setup_takeover_dataset(
    robot: Robot,
    dataset_cfg: TakeoverDatasetConfig,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    resume: bool,
) -> LeRobotDataset:
    """Create or resume a LeRobot dataset with the same features as lerobot-record."""
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=dataset_cfg.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=dataset_cfg.video,
        ),
    )

    num_cameras = len(robot.cameras) if hasattr(robot, "cameras") else 0
    image_writer_threads = dataset_cfg.num_image_writer_threads_per_camera * num_cameras
    existing = dataset_meta_exists(dataset_cfg.root)

    if existing:
        if not resume:
            raise ValueError(
                f"Dataset already exists at {dataset_cfg.root}. "
                "Pass --resume=true to append, or use a new --dataset.root."
            )
        dataset = LeRobotDataset(
            dataset_cfg.repo_id,
            root=dataset_cfg.root,
            batch_encoding_size=dataset_cfg.video_encoding_batch_size,
        )
        if num_cameras > 0:
            dataset.start_image_writer(
                num_processes=dataset_cfg.num_image_writer_processes,
                num_threads=image_writer_threads,
            )
        sanity_check_dataset_robot_compatibility(dataset, robot, dataset_cfg.fps, dataset_features)
        logger.info(f"接管数据集已恢复: {dataset_cfg.root} (已有 {dataset.num_episodes} 条 episode)")
        return dataset

    sanity_check_dataset_name(dataset_cfg.repo_id, policy_cfg=None)
    dataset = LeRobotDataset.create(
        dataset_cfg.repo_id,
        dataset_cfg.fps,
        root=dataset_cfg.root,
        robot_type=robot.name,
        features=dataset_features,
        use_videos=dataset_cfg.video,
        image_writer_processes=dataset_cfg.num_image_writer_processes,
        image_writer_threads=image_writer_threads,
        batch_encoding_size=dataset_cfg.video_encoding_batch_size,
    )
    logger.info(f"接管数据集已创建: {dataset_cfg.root or dataset_cfg.repo_id}")
    return dataset


def record_takeover_episode(
    robot: Robot,
    teleop: Teleoperator,
    dataset: LeRobotDataset,
    events: dict[str, Any],
    fps: int,
    single_task: str,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    display_data: bool = False,
    play_sounds: bool = False,
    enable_leader_align: bool = False,
    leader_align_ramp_duration_s: float = 1.5,
) -> str:
    """
    Record one teleop episode after the human takes over.

    Returns:
        "saved", "discarded", or "stopped"
    """
    events["takeover"] = False
    events["exit_early"] = False
    events["confirm_episode"] = False
    events["discard_episode"] = False

    if enable_leader_align:
        proceed = wait_for_alignment_or_start(
            robot=robot,
            teleop=teleop,
            events=events,
            ramp_duration_s=leader_align_ramp_duration_s,
            play_sounds=play_sounds,
        )
        if not proceed:
            return "stopped"

    events["recording"] = True
    log_say(
        "Human takeover. Align leader arms, then teleoperate. Enter saves, R discards, Esc stops.",
        play_sounds,
    )
    logger.info("人工接管中：请先把主臂对齐从臂再开始遥操。Enter 保存，R 丢弃，Esc 退出")

    record_loop(
        robot=robot,
        events=events,
        fps=fps,
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
        teleop=teleop,
        dataset=dataset,
        control_time_s=None,
        single_task=single_task,
        display_data=display_data,
    )
    events["recording"] = False

    if events.get("stop_recording"):
        dataset.clear_episode_buffer()
        return "stopped"

    if events.get("discard_episode"):
        log_say("Discard episode", play_sounds)
        dataset.clear_episode_buffer()
        events["discard_episode"] = False
        events["exit_early"] = False
        return "discarded"

    if not events.get("confirm_episode"):
        dataset.clear_episode_buffer()
        raise RuntimeError("Takeover recording ended without an episode decision.")

    dataset.save_episode()
    events["confirm_episode"] = False
    events["exit_early"] = False
    log_say("Episode saved", play_sounds)
    logger.info(f"接管 episode 已保存，当前共 {dataset.num_episodes} 条")
    return "saved"


def wait_for_vla_resume(events: dict[str, Any], shutdown_event: Event, play_sounds: bool = False) -> bool:
    """Wait until Enter resumes VLA inference. Returns False when the session should stop."""
    events["recording"] = False
    events["start_episode"] = False
    events["exit_early"] = False
    events["takeover"] = False
    log_say("Press Enter to resume VLA inference, or Escape to stop", play_sounds)
    logger.info("按 Enter 继续 VLA 推理，Esc 退出")

    while (
        not events["start_episode"]
        and not events["stop_recording"]
        and not shutdown_event.is_set()
    ):
        time.sleep(0.05)

    events["start_episode"] = False
    events["takeover"] = False
    return not events["stop_recording"] and not shutdown_event.is_set()


def wait_for_alignment_or_start(
    robot: Robot,
    teleop: Teleoperator,
    events: dict[str, Any],
    ramp_duration_s: float = 1.5,
    play_sounds: bool = False,
) -> bool:
    """Pause takeover: keep the follower still and wait for the human to either
    align the leader (A), start teleoperation (Enter), or stop (Esc).

    Returns True when teleoperation should start, False to stop the session.
    """
    events["recording"] = False
    events["start_episode"] = False
    events["align_leader"] = False
    events["exit_early"] = False
    events["takeover"] = False
    log_say("Takeover paused. Press A to align leader, Enter to start, Esc to stop.", play_sounds)
    logger.info("接管已暂停：按 A 让主臂对齐从臂，Enter 开始遥操录制，Esc 退出")

    while not events["start_episode"] and not events["stop_recording"]:
        if events["align_leader"]:
            events["align_leader"] = False
            try:
                follower_obs = robot.get_observation()
                teleop.align_to_follower(follower_obs, ramp_duration_s=ramp_duration_s)
                log_say("Leader aligned to follower.", play_sounds)
                logger.info("主臂已对齐从臂")
            except Exception as e:
                logger.error(f"主臂对齐失败: {e}")
                import traceback

                traceback.print_exc()
        time.sleep(0.05)

    # Release the leader torque now (before teleoperation starts, or before
    # stopping) so the leader becomes passive for the human to move it. This runs
    # on Enter (start recording) or Esc (stop), matching the flow where the leader
    # holds the aligned pose until the human is ready.
    try:
        teleop.release_torque()
    except Exception as e:
        logger.error(f"释放主臂扭矩失败: {e}")

    events["start_episode"] = False
    return not events["stop_recording"]


@parser.wrap()
def main(cfg: BimanualVLAInferConfig):
    """主函数"""
    init_logging()

    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, cfg.log_level.upper()))

    logger.info("=" * 60)
    logger.info("双臂 VLA 远程推理")
    logger.info("=" * 60)
    logger.info(f"VLA 服务器: {cfg.vla_server_url}")
    logger.info(f"任务: {cfg.task}")
    logger.info(f"全局相机: {cfg.global_camera_key}")
    logger.info(f"左手腕相机: {cfg.left_wrist_camera_key}")
    logger.info(f"右手腕相机: {cfg.right_wrist_camera_key}")
    logger.info(f"动作维度: {cfg.action_dim} (从 32 维截取)")
    logger.info(f"FPS: {cfg.fps}")
    logger.info(f"运行时长: {cfg.duration} 秒")
    logger.info(f"Action Chunk: {cfg.use_action_chunk} (horizon={cfg.chunk_execution_horizon})")
    logger.info(f"动作限幅: {cfg.enable_action_limit} (max_delta={cfg.max_action_delta})")
    logger.info(f"人工接管:         {cfg.enable_takeover}")
    if cfg.enable_takeover:
        logger.info(f"主臂:             {cfg.teleop.type} (id={cfg.teleop.id})")
        logger.info(f"接管数据集:       {cfg.dataset.repo_id} -> {cfg.dataset.root}")
        logger.info(f"接管任务文本:     {cfg.dataset.single_task}")
        logger.info(f"最大接管条数:     {cfg.dataset.num_episodes}")
        logger.info(f"主臂自动对齐:     {cfg.enable_leader_align}")
        logger.info("按键: Space/T 接管, Enter 保存并继续, R 丢弃, Esc 退出")
        if cfg.enable_leader_align:
            logger.info("接管暂停: A 对齐主臂, Enter 开始遥操, Esc 退出")
    logger.info("=" * 60)

    vla_client = None
    teleop = None
    listener = None
    dataset = None

    try:
        if cfg.enable_takeover and is_headless():
            raise RuntimeError("人工接管需要可用的键盘监听，当前环境是 headless。")

        # 1. 初始化机器人
        logger.info(f"初始化双臂机器人: {cfg.robot.type}")
        robot = make_robot_from_config(cfg.robot)
        robot.connect()
        robot_wrapper = RobotWrapper(robot)
        logger.info("双臂机器人连接成功")

        # 2. 初始化 VLA 客户端
        logger.info(f"连接到 VLA 服务器: {cfg.vla_server_url}")
        vla_client = VLAClient(server_url=cfg.vla_server_url, timeout=cfg.vla_timeout)
        logger.info("VLA 客户端连接成功")

        # 3. 初始化双臂适配器
        target_size = None
        if cfg.target_image_height and cfg.target_image_width:
            target_size = (cfg.target_image_height, cfg.target_image_width)

        obs_adapter = BimanualVLAObservationAdapter(
            single_arm_joint_order=cfg.single_arm_joint_order,
            global_camera_key=cfg.global_camera_key,
            left_wrist_camera_key=cfg.left_wrist_camera_key,
            right_wrist_camera_key=cfg.right_wrist_camera_key,
            target_image_size=target_size,
            default_task=cfg.task,
        )

        action_adapter = BimanualVLAActionAdapter(
            single_arm_joint_order=cfg.single_arm_joint_order,
            action_dim=cfg.action_dim,
            chunk_index=0,
        )

        # 4. 初始化动作限幅器
        action_limiter = None
        if cfg.enable_action_limit:
            action_limiter = BimanualVLAActionLimiter(
                single_arm_joint_order=cfg.single_arm_joint_order,
                max_delta=cfg.max_action_delta,
            )
            # 获取机器人当前位置作为初始位置
            initial_obs = robot_wrapper.get_observation()
            initial_pos = {k: v for k, v in initial_obs.items() if k.endswith(".pos")}
            action_limiter.set_last_action(initial_pos)
            logger.info(f"双臂动作限幅器初始化，当前位置: {initial_pos}")

        # 5. 可选：主臂 + 接管数据集
        events = None
        teleop_action_processor = robot_action_processor = robot_observation_processor = None
        if cfg.enable_takeover:
            if cfg.display_data:
                init_rerun(session_name="vla_takeover")

            teleop = make_teleoperator_from_config(cfg.teleop)
            teleop.connect()
            logger.info("主臂遥操器连接成功")

            teleop_action_processor, robot_action_processor, robot_observation_processor = (
                make_default_processors()
            )
            dataset = setup_takeover_dataset(
                robot=robot,
                dataset_cfg=cfg.dataset,
                teleop_action_processor=teleop_action_processor,
                robot_observation_processor=robot_observation_processor,
                resume=cfg.resume,
            )
            listener, events = init_keyboard_listener(
                manual_episode_control=True,
                enable_takeover=True,
            )

        # 6. 开始推理循环（可多次接管）
        logger.info("开始双臂推理循环...")
        if cfg.enable_takeover:
            logger.info("VLA 运行中。按 Space 或 T 人工接管，Esc 退出")
        else:
            logger.info("按 Ctrl+C 停止")

        recorded_episodes = 0
        session_start = time.perf_counter()
        video_ctx = VideoEncodingManager(dataset) if dataset is not None else nullcontext()

        with video_ctx:
            while not shutdown_event.is_set():
                if events is not None and events.get("stop_recording"):
                    break
                if cfg.enable_takeover and recorded_episodes >= cfg.dataset.num_episodes:
                    logger.info(f"已达到最大接管条数 {cfg.dataset.num_episodes}，结束会话")
                    break

                infer_kwargs = dict(
                    robot=robot_wrapper,
                    vla_client=vla_client,
                    obs_adapter=obs_adapter,
                    action_adapter=action_adapter,
                    action_limiter=action_limiter,
                    shutdown_event=shutdown_event,
                    task=cfg.task,
                    fps=cfg.fps,
                    duration=cfg.duration,
                    events=events,
                    start_time=session_start,
                )
                if cfg.use_action_chunk:
                    reason = run_inference_loop_with_chunk(
                        **infer_kwargs,
                        chunk_execution_horizon=cfg.chunk_execution_horizon,
                    )
                else:
                    reason = run_inference_loop(**infer_kwargs)

                if reason != STOP_TAKEOVER:
                    if reason == STOP_DURATION:
                        logger.info(f"达到运行时长 {cfg.duration} 秒，停止推理")
                    break

                result = record_takeover_episode(
                    robot=robot,
                    teleop=teleop,
                    dataset=dataset,
                    events=events,
                    fps=dataset.fps,
                    single_task=cfg.dataset.single_task,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    display_data=cfg.display_data,
                    play_sounds=cfg.play_sounds,
                    enable_leader_align=cfg.enable_leader_align,
                    leader_align_ramp_duration_s=cfg.leader_align_ramp_duration_s,
                )
                if result == "saved":
                    recorded_episodes += 1
                if result == "stopped" or events.get("stop_recording") or shutdown_event.is_set():
                    break
                if not wait_for_vla_resume(events, shutdown_event, play_sounds=cfg.play_sounds):
                    break

        if dataset is not None and cfg.dataset.push_to_hub:
            dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

    except Exception as e:
        logger.error(f"推理过程出错: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    finally:
        if listener is not None:
            try:
                listener.stop()
            except Exception as e:
                logger.error(f"停止键盘监听时出错: {e}")

        if teleop is not None:
            try:
                teleop.disconnect()
                logger.info("主臂遥操器已断开连接")
            except Exception as e:
                logger.error(f"断开主臂连接时出错: {e}")

        if "robot" in locals():
            try:
                robot.disconnect()
                logger.info("双臂机器人已断开连接")
            except Exception as e:
                logger.error(f"断开机器人连接时出错: {e}")

        logger.info("推理结束")


def run_inference_loop(
    robot: RobotWrapper,
    vla_client: VLAClient,
    obs_adapter: BimanualVLAObservationAdapter,
    action_adapter: BimanualVLAActionAdapter,
    action_limiter: BimanualVLAActionLimiter | None,
    shutdown_event: Event,
    task: str,
    fps: float,
    duration: float,
    events: dict[str, Any] | None = None,
    start_time: float | None = None,
) -> str:
    """
    双臂单步动作推理循环
    每个时间步都获取新的观察并推理动作
    """
    step_count = 0
    if start_time is None:
        start_time = time.perf_counter()
    action_interval = 1.0 / fps
    reason = STOP_SHUTDOWN

    while True:
        loop_start = time.perf_counter()
        elapsed = loop_start - start_time
        reason = inference_stop_reason(shutdown_event, events, start_time, duration)
        if reason is not None:
            break

        try:
            # 1. 获取双臂机器人观察
            lerobot_obs = robot.get_observation()

            # 2. 转换为 VLA 格式 (12-dim state + 3 images)
            vla_example = obs_adapter.convert(lerobot_obs, task=task)

            # 3. 调用 VLA 服务器获取动作
            vla_actions = vla_client.get_actions(vla_example)

            reason = inference_stop_reason(shutdown_event, events, start_time, duration)
            if reason is not None:
                break

            # 4. 转换为双臂 LeRobot 格式 (取前 12 维)
            lerobot_action = action_adapter.convert(vla_actions)

            # 5. 应用动作限幅
            if action_limiter is not None:
                current_pos = {k: v for k, v in lerobot_obs.items() if k.endswith(".pos")}
                lerobot_action = action_limiter.limit(lerobot_action, current_pos)

            # 6. 发送动作到双臂机器人
            robot.send_action(lerobot_action)

            step_count += 1

            # 日志输出（每 30 步）
            if step_count % 30 == 0:
                logger.info(
                    f"步骤 {step_count} | 已运行 {elapsed:.1f}s | "
                    f"左夹爪={lerobot_action.get('left_gripper.pos', 0):.2f} "
                    f"右夹爪={lerobot_action.get('right_gripper.pos', 0):.2f}"
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
    return reason


def run_inference_loop_with_chunk(
    robot: RobotWrapper,
    vla_client: VLAClient,
    obs_adapter: BimanualVLAObservationAdapter,
    action_adapter: BimanualVLAActionAdapter,
    action_limiter: BimanualVLAActionLimiter | None,
    shutdown_event: Event,
    task: str,
    fps: float,
    duration: float,
    chunk_execution_horizon: int,
    events: dict[str, Any] | None = None,
    start_time: float | None = None,
) -> str:
    """
    双臂 Action Chunk 推理循环
    每隔 chunk_execution_horizon 步获取新的 action chunk，中间使用缓存的动作
    """
    step_count = 0
    chunk_count = 0
    if start_time is None:
        start_time = time.perf_counter()
    action_interval = 1.0 / fps
    reason = STOP_SHUTDOWN

    current_chunk: np.ndarray | None = None
    chunk_step = 0
    last_lerobot_obs: dict[str, Any] | None = None

    while True:
        loop_start = time.perf_counter()
        elapsed = loop_start - start_time
        reason = inference_stop_reason(shutdown_event, events, start_time, duration)
        if reason is not None:
            break

        try:
            # 检查是否需要获取新的 action chunk
            need_new_chunk = (
                current_chunk is None
                or chunk_step >= chunk_execution_horizon
                or chunk_step >= len(current_chunk)
            )

            if need_new_chunk:
                # 1. 获取双臂机器人观察
                lerobot_obs = robot.get_observation()
                last_lerobot_obs = lerobot_obs

                # 2. 转换为 VLA 格式
                vla_example = obs_adapter.convert(lerobot_obs, task=task)

                # 3. 调用 VLA 服务器获取 action chunk
                vla_actions = vla_client.get_actions(vla_example)

                reason = inference_stop_reason(shutdown_event, events, start_time, duration)
                if reason is not None:
                    break

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

            reason = inference_stop_reason(shutdown_event, events, start_time, duration)
            if reason is not None:
                break

            # 6. 发送动作到双臂机器人
            robot.send_action(lerobot_action)

            step_count += 1

            # 日志输出：仅在 chunk 执行完时打印一次，避免误解为“每 50 步重新推理”
            if chunk_step >= min(chunk_execution_horizon, len(current_chunk)):
                logger.info(
                    f"步骤 {step_count} | Chunk #{chunk_count} 完成 {chunk_step}/{len(current_chunk)} 步 | "
                    f"已运行 {elapsed:.1f}s | "
                    f"左夹爪={lerobot_action.get('left_gripper.pos', 0):.2f} "
                    f"右夹爪={lerobot_action.get('right_gripper.pos', 0):.2f}"
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
    return reason


if __name__ == "__main__":
    main()

