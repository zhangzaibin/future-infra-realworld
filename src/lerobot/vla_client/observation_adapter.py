#!/usr/bin/env python3
"""
VLA 观察格式适配器
将 LeRobot 机器人的观察格式转换为 VLA 服务器期望的格式
"""
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VLAObservationAdapter:
    """
    将 LeRobot 观察格式转换为 VLA 服务器期望的格式

    LeRobot SO101 观察格式:
    {
        "shoulder_pan.pos": float,
        "shoulder_lift.pos": float,
        "elbow_flex.pos": float,
        "wrist_flex.pos": float,
        "wrist_roll.pos": float,
        "gripper.pos": float,
        "front": np.ndarray (H, W, 3),  # BGR, uint8 [0, 255]
        "wrist": np.ndarray (H, W, 3),  # BGR, uint8 [0, 255] (可选)
    }

    VLA 服务器期望格式 (参考 so101policy.py):
    {
        "observation/state": np.ndarray (6,),  # 关节状态
        "observation/image": np.ndarray (H, W, 3),  # 前置摄像头 RGB, uint8
        "observation/wrist_image": np.ndarray (H, W, 3),  # 腕部摄像头 RGB, uint8
        "task": str,  # 任务描述
    }
    """

    # SO101 默认关节顺序
    joint_order: list[str] = field(
        default_factory=lambda: [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]
    )

    # 相机键名映射
    front_camera_key: str = "global"  # 主相机（全局视角）
    wrist_camera_key: str | None = "wrist"  # 腕部相机

    # 目标图像尺寸 (H, W)，如果为 None 则不调整大小
    target_image_size: tuple[int, int] | None = None

    # 默认任务描述
    default_task: str = "完成操作任务"

    def __post_init__(self):
        logger.info(f"VLAObservationAdapter 初始化:")
        logger.info(f"  关节顺序: {self.joint_order}")
        logger.info(f"  前置相机键: {self.front_camera_key}")
        logger.info(f"  腕部相机键: {self.wrist_camera_key}")
        logger.info(f"  目标图像尺寸: {self.target_image_size}")

    def convert(
        self,
        lerobot_obs: dict[str, Any],
        task: str | None = None,
    ) -> dict[str, Any]:
        """
        将 LeRobot 观察转换为 VLA 格式

        Args:
            lerobot_obs: LeRobot 机器人的原始观察
            task: 任务描述，如果不提供则使用默认任务

        Returns:
            VLA 服务器期望格式的样本字典
        """
        # 转换关节状态
        state = self._convert_state(lerobot_obs)

        # 转换图像
        front_image = self._convert_image(lerobot_obs, self.front_camera_key)

        # 腕部图像（可选）
        if self.wrist_camera_key and self.wrist_camera_key in lerobot_obs:
            wrist_image = self._convert_image(lerobot_obs, self.wrist_camera_key)
        else:
            # 使用零图像作为占位符
            wrist_image = np.zeros_like(front_image)

        # 构建 VLA 样本
        vla_example = {
            "observation/state": state,
            "observation/image": front_image,
            "observation/wrist_image": wrist_image,
            "task": task or self.default_task,
        }

        logger.debug(
            f"观察转换完成: state shape={state.shape}, "
            f"image shape={front_image.shape}, "
            f"wrist_image shape={wrist_image.shape}"
        )
        return vla_example

    def _convert_state(self, lerobot_obs: dict[str, Any]) -> np.ndarray:
        """
        提取并排序关节状态

        Args:
            lerobot_obs: LeRobot 观察字典

        Returns:
            按指定顺序排列的关节状态数组 (6,)
        """
        positions = []
        for joint_name in self.joint_order:
            key = f"{joint_name}.pos"
            if key not in lerobot_obs:
                raise KeyError(
                    f"关节 '{key}' 在观察中不存在。可用的键: {list(lerobot_obs.keys())}"
                )
            positions.append(float(lerobot_obs[key]))

        return np.array(positions, dtype=np.float32)

    def _convert_image(
        self,
        lerobot_obs: dict[str, Any],
        camera_key: str,
    ) -> np.ndarray:
        """
        转换图像格式

        LeRobot: (H, W, 3) BGR uint8 [0, 255]
        VLA: (H, W, 3) RGB uint8 [0, 255]
        """
        if camera_key not in lerobot_obs:
            raise KeyError(
                f"相机 '{camera_key}' 在观察中不存在。可用的键: {list(lerobot_obs.keys())}"
            )

        image = lerobot_obs[camera_key]

        # 确保是 numpy array
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        # 确保是 uint8
        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating):
                image = (255 * image).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        # BGR -> RGB（OpenCV 默认使用 BGR）
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = image[:, :, ::-1].copy()  # BGR to RGB

        # 调整大小（如果需要）
        if self.target_image_size is not None:
            image = self._resize_image(image, self.target_image_size)

        return image

    def _resize_image(
        self,
        image: np.ndarray,
        target_size: tuple[int, int],
    ) -> np.ndarray:
        """
        调整图像大小

        Args:
            image: 输入图像 (H, W, C)
            target_size: 目标尺寸 (H, W)

        Returns:
            调整大小后的图像
        """
        try:
            import cv2

            target_h, target_w = target_size
            resized = cv2.resize(
                image, (target_w, target_h), interpolation=cv2.INTER_LINEAR
            )
            return resized
        except ImportError:
            logger.warning("cv2 未安装，无法调整图像大小")
            return image


@dataclass
class DualCameraVLAObservationAdapter(VLAObservationAdapter):
    """
    双摄像头 VLA 观察适配器
    支持前置和腕部两个摄像头
    """

    front_camera_key: str = "front"
    wrist_camera_key: str | None = "wrist"

    def __post_init__(self):
        super().__post_init__()
        logger.info(f"DualCameraVLAObservationAdapter 初始化完成")
        logger.info(f"  前置相机: {self.front_camera_key}")
        logger.info(f"  腕部相机: {self.wrist_camera_key}")


@dataclass
class BimanualVLAObservationAdapter:
    """
    双臂 VLA 观察格式适配器
    将 BiSO101Follower 的观察格式转换为 VLA 服务器期望的格式

    BiSO101Follower 观察格式:
    {
        "left_shoulder_pan.pos": float,
        "left_shoulder_lift.pos": float,
        "left_elbow_flex.pos": float,
        "left_wrist_flex.pos": float,
        "left_wrist_roll.pos": float,
        "left_gripper.pos": float,
        "right_shoulder_pan.pos": float,
        "right_shoulder_lift.pos": float,
        "right_elbow_flex.pos": float,
        "right_wrist_flex.pos": float,
        "right_wrist_roll.pos": float,
        "right_gripper.pos": float,
        "global": np.ndarray (H, W, 3),
        "left_wrist": np.ndarray (H, W, 3),
        "right_wrist": np.ndarray (H, W, 3),
    }

    VLA 服务器期望格式:
    {
        "observation/state": np.ndarray (12,),  # left 6 + right 6
        "observation/image": np.ndarray (H, W, 3),  # global camera RGB
        "observation/wrist_image": np.ndarray (H, W, 3),  # left wrist camera RGB
        "observation/right_wrist_image": np.ndarray (H, W, 3),  # right wrist camera RGB
        "task": str,
    }
    """

    # 单臂关节顺序 (每臂 6 自由度)
    single_arm_joint_order: list[str] = field(
        default_factory=lambda: [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]
    )

    # 相机键名
    global_camera_key: str = "global"
    left_wrist_camera_key: str = "left_wrist"
    right_wrist_camera_key: str = "right_wrist"

    # 目标图像尺寸 (H, W)，如果为 None 则不调整大小
    target_image_size: tuple[int, int] | None = None

    # 默认任务描述
    default_task: str = "完成操作任务"

    def __post_init__(self):
        # 构建双臂完整关节顺序: left first, then right
        self.bimanual_joint_order = (
            [f"left_{j}" for j in self.single_arm_joint_order]
            + [f"right_{j}" for j in self.single_arm_joint_order]
        )
        logger.info(f"BimanualVLAObservationAdapter 初始化:")
        logger.info(f"  双臂关节顺序 ({len(self.bimanual_joint_order)} 维): {self.bimanual_joint_order}")
        logger.info(f"  全局相机: {self.global_camera_key}")
        logger.info(f"  左手腕相机: {self.left_wrist_camera_key}")
        logger.info(f"  右手腕相机: {self.right_wrist_camera_key}")
        logger.info(f"  目标图像尺寸: {self.target_image_size}")

    def convert(
        self,
        lerobot_obs: dict[str, Any],
        task: str | None = None,
    ) -> dict[str, Any]:
        """
        将双臂 LeRobot 观察转换为 VLA 格式

        Args:
            lerobot_obs: BiSO101Follower 的原始观察
            task: 任务描述

        Returns:
            VLA 服务器期望格式的样本字典
        """
        # 转换关节状态 (12 维)
        state = self._convert_state(lerobot_obs)

        # 转换图像
        global_image = self._convert_image(lerobot_obs, self.global_camera_key)

        left_wrist_image = self._convert_image(lerobot_obs, self.left_wrist_camera_key) \
            if self.left_wrist_camera_key in lerobot_obs else np.zeros_like(global_image)

        right_wrist_image = self._convert_image(lerobot_obs, self.right_wrist_camera_key) \
            if self.right_wrist_camera_key in lerobot_obs else np.zeros_like(global_image)

        # 构建 VLA 样本
        vla_example = {
            "observation/state": state,
            "observation/image": global_image,
            "observation/wrist_image": left_wrist_image,
            "observation/right_wrist_image": right_wrist_image,
            "task": task or self.default_task,
        }

        logger.debug(
            f"双臂观察转换完成: state shape={state.shape}, "
            f"global_image shape={global_image.shape}, "
            f"left_wrist shape={left_wrist_image.shape}, "
            f"right_wrist shape={right_wrist_image.shape}"
        )
        return vla_example

    def _convert_state(self, lerobot_obs: dict[str, Any]) -> np.ndarray:
        """
        提取并排序双臂关节状态

        Returns:
            按 [left_6, right_6] 顺序排列的关节状态数组 (12,)
        """
        positions = []
        for joint_name in self.bimanual_joint_order:
            key = f"{joint_name}.pos"
            if key not in lerobot_obs:
                raise KeyError(
                    f"关节 '{key}' 在观察中不存在。可用的键: {list(lerobot_obs.keys())}"
                )
            positions.append(float(lerobot_obs[key]))

        return np.array(positions, dtype=np.float32)

    def _convert_image(
        self,
        lerobot_obs: dict[str, Any],
        camera_key: str,
    ) -> np.ndarray:
        """
        转换图像格式

        LeRobot: (H, W, 3) BGR uint8 [0, 255]
        VLA: (H, W, 3) RGB uint8 [0, 255]
        """
        if camera_key not in lerobot_obs:
            raise KeyError(
                f"相机 '{camera_key}' 在观察中不存在。可用的键: {list(lerobot_obs.keys())}"
            )

        image = lerobot_obs[camera_key]

        if not isinstance(image, np.ndarray):
            image = np.array(image)

        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating):
                image = (255 * image).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        # BGR -> RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = image[:, :, ::-1].copy()

        # 调整大小
        if self.target_image_size is not None:
            image = self._resize_image(image, self.target_image_size)

        return image

    def _resize_image(
        self,
        image: np.ndarray,
        target_size: tuple[int, int],
    ) -> np.ndarray:
        """调整图像大小"""
        try:
            import cv2
            target_h, target_w = target_size
            return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        except ImportError:
            logger.warning("cv2 未安装，无法调整图像大小")
            return image

