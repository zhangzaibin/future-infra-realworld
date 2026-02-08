#!/usr/bin/env python3
"""
VLA 动作格式适配器
将 VLA 服务器返回的动作转换为 LeRobot 机器人期望的格式
"""
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VLAActionAdapter:
    """
    将 VLA 服务器返回的动作转换为 LeRobot 格式

    VLA 服务器返回格式 (参考 so101policy.py SO101Outputs):
        np.ndarray (chunk_size, 6) - 6 维动作 (5 关节 + 1 夹爪)

    LeRobot SO101 动作格式:
    {
        "shoulder_pan.pos": float,
        "shoulder_lift.pos": float,
        "elbow_flex.pos": float,
        "wrist_flex.pos": float,
        "wrist_roll.pos": float,
        "gripper.pos": float,
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

    # 使用 action chunk 中的第几步（0-indexed）
    chunk_index: int = 0

    def __post_init__(self):
        logger.info(f"VLAActionAdapter 初始化:")
        logger.info(f"  关节顺序: {self.joint_order}")
        logger.info(f"  Chunk 索引: {self.chunk_index}")

    def convert(self, vla_actions: np.ndarray) -> dict[str, float]:
        """
        将 VLA 动作转换为 LeRobot 格式

        Args:
            vla_actions: VLA 服务器返回的动作数组
                - 单步: shape (action_dim,)
                - Chunk: shape (chunk_size, action_dim)

        Returns:
            LeRobot 格式的动作字典
        """
        # 处理 action chunk
        if len(vla_actions.shape) == 2:
            # 这是一个 action chunk，取指定索引的动作
            if self.chunk_index >= len(vla_actions):
                logger.warning(
                    f"chunk_index ({self.chunk_index}) >= chunk_size ({len(vla_actions)}), "
                    f"使用最后一步"
                )
                action_values = vla_actions[-1]
            else:
                action_values = vla_actions[self.chunk_index]
        elif len(vla_actions.shape) == 1:
            # 单步动作
            action_values = vla_actions
        else:
            raise ValueError(f"不支持的动作形状: {vla_actions.shape}")

        # 检查维度匹配
        if len(action_values) < len(self.joint_order):
            raise ValueError(
                f"动作维度 ({len(action_values)}) 小于关节数量 ({len(self.joint_order)})"
            )

        # 转换为字典格式
        action_dict = {}
        for i, joint_name in enumerate(self.joint_order):
            key = f"{joint_name}.pos"
            action_dict[key] = float(action_values[i])

        logger.debug(f"动作转换完成: {action_dict}")
        return action_dict

    def convert_chunk(self, vla_actions: np.ndarray) -> list[dict[str, float]]:
        """
        将整个 action chunk 转换为 LeRobot 格式列表

        Args:
            vla_actions: VLA 服务器返回的动作数组 (chunk_size, action_dim)

        Returns:
            LeRobot 格式的动作字典列表
        """
        if len(vla_actions.shape) == 1:
            vla_actions = vla_actions.reshape(1, -1)

        actions = []
        for action_values in vla_actions:
            action_dict = {}
            for i, joint_name in enumerate(self.joint_order):
                if i < len(action_values):
                    key = f"{joint_name}.pos"
                    action_dict[key] = float(action_values[i])
            actions.append(action_dict)

        return actions


@dataclass
class VLAActionLimiter:
    """
    VLA 动作限幅器
    限制每步动作的最大变化量，防止机械臂突然大幅度跳跃
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

    # 每步最大变化量（度）
    max_delta: float = 5.0

    def __post_init__(self):
        # 上一次的动作
        self._last_action: dict[str, float] | None = None
        logger.info(f"VLAActionLimiter 初始化，最大变化量: {self.max_delta}")

    def limit(
        self,
        target_action: dict[str, float],
        current_pos: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """
        限制动作变化量

        Args:
            target_action: 目标动作
            current_pos: 当前机器人位置（如果提供，使用这个作为基准）

        Returns:
            限幅后的动作
        """
        # 确定基准位置
        if current_pos is not None:
            base_pos = current_pos
        elif self._last_action is not None:
            base_pos = self._last_action
        else:
            # 第一次调用，没有基准，直接使用目标动作
            self._last_action = target_action.copy()
            logger.info("VLAActionLimiter: 首次调用，初始化位置")
            return target_action

        limited_action = {}
        clipped_joints = []

        for joint_name in self.joint_order:
            key = f"{joint_name}.pos"

            if key not in target_action:
                continue

            target_val = target_action[key]
            base_val = base_pos.get(key, target_val)

            # 计算变化量
            delta = target_val - base_val

            # 限幅
            if abs(delta) > self.max_delta:
                clipped_delta = self.max_delta if delta > 0 else -self.max_delta
                limited_val = base_val + clipped_delta
                clipped_joints.append(f"{joint_name}({delta:.1f}->{clipped_delta:.1f})")
            else:
                limited_val = target_val

            limited_action[key] = limited_val

        # 记录被限幅的关节
        if clipped_joints:
            logger.debug(f"限幅关节: {', '.join(clipped_joints)}")

        # 更新上一次动作
        self._last_action = limited_action.copy()

        return limited_action

    def reset(self, initial_pos: dict[str, float] | None = None):
        """
        重置限幅器状态

        Args:
            initial_pos: 初始位置（如果提供）
        """
        self._last_action = initial_pos.copy() if initial_pos else None
        logger.info("VLAActionLimiter 已重置")

    def set_last_action(self, action: dict[str, float]):
        """
        设置上一次动作（用于初始化）

        Args:
            action: 动作字典
        """
        self._last_action = action.copy()


@dataclass
class BimanualVLAActionAdapter:
    """
    双臂 VLA 动作适配器
    将 VLA 服务器返回的 12 维动作 (取 32 维的前 12 维) 转换为双臂 LeRobot 格式

    VLA 服务器返回:
        np.ndarray (chunk_size, 32) - 取前 12 维 (left 6 + right 6)

    LeRobot BiSO101 动作格式:
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
    }
    """

    # 单臂关节顺序
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

    # 从 32 维输出中截取的维度数
    action_dim: int = 12

    # 使用 action chunk 中的第几步（0-indexed）
    chunk_index: int = 0

    def __post_init__(self):
        # 构建双臂完整关节顺序
        self.bimanual_joint_order = (
            [f"left_{j}" for j in self.single_arm_joint_order]
            + [f"right_{j}" for j in self.single_arm_joint_order]
        )
        logger.info(f"BimanualVLAActionAdapter 初始化:")
        logger.info(f"  双臂关节顺序 ({len(self.bimanual_joint_order)} 维): {self.bimanual_joint_order}")
        logger.info(f"  截取动作维度: {self.action_dim}")
        logger.info(f"  Chunk 索引: {self.chunk_index}")

    def convert(self, vla_actions: np.ndarray) -> dict[str, float]:
        """
        将 VLA 动作转换为双臂 LeRobot 格式

        Args:
            vla_actions: VLA 服务器返回的动作数组
                - 单步: shape (action_dim,) 或 (32,)
                - Chunk: shape (chunk_size, action_dim) 或 (chunk_size, 32)

        Returns:
            双臂 LeRobot 格式的动作字典
        """
        # 处理 action chunk
        if len(vla_actions.shape) == 2:
            if self.chunk_index >= len(vla_actions):
                logger.warning(
                    f"chunk_index ({self.chunk_index}) >= chunk_size ({len(vla_actions)}), "
                    f"使用最后一步"
                )
                action_values = vla_actions[-1]
            else:
                action_values = vla_actions[self.chunk_index]
        elif len(vla_actions.shape) == 1:
            action_values = vla_actions
        else:
            raise ValueError(f"不支持的动作形状: {vla_actions.shape}")

        # 截取前 action_dim 维 (从 32 维截取前 12 维)
        action_values = action_values[: self.action_dim]

        # 检查维度匹配
        if len(action_values) < len(self.bimanual_joint_order):
            raise ValueError(
                f"动作维度 ({len(action_values)}) 小于关节数量 ({len(self.bimanual_joint_order)})"
            )

        # 转换为字典格式
        action_dict = {}
        for i, joint_name in enumerate(self.bimanual_joint_order):
            key = f"{joint_name}.pos"
            action_dict[key] = float(action_values[i])

        logger.debug(f"双臂动作转换完成: {action_dict}")
        return action_dict

    def convert_chunk(self, vla_actions: np.ndarray) -> list[dict[str, float]]:
        """
        将整个 action chunk 转换为双臂 LeRobot 格式列表

        Args:
            vla_actions: (chunk_size, 32) 或 (chunk_size, action_dim)

        Returns:
            双臂 LeRobot 格式的动作字典列表
        """
        if len(vla_actions.shape) == 1:
            vla_actions = vla_actions.reshape(1, -1)

        actions = []
        for action_values in vla_actions:
            # 截取前 action_dim 维
            action_values = action_values[: self.action_dim]
            action_dict = {}
            for i, joint_name in enumerate(self.bimanual_joint_order):
                if i < len(action_values):
                    key = f"{joint_name}.pos"
                    action_dict[key] = float(action_values[i])
            actions.append(action_dict)

        return actions


@dataclass
class BimanualVLAActionLimiter:
    """
    双臂 VLA 动作限幅器
    限制每步动作的最大变化量，防止双臂突然大幅度跳跃
    """

    # 单臂关节顺序
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

    # 每步最大变化量（度）
    max_delta: float = 5.0

    def __post_init__(self):
        self.bimanual_joint_order = (
            [f"left_{j}" for j in self.single_arm_joint_order]
            + [f"right_{j}" for j in self.single_arm_joint_order]
        )
        self._last_action: dict[str, float] | None = None
        logger.info(f"BimanualVLAActionLimiter 初始化，最大变化量: {self.max_delta}")

    def limit(
        self,
        target_action: dict[str, float],
        current_pos: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """限制双臂动作变化量"""
        if current_pos is not None:
            base_pos = current_pos
        elif self._last_action is not None:
            base_pos = self._last_action
        else:
            self._last_action = target_action.copy()
            logger.info("BimanualVLAActionLimiter: 首次调用，初始化位置")
            return target_action

        limited_action = {}
        clipped_joints = []

        for joint_name in self.bimanual_joint_order:
            key = f"{joint_name}.pos"

            if key not in target_action:
                continue

            target_val = target_action[key]
            base_val = base_pos.get(key, target_val)

            delta = target_val - base_val

            if abs(delta) > self.max_delta:
                clipped_delta = self.max_delta if delta > 0 else -self.max_delta
                limited_val = base_val + clipped_delta
                clipped_joints.append(f"{joint_name}({delta:.1f}->{clipped_delta:.1f})")
            else:
                limited_val = target_val

            limited_action[key] = limited_val

        if clipped_joints:
            logger.debug(f"双臂限幅关节: {', '.join(clipped_joints)}")

        self._last_action = limited_action.copy()
        return limited_action

    def reset(self, initial_pos: dict[str, float] | None = None):
        """重置限幅器状态"""
        self._last_action = initial_pos.copy() if initial_pos else None
        logger.info("BimanualVLAActionLimiter 已重置")

    def set_last_action(self, action: dict[str, float]):
        """设置上一次动作（用于初始化）"""
        self._last_action = action.copy()


@dataclass
class VLAActionChunkExecutor:
    """
    VLA Action Chunk 执行器
    管理 action chunk 的执行，支持时序执行
    """

    action_adapter: VLAActionAdapter

    # 每个 chunk 执行多少步后获取新的 chunk
    chunk_execution_horizon: int = 8

    def __post_init__(self):
        self._current_chunk: np.ndarray | None = None
        self._chunk_position: int = 0
        self._need_new_chunk: bool = True

        logger.info(f"VLAActionChunkExecutor 初始化:")
        logger.info(f"  Chunk 执行范围: {self.chunk_execution_horizon}")

    def update_chunk(self, vla_actions: np.ndarray):
        """
        更新当前的 action chunk

        Args:
            vla_actions: 新的 action chunk
        """
        if len(vla_actions.shape) == 1:
            # 单步动作，扩展为 chunk
            vla_actions = vla_actions.reshape(1, -1)

        self._current_chunk = vla_actions
        self._chunk_position = 0
        self._need_new_chunk = False

        logger.debug(f"更新 action chunk: shape={vla_actions.shape}")

    def get_action(self) -> dict[str, float] | None:
        """
        获取下一个要执行的动作

        Returns:
            动作字典，如果没有可用动作则返回 None
        """
        if self._current_chunk is None:
            return None

        if self._chunk_position >= len(self._current_chunk):
            self._need_new_chunk = True
            return None

        action_values = self._current_chunk[self._chunk_position]
        self._chunk_position += 1

        # 检查是否需要新的 chunk
        if self._chunk_position >= self.chunk_execution_horizon:
            self._need_new_chunk = True

        # 转换为字典格式
        self.action_adapter.chunk_index = 0  # 使用第一步
        return self.action_adapter.convert(action_values)

    def needs_new_chunk(self) -> bool:
        """
        检查是否需要新的 action chunk

        Returns:
            True 如果需要新的 chunk
        """
        return self._need_new_chunk

    def reset(self):
        """重置执行器状态"""
        self._current_chunk = None
        self._chunk_position = 0
        self._need_new_chunk = True




