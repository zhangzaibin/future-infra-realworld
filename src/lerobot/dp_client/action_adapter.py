#!/usr/bin/env python3
"""
动作格式适配器
将 Diffusion Policy 服务器返回的动作转换为 LeRobot 机器人期望的格式
"""
import numpy as np
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class ActionLimiter:
    """
    动作限幅器
    限制每步动作的最大变化量，防止机械臂突然大幅度跳跃
    """
    
    # SO101 默认关节顺序
    DEFAULT_JOINT_ORDER = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]
    
    # 默认每步最大变化量
    DEFAULT_MAX_DELTA = 5.0
    
    def __init__(
        self,
        joint_order: Optional[List[str]] = None,
    ):
        """
        初始化动作限幅器
        
        Args:
            joint_order: 关节顺序
        """
        self.joint_order = joint_order or self.DEFAULT_JOINT_ORDER
        
        # 上一次的动作
        self._last_action: Optional[Dict[str, float]] = None
        
        logger.info(f"ActionLimiter 初始化，最大变化量: {self.DEFAULT_MAX_DELTA}")
    
    def limit(
        self,
        target_action: Dict[str, float],
        current_pos: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        限制动作变化量
        
        Args:
            target_action: 目标动作
            current_pos: 当前机器人位置（如果提供，使用这个作为基准）
                        如果不提供，使用上一次的动作作为基准
        
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
            logger.info(f"ActionLimiter: 首次调用，初始化位置")
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
            if abs(delta) > self.DEFAULT_MAX_DELTA:
                clipped_delta = self.DEFAULT_MAX_DELTA if delta > 0 else -self.DEFAULT_MAX_DELTA
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
    
    def reset(self, initial_pos: Optional[Dict[str, float]] = None):
        """
        重置限幅器状态
        
        Args:
            initial_pos: 初始位置（如果提供）
        """
        self._last_action = initial_pos.copy() if initial_pos else None
        logger.info("ActionLimiter 已重置")
    
    def set_last_action(self, action: Dict[str, float]):
        """
        设置上一次动作（用于初始化）
        
        Args:
            action: 动作字典
        """
        self._last_action = action.copy()


class ActionAdapter:
    """
    将 DP 服务器返回的动作转换为 LeRobot 格式
    
    DP 服务器返回格式:
        np.ndarray (action_dim,) 或 (chunk_size, action_dim)
        
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
    DEFAULT_JOINT_ORDER = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]
    
    def __init__(
        self,
        joint_order: Optional[List[str]] = None,
        use_action_chunk: bool = False,
        chunk_index: int = 0,
    ):
        """
        初始化动作适配器
        
        Args:
            joint_order: 关节名称顺序列表，默认使用 SO101 顺序
            use_action_chunk: 是否使用 action chunk 模式
            chunk_index: 当使用 action chunk 时，从 chunk 中取第几步的动作
        """
        self.joint_order = joint_order or self.DEFAULT_JOINT_ORDER
        self.use_action_chunk = use_action_chunk
        self.chunk_index = chunk_index
        
        # 用于 action chunk 模式的缓存
        self._action_chunk: Optional[np.ndarray] = None
        self._chunk_step: int = 0
        
        logger.info(f"ActionAdapter 初始化:")
        logger.info(f"  关节顺序: {self.joint_order}")
        logger.info(f"  Action chunk 模式: {self.use_action_chunk}")
    
    def convert(self, dp_action: np.ndarray) -> Dict[str, float]:
        """
        将 DP 动作转换为 LeRobot 格式
        
        Args:
            dp_action: DP 服务器返回的动作数组
                      - 单步: shape (action_dim,)
                      - Chunk: shape (chunk_size, action_dim)
                      
        Returns:
            LeRobot 格式的动作字典
        """
        # 处理 action chunk
        if len(dp_action.shape) == 2:
            # 这是一个 action chunk
            if self.use_action_chunk:
                # 存储整个 chunk，返回当前步的动作
                self._action_chunk = dp_action
                self._chunk_step = 0
                action_values = dp_action[self.chunk_index]
            else:
                # 只使用第一步的动作
                action_values = dp_action[0]
        elif len(dp_action.shape) == 1:
            # 单步动作
            action_values = dp_action
        else:
            raise ValueError(f"不支持的动作形状: {dp_action.shape}")
        
        # 检查维度匹配
        if len(action_values) != len(self.joint_order):
            raise ValueError(
                f"动作维度 ({len(action_values)}) 与关节数量 ({len(self.joint_order)}) 不匹配"
            )
        
        # 转换为字典格式
        action_dict = {}
        for i, joint_name in enumerate(self.joint_order):
            key = f"{joint_name}.pos"
            action_dict[key] = float(action_values[i])
        
        logger.debug(f"动作转换完成: {action_dict}")
        return action_dict
    
    def get_next_chunk_action(self) -> Optional[Dict[str, float]]:
        """
        获取 action chunk 中的下一步动作
        
        Returns:
            下一步的动作字典，如果 chunk 已用完则返回 None
        """
        if self._action_chunk is None:
            return None
        
        self._chunk_step += 1
        
        if self._chunk_step >= len(self._action_chunk):
            # Chunk 用完了
            self._action_chunk = None
            self._chunk_step = 0
            return None
        
        action_values = self._action_chunk[self._chunk_step]
        action_dict = {}
        for i, joint_name in enumerate(self.joint_order):
            key = f"{joint_name}.pos"
            action_dict[key] = float(action_values[i])
        
        return action_dict
    
    def has_remaining_chunk_actions(self) -> bool:
        """
        检查是否还有剩余的 chunk 动作
        
        Returns:
            True 如果还有剩余动作
        """
        if self._action_chunk is None:
            return False
        return self._chunk_step < len(self._action_chunk) - 1
    
    def reset_chunk(self):
        """重置 action chunk 状态"""
        self._action_chunk = None
        self._chunk_step = 0


class ActionChunkExecutor:
    """
    Action Chunk 执行器
    管理 action chunk 的执行，支持在执行过程中获取新的 chunk
    """
    
    def __init__(
        self,
        action_adapter: ActionAdapter,
        chunk_execution_horizon: int = 8,
    ):
        """
        初始化 Action Chunk 执行器
        
        Args:
            action_adapter: 动作适配器实例
            chunk_execution_horizon: 每个 chunk 执行多少步后获取新的 chunk
        """
        self.action_adapter = action_adapter
        self.chunk_execution_horizon = chunk_execution_horizon
        
        self._current_chunk: Optional[np.ndarray] = None
        self._chunk_position: int = 0
        self._need_new_chunk: bool = True
        
        logger.info(f"ActionChunkExecutor 初始化:")
        logger.info(f"  Chunk 执行范围: {self.chunk_execution_horizon}")
    
    def update_chunk(self, dp_action: np.ndarray):
        """
        更新当前的 action chunk
        
        Args:
            dp_action: 新的 action chunk
        """
        if len(dp_action.shape) == 1:
            # 单步动作，扩展为 chunk
            dp_action = dp_action.reshape(1, -1)
        
        self._current_chunk = dp_action
        self._chunk_position = 0
        self._need_new_chunk = False
        
        logger.debug(f"更新 action chunk: shape={dp_action.shape}")
    
    def get_action(self) -> Optional[Dict[str, float]]:
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

