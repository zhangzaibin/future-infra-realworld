#!/usr/bin/env python3
"""
观察格式适配器
将 LeRobot 机器人的观察格式转换为 Diffusion Policy 服务器期望的格式
"""
import numpy as np
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class ObservationAdapter:
    """
    将 LeRobot 观察格式转换为 DP 服务器期望的格式
    
    LeRobot SO101 观察格式:
    {
        "shoulder_pan.pos": float,
        "shoulder_lift.pos": float,
        "elbow_flex.pos": float,
        "wrist_flex.pos": float,
        "wrist_roll.pos": float,
        "gripper.pos": float,
        "front": np.ndarray (H, W, 3),  # BGR, uint8 [0, 255]
        "side": np.ndarray (H, W, 3),   # BGR, uint8 [0, 255]
    }
    
    DP 服务器期望格式:
    {
        "head_cam": np.ndarray (C, H, W),  # RGB, float32 [0, 1]
        "agent_pos": np.ndarray (n,),       # 关节位置
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
        camera_key: str = "front",
        joint_order: Optional[List[str]] = None,
        target_image_size: Optional[tuple] = None,
        image_key_in_dp: str = "head_cam",
        state_key_in_dp: str = "agent_pos",
    ):
        """
        初始化观察适配器
        
        Args:
            camera_key: LeRobot 观察中使用的相机名称 (如 "front", "side")
            joint_order: 关节名称顺序列表，默认使用 SO101 顺序
            target_image_size: 目标图像尺寸 (H, W)，如果为 None 则不调整大小
            image_key_in_dp: DP 服务器期望的图像键名
            state_key_in_dp: DP 服务器期望的状态键名
        """
        self.camera_key = camera_key
        self.joint_order = joint_order or self.DEFAULT_JOINT_ORDER
        self.target_image_size = target_image_size
        self.image_key_in_dp = image_key_in_dp
        self.state_key_in_dp = state_key_in_dp
        
        logger.info(f"ObservationAdapter 初始化:")
        logger.info(f"  相机键: {self.camera_key}")
        logger.info(f"  关节顺序: {self.joint_order}")
        logger.info(f"  目标图像尺寸: {self.target_image_size}")
    
    def convert(self, lerobot_obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        将 LeRobot 观察转换为 DP 格式
        
        Args:
            lerobot_obs: LeRobot 机器人的原始观察
            
        Returns:
            DP 服务器期望格式的观察字典
        """
        # 转换图像
        image = self._convert_image(lerobot_obs)
        
        # 转换关节位置
        agent_pos = self._convert_joint_positions(lerobot_obs)
        
        dp_obs = {
            self.image_key_in_dp: image,
            self.state_key_in_dp: agent_pos,
        }
        
        logger.debug(f"观察转换完成: image shape={image.shape}, agent_pos shape={agent_pos.shape}")
        return dp_obs
    
    def _convert_image(self, lerobot_obs: Dict[str, Any]) -> np.ndarray:
        """
        转换图像格式
        
        LeRobot: (H, W, 3) BGR uint8 [0, 255]
        DP: (3, H, W) RGB float32 [0, 1]
        """
        if self.camera_key not in lerobot_obs:
            raise KeyError(f"相机 '{self.camera_key}' 在观察中不存在。可用的键: {list(lerobot_obs.keys())}")
        
        image = lerobot_obs[self.camera_key]
        
        # 确保是 numpy array
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # BGR -> RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = image[:, :, ::-1].copy()  # BGR to RGB
        
        # 调整大小（如果需要）
        if self.target_image_size is not None:
            image = self._resize_image(image, self.target_image_size)
        
        # (H, W, C) -> (C, H, W)
        image = np.transpose(image, (2, 0, 1))
        
        # uint8 [0, 255] -> float32 [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def _resize_image(self, image: np.ndarray, target_size: tuple) -> np.ndarray:
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
            resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            return resized
        except ImportError:
            logger.warning("cv2 未安装，无法调整图像大小")
            return image
    
    def _convert_joint_positions(self, lerobot_obs: Dict[str, Any]) -> np.ndarray:
        """
        提取并排序关节位置
        
        Args:
            lerobot_obs: LeRobot 观察字典
            
        Returns:
            按指定顺序排列的关节位置数组
        """
        positions = []
        for joint_name in self.joint_order:
            key = f"{joint_name}.pos"
            if key not in lerobot_obs:
                raise KeyError(f"关节 '{key}' 在观察中不存在。可用的键: {list(lerobot_obs.keys())}")
            positions.append(float(lerobot_obs[key]))
        
        return np.array(positions, dtype=np.float32)


class MultiCameraObservationAdapter(ObservationAdapter):
    """
    多相机观察适配器
    支持将多个相机的图像发送到 DP 服务器
    """
    
    def __init__(
        self,
        camera_mapping: Dict[str, str],
        joint_order: Optional[List[str]] = None,
        target_image_size: Optional[tuple] = None,
        state_key_in_dp: str = "agent_pos",
    ):
        """
        初始化多相机观察适配器
        
        Args:
            camera_mapping: 相机映射字典 {lerobot_key: dp_key}
                           例如: {"front": "head_cam", "side": "side_cam"}
            joint_order: 关节名称顺序列表
            target_image_size: 目标图像尺寸 (H, W)
            state_key_in_dp: DP 服务器期望的状态键名
        """
        # 使用第一个相机作为主相机初始化父类
        first_camera = list(camera_mapping.keys())[0]
        first_dp_key = camera_mapping[first_camera]
        super().__init__(
            camera_key=first_camera,
            joint_order=joint_order,
            target_image_size=target_image_size,
            image_key_in_dp=first_dp_key,
            state_key_in_dp=state_key_in_dp,
        )
        self.camera_mapping = camera_mapping
        
        logger.info(f"MultiCameraObservationAdapter 初始化:")
        logger.info(f"  相机映射: {self.camera_mapping}")
    
    def convert(self, lerobot_obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        将 LeRobot 观察转换为 DP 格式（多相机版本）
        """
        dp_obs = {}
        
        # 转换所有相机图像
        for lerobot_key, dp_key in self.camera_mapping.items():
            self.camera_key = lerobot_key
            self.image_key_in_dp = dp_key
            image = self._convert_image(lerobot_obs)
            dp_obs[dp_key] = image
        
        # 转换关节位置
        agent_pos = self._convert_joint_positions(lerobot_obs)
        dp_obs[self.state_key_in_dp] = agent_pos
        
        return dp_obs





