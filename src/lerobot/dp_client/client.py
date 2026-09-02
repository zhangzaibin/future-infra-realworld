#!/usr/bin/env python3
"""
Diffusion Policy 客户端
通过 HTTP 请求与 server_dp 通信，提供与本地 DP 类相同的接口
"""
import requests
import base64
import pickle
import numpy as np
import logging
from typing import Optional, Dict

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DPClient:
    """Diffusion Policy 客户端类，通过 HTTP 与服务器通信"""

    def __init__(self, agent_id: int, server_url: str = "http://localhost:20020"):
        """
        初始化客户端
        
        Args:
            agent_id: Agent ID
            server_url: 服务器 URL，默认为 "http://localhost:20020"
        """
        self.agent_id = agent_id
        self.server_url = server_url.rstrip('/')
        self.pending_obs: Optional[Dict[str, np.ndarray]] = None

        # 检查服务器健康状态
        self._check_health()

    def _check_health(self):
        """检查服务器健康状态"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            response.raise_for_status()
            health_data = response.json()
            if not health_data.get("models_loaded", False):
                logger.warning("服务器模型未加载")
            if self.agent_id not in health_data.get("loaded_agents", []):
                logger.warning(f"Agent {self.agent_id} 的模型可能未在服务器上加载")
        except requests.exceptions.RequestException as e:
            logger.error(f"无法连接到服务器 {self.server_url}: {e}")
            raise

    def _serialize_observation(self, observation: Dict[str, np.ndarray]) -> str:
        """
        序列化观察数据为 base64 编码的 pickle 字符串
        
        Args:
            observation: 包含 head_cam 和 agent_pos 的字典
        
        Returns:
            base64 编码的 pickle 字符串
        """
        obs_bytes = pickle.dumps(observation)
        obs_b64 = base64.b64encode(obs_bytes).decode("utf-8")
        return obs_b64

    def update_obs(self, observation: Dict[str, np.ndarray]):
        """
        更新观察历史
        
        Args:
            observation: 包含 head_cam 和 agent_pos 的字典
                - head_cam: numpy array, shape (C, H, W), 值范围 [0, 1]
                - agent_pos: numpy array, shape (n,)
        """
        # 存储待更新的观察，在下次 get_action 时发送
        self.pending_obs = observation
        logger.debug(f"Agent {self.agent_id} 观察已更新（待发送）")

    def get_action(self, observation: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """
        获取动作
        
        Args:
            observation: 可选的观察数据。如果提供，会先更新观察历史再获取动作
                        如果不提供，使用之前通过 update_obs 设置的观察
        
        Returns:
            action: numpy array，动作序列
        """
        # 确定要发送的观察数据
        obs_to_send = observation if observation is not None else self.pending_obs
        update_obs = obs_to_send is not None

        # 准备请求数据
        request_data = {
            "agent_id": self.agent_id,
            "update_obs": update_obs
        }

        if update_obs:
            # 序列化观察数据为 base64 编码的 pickle 字符串
            request_data["observation"] = self._serialize_observation(obs_to_send)
            # 清除待发送的观察
            self.pending_obs = None

        try:
            # 发送请求
            response = requests.post(
                f"{self.server_url}/infer",
                json=request_data,
                timeout=30  # 推理可能需要一些时间
            )
            response.raise_for_status()

            result = response.json()

            if not result.get("success", False):
                error_msg = result.get("error", "未知错误")
                raise RuntimeError(f"服务器返回错误: {error_msg}")

            # 解码结果
            result_b64 = result["result"]
            result_bytes = base64.b64decode(result_b64)
            result_dict = pickle.loads(result_bytes)

            action = result_dict["action"]

            # 确保返回 numpy array
            if not isinstance(action, np.ndarray):
                action = np.array(action)

            logger.debug(f"Agent {self.agent_id} 获取动作成功，shape: {action.shape}")
            return action

        except requests.exceptions.RequestException as e:
            logger.error(f"请求失败: {e}")
            raise
        except Exception as e:
            logger.error(f"处理响应失败: {e}")
            raise

    def reset_obs(self):
        """重置观察历史"""
        try:
            response = requests.post(
                f"{self.server_url}/reset",
                json={"agent_id": self.agent_id},
                timeout=5
            )
            response.raise_for_status()
            self.pending_obs = None
            logger.debug(f"Agent {self.agent_id} 观察历史已重置")
        except requests.exceptions.RequestException as e:
            logger.error(f"重置观察历史失败: {e}")
            raise

    def get_last_obs(self) -> Optional[Dict[str, np.ndarray]]:
        """
        获取最后一次观察（客户端不保存观察历史，返回 None 或 pending_obs）
        
        注意：由于观察历史存储在服务器端，客户端无法直接获取。
        这个方法返回 pending_obs（如果有），否则返回 None。
        """
        return self.pending_obs





