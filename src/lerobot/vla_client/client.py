#!/usr/bin/env python3
"""
VLA (Vision-Language-Action) 客户端
通过 HTTP 请求与 VLA 服务器 (read/server.py) 通信
"""
import base64
import logging
import pickle
from typing import Any

import numpy as np
import requests

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VLAClient:
    """VLA 客户端类，通过 HTTP 与服务器通信"""

    def __init__(
        self,
        server_url: str = "http://localhost:20019",
        timeout: float = 120.0,
    ):
        """
        初始化客户端

        Args:
            server_url: VLA 服务器 URL，默认为 "http://localhost:20019"
            timeout: 请求超时时间（秒）。Pi0 推理可能较慢，默认设置为 120s。
        """
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

        # 检查服务器健康状态
        self._check_health()

    def _check_health(self) -> bool:
        """检查服务器健康状态"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            response.raise_for_status()
            health_data = response.json()
            if not health_data.get("model_loaded", False):
                logger.warning("VLA 服务器模型未加载")
                return False
            logger.info(f"VLA 服务器健康检查通过: {health_data}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"无法连接到 VLA 服务器 {self.server_url}: {e}")
            raise

    def _serialize_example(self, example: dict[str, Any]) -> str:
        """
        序列化样本数据为 base64 编码的 pickle 字符串

        Args:
            example: VLA 样本字典

        Returns:
            base64 编码的 pickle 字符串
        """
        example_bytes = pickle.dumps(example)
        example_b64 = base64.b64encode(example_bytes).decode("utf-8")
        return example_b64

    def _deserialize_result(self, result_b64: str) -> dict[str, Any]:
        """
        反序列化服务器返回的结果

        Args:
            result_b64: base64 编码的 pickle 字符串

        Returns:
            结果字典
        """
        result_bytes = base64.b64decode(result_b64)
        result = pickle.loads(result_bytes)
        return result

    def infer(self, example: dict[str, Any]) -> dict[str, Any]:
        """
        调用 VLA 服务器进行推理

        Args:
            example: VLA 样本字典，应包含：
                - observation/state: np.ndarray, shape (6,) - 关节状态
                - observation/image: np.ndarray, shape (H, W, 3) - 前置摄像头图像
                - observation/wrist_image: np.ndarray, shape (H, W, 3) - 腕部摄像头图像
                - task: str - 任务描述

        Returns:
            推理结果字典，包含 "actions" 键
        """
        try:
            # 序列化样本数据
            example_b64 = self._serialize_example(example)

            # 准备请求数据
            request_data = {"example": example_b64}

            # 发送请求
            response = requests.post(
                f"{self.server_url}/infer",
                json=request_data,
                timeout=self.timeout,
            )
            response.raise_for_status()

            result = response.json()

            if not result.get("success", False):
                error_msg = result.get("error", "未知错误")
                raise RuntimeError(f"VLA 服务器返回错误: {error_msg}")

            # 解码结果
            result_data = self._deserialize_result(result["result"])

            logger.debug(f"VLA 推理成功，actions_shape: {result.get('actions_shape')}")
            return result_data

        except requests.exceptions.RequestException as e:
            logger.error(f"VLA 请求失败: {e}")
            raise
        except Exception as e:
            logger.error(f"VLA 推理处理失败: {e}")
            raise

    def get_actions(self, example: dict[str, Any]) -> np.ndarray:
        """
        获取动作（便捷方法）

        Args:
            example: VLA 样本字典

        Returns:
            actions: np.ndarray, shape (chunk_size, action_dim)
        """
        result = self.infer(example)
        actions = result.get("actions")

        if actions is None:
            raise RuntimeError("VLA 服务器返回的结果中没有 'actions' 键")

        # 确保返回 numpy array
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)

        return actions

    def test(self) -> dict[str, Any]:
        """
        调用测试接口

        Returns:
            测试结果字典
        """
        try:
            response = requests.get(f"{self.server_url}/test", timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"VLA 测试请求失败: {e}")
            raise




