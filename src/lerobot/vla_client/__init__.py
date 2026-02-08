# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
VLA (Vision-Language-Action) Client Module

This module provides a client to communicate with a remote VLA server
(e.g., OpenPI, Pi0) for real robot inference.

Usage:
    from lerobot.vla_client import VLAClient, VLAObservationAdapter, VLAActionAdapter

    # 创建客户端
    client = VLAClient(server_url="http://192.168.1.100:20019")

    # 创建适配器
    obs_adapter = VLAObservationAdapter()
    action_adapter = VLAActionAdapter()

    # 获取机器人观察
    lerobot_obs = robot.get_observation()

    # 转换为 VLA 格式并推理
    vla_example = obs_adapter.convert(lerobot_obs, task="pick up the cup")
    vla_actions = client.get_actions(vla_example)

    # 转换为 LeRobot 格式并执行
    lerobot_action = action_adapter.convert(vla_actions)
    robot.send_action(lerobot_action)
"""

from .action_adapter import (
    BimanualVLAActionAdapter,
    BimanualVLAActionLimiter,
    VLAActionAdapter,
    VLAActionChunkExecutor,
    VLAActionLimiter,
)
from .client import VLAClient
from .observation_adapter import (
    BimanualVLAObservationAdapter,
    DualCameraVLAObservationAdapter,
    VLAObservationAdapter,
)

__all__ = [
    "VLAClient",
    "VLAObservationAdapter",
    "DualCameraVLAObservationAdapter",
    "BimanualVLAObservationAdapter",
    "VLAActionAdapter",
    "VLAActionLimiter",
    "VLAActionChunkExecutor",
    "BimanualVLAActionAdapter",
    "BimanualVLAActionLimiter",
]




