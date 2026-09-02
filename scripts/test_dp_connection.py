#!/usr/bin/env python3
"""
测试 DP 服务器连接

使用方法:
    python scripts/test_dp_connection.py --server_url http://192.168.1.100:20020 --agent_id 0
"""

import argparse
import numpy as np
import sys

sys.path.insert(0, "src")

from lerobot.dp_client import DPClient, ObservationAdapter, ActionAdapter


def test_connection(server_url: str, agent_id: int):
    """测试与 DP 服务器的连接"""
    print("=" * 60)
    print("测试 DP 服务器连接")
    print("=" * 60)
    print(f"服务器 URL: {server_url}")
    print(f"Agent ID: {agent_id}")
    print("=" * 60)

    try:
        # 1. 测试连接
        print("\n[1/4] 连接到 DP 服务器...")
        client = DPClient(agent_id=agent_id, server_url=server_url)
        print("✓ 连接成功")

        # 2. 测试重置
        print("\n[2/4] 重置观察历史...")
        client.reset_obs()
        print("✓ 重置成功")

        # 3. 创建模拟观察数据
        print("\n[3/4] 发送测试观察数据...")
        
        # 模拟图像: (3, 480, 640) RGB float32 [0, 1]
        fake_image = np.random.rand(3, 480, 640).astype(np.float32)
        
        # 模拟关节位置: 6 个关节
        fake_agent_pos = np.zeros(6, dtype=np.float32)
        
        test_obs = {
            "head_cam": fake_image,
            "agent_pos": fake_agent_pos,
        }
        
        print(f"  图像 shape: {fake_image.shape}")
        print(f"  关节位置 shape: {fake_agent_pos.shape}")

        # 4. 测试推理
        print("\n[4/4] 测试推理...")
        action = client.get_action(test_obs)
        print(f"✓ 推理成功")
        print(f"  动作 shape: {action.shape}")
        print(f"  动作值: {action}")

        print("\n" + "=" * 60)
        print("所有测试通过!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_adapters():
    """测试适配器"""
    print("\n" + "=" * 60)
    print("测试观察和动作适配器")
    print("=" * 60)

    # 模拟 LeRobot 观察
    lerobot_obs = {
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": 10.0,
        "elbow_flex.pos": 20.0,
        "wrist_flex.pos": 30.0,
        "wrist_roll.pos": 40.0,
        "gripper.pos": 50.0,
        "front": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
    }

    print("\n[1/2] 测试观察适配器...")
    obs_adapter = ObservationAdapter(camera_key="front")
    dp_obs = obs_adapter.convert(lerobot_obs)
    print(f"  head_cam shape: {dp_obs['head_cam'].shape}")
    print(f"  head_cam dtype: {dp_obs['head_cam'].dtype}")
    print(f"  head_cam range: [{dp_obs['head_cam'].min():.3f}, {dp_obs['head_cam'].max():.3f}]")
    print(f"  agent_pos: {dp_obs['agent_pos']}")
    print("✓ 观察适配器测试通过")

    print("\n[2/2] 测试动作适配器...")
    # 模拟 DP 返回的动作
    dp_action = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    action_adapter = ActionAdapter()
    lerobot_action = action_adapter.convert(dp_action)
    print(f"  LeRobot 动作: {lerobot_action}")
    print("✓ 动作适配器测试通过")

    print("\n" + "=" * 60)
    print("适配器测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试 DP 服务器连接")
    parser.add_argument(
        "--server_url",
        type=str,
        default="http://localhost:20020",
        help="DP 服务器 URL",
    )
    parser.add_argument(
        "--agent_id",
        type=int,
        default=0,
        help="Agent ID",
    )
    parser.add_argument(
        "--test_adapters_only",
        action="store_true",
        help="只测试适配器，不连接服务器",
    )

    args = parser.parse_args()

    if args.test_adapters_only:
        test_adapters()
    else:
        test_adapters()
        print("\n")
        success = test_connection(args.server_url, args.agent_id)
        sys.exit(0 if success else 1)





