#!/usr/bin/env python3
"""
测试 VLA 服务器连接

使用方法:
    python scripts/test_vla_connection.py --server_url http://192.168.1.100:20019
    python scripts/test_vla_connection.py --test_adapters_only
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, "src")

from lerobot.vla_client import (
    VLAActionAdapter,
    VLAActionLimiter,
    VLAClient,
    VLAObservationAdapter,
)


def test_connection(server_url: str, task: str = "pick up the cup"):
    """测试与 VLA 服务器的连接"""
    print("=" * 60)
    print("测试 VLA 服务器连接")
    print("=" * 60)
    print(f"服务器 URL: {server_url}")
    print(f"任务: {task}")
    print("=" * 60)

    try:
        # 1. 测试连接
        print("\n[1/4] 连接到 VLA 服务器...")
        client = VLAClient(server_url=server_url)
        print("✓ 连接成功")

        # 2. 测试健康检查
        print("\n[2/4] 调用测试接口...")
        test_result = client.test()
        print(f"✓ 测试接口调用成功: {test_result}")

        # 3. 创建模拟观察数据
        print("\n[3/4] 发送测试观察数据...")

        # 模拟 VLA 输入数据
        test_example = {
            "observation/state": np.random.rand(6).astype(np.float32),
            "observation/image": np.random.randint(
                256, size=(480, 640, 3), dtype=np.uint8
            ),
            "observation/wrist_image": np.random.randint(
                256, size=(480, 640, 3), dtype=np.uint8
            ),
            "task": task,
        }

        print(f"  状态 shape: {test_example['observation/state'].shape}")
        print(f"  前置图像 shape: {test_example['observation/image'].shape}")
        print(f"  腕部图像 shape: {test_example['observation/wrist_image'].shape}")
        print(f"  任务: {test_example['task']}")

        # 4. 测试推理
        print("\n[4/4] 测试推理...")
        actions = client.get_actions(test_example)
        print("✓ 推理成功")
        print(f"  动作 shape: {actions.shape}")
        print(f"  动作值 (前6维): {actions[:, :6] if len(actions.shape) == 2 else actions[:6]}")

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
    print("测试 VLA 观察和动作适配器")
    print("=" * 60)

    # 模拟 LeRobot 观察 (global + wrist 双摄像头)
    lerobot_obs = {
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": 10.0,
        "elbow_flex.pos": 20.0,
        "wrist_flex.pos": 30.0,
        "wrist_roll.pos": 40.0,
        "gripper.pos": 50.0,
        "global": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        "wrist": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
    }

    print("\n[1/3] 测试观察适配器...")
    obs_adapter = VLAObservationAdapter(
        front_camera_key="global",
        wrist_camera_key="wrist",
    )
    vla_example = obs_adapter.convert(lerobot_obs, task="pick up the cup")
    print(f"  observation/state: {vla_example['observation/state']}")
    print(f"  observation/image shape: {vla_example['observation/image'].shape}")
    print(f"  observation/wrist_image shape: {vla_example['observation/wrist_image'].shape}")
    print(f"  task: {vla_example['task']}")
    print("✓ 观察适配器测试通过")

    print("\n[2/3] 测试动作适配器...")
    # 模拟 VLA 返回的动作 (chunk_size, action_dim)
    vla_actions = np.random.rand(16, 6).astype(np.float32) * 100 - 50
    action_adapter = VLAActionAdapter()
    lerobot_action = action_adapter.convert(vla_actions)
    print(f"  输入动作 shape: {vla_actions.shape}")
    print(f"  LeRobot 动作: {lerobot_action}")
    print("✓ 动作适配器测试通过")

    print("\n[3/3] 测试动作限幅器...")
    action_limiter = VLAActionLimiter(max_delta=5.0)

    # 初始位置
    initial_pos = {
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": 0.0,
        "elbow_flex.pos": 0.0,
        "wrist_flex.pos": 0.0,
        "wrist_roll.pos": 0.0,
        "gripper.pos": 0.0,
    }
    action_limiter.set_last_action(initial_pos)

    # 目标动作（大幅度跳跃）
    target_action = {
        "shoulder_pan.pos": 50.0,  # 应该被限幅
        "shoulder_lift.pos": 3.0,  # 不需要限幅
        "elbow_flex.pos": -20.0,  # 应该被限幅
        "wrist_flex.pos": 2.0,  # 不需要限幅
        "wrist_roll.pos": 100.0,  # 应该被限幅
        "gripper.pos": 4.0,  # 不需要限幅
    }

    limited_action = action_limiter.limit(target_action)
    print(f"  目标动作: {target_action}")
    print(f"  限幅后动作: {limited_action}")
    print("✓ 动作限幅器测试通过")

    print("\n" + "=" * 60)
    print("适配器测试通过!")
    print("=" * 60)


def test_full_pipeline(server_url: str, task: str = "pick up the cup"):
    """测试完整的推理流程"""
    print("\n" + "=" * 60)
    print("测试完整的 VLA 推理流程")
    print("=" * 60)

    try:
        # 1. 初始化所有组件
        print("\n[1/5] 初始化组件...")
        client = VLAClient(server_url=server_url)
        obs_adapter = VLAObservationAdapter(
            front_camera_key="global",
            wrist_camera_key="wrist",
        )
        action_adapter = VLAActionAdapter()
        action_limiter = VLAActionLimiter(max_delta=5.0)
        print("✓ 组件初始化成功")

        # 2. 模拟 LeRobot 观察 (global + wrist 双摄像头)
        print("\n[2/5] 创建模拟 LeRobot 观察...")
        lerobot_obs = {
            "shoulder_pan.pos": 0.0,
            "shoulder_lift.pos": 10.0,
            "elbow_flex.pos": 20.0,
            "wrist_flex.pos": 30.0,
            "wrist_roll.pos": 40.0,
            "gripper.pos": 50.0,
            "global": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            "wrist": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        }
        print("✓ LeRobot 观察创建成功")

        # 初始化限幅器
        initial_pos = {k: v for k, v in lerobot_obs.items() if k.endswith(".pos")}
        action_limiter.set_last_action(initial_pos)

        # 3. 转换为 VLA 格式
        print("\n[3/5] 转换为 VLA 格式...")
        vla_example = obs_adapter.convert(lerobot_obs, task=task)
        print(f"  状态: {vla_example['observation/state']}")
        print(f"  任务: {vla_example['task']}")
        print("✓ 格式转换成功")

        # 4. 调用 VLA 服务器推理
        print("\n[4/5] 调用 VLA 服务器推理...")
        vla_actions = client.get_actions(vla_example)
        print(f"  返回动作 shape: {vla_actions.shape}")
        print("✓ VLA 推理成功")

        # 5. 转换回 LeRobot 格式并限幅
        print("\n[5/5] 转换为 LeRobot 动作格式...")
        lerobot_action = action_adapter.convert(vla_actions)
        limited_action = action_limiter.limit(lerobot_action)
        print(f"  原始动作: {lerobot_action}")
        print(f"  限幅动作: {limited_action}")
        print("✓ 动作转换成功")

        print("\n" + "=" * 60)
        print("完整流程测试通过!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试 VLA 服务器连接")
    parser.add_argument(
        "--server_url",
        type=str,
        default="http://localhost:20019",
        help="VLA 服务器 URL",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="pick up the cup and place it in the basket",
        help="测试任务描述",
    )
    parser.add_argument(
        "--test_adapters_only",
        action="store_true",
        help="只测试适配器，不连接服务器",
    )
    parser.add_argument(
        "--full_pipeline",
        action="store_true",
        help="测试完整的推理流程",
    )

    args = parser.parse_args()

    if args.test_adapters_only:
        test_adapters()
    elif args.full_pipeline:
        test_adapters()
        print("\n")
        success = test_full_pipeline(args.server_url, args.task)
        sys.exit(0 if success else 1)
    else:
        test_adapters()
        print("\n")
        success = test_connection(args.server_url, args.task)
        sys.exit(0 if success else 1)

