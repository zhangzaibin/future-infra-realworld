#!/usr/bin/env python3
"""
将 LeRobot 保存的 depth_raw 视频转换为深度数据

使用方法:
    python convert_depth_video.py <视频路径> [输出目录]

示例:
    # 转换单个视频
    python convert_depth_video.py ~/.cache/huggingface/lerobot/robodata/test/videos/observation.images.orbbec_depth/chunk-000/file-000.mp4
    
    # 指定输出目录
    python convert_depth_video.py video.mp4 ./output_depth
    
    # 转换整个数据集的所有深度视频
    python convert_depth_video.py ~/.cache/huggingface/lerobot/robodata/test --dataset

输出:
    - depth_frames/frame_XXXXX.npy  : 每帧的原始深度数据 (uint16, 毫米)
    - depth_frames/frame_XXXXX.png  : 每帧的彩色深度可视化
    - depth_all.npy                  : 所有帧的深度数据 (N, H, W)
    - depth_stats.json               : 深度统计信息
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def decode_depth_image(rgb_image: np.ndarray) -> np.ndarray:
    """
    将 depth_raw 编码的 RGB 图像解码为深度值（毫米）
    
    编码方式: R = 高8位, G = 低8位, B = 0
    解码公式: depth_mm = R * 256 + G
    """
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError(f"期望 RGB 图像 (H, W, 3), 实际: {rgb_image.shape}")
    
    high_byte = rgb_image[:, :, 0].astype(np.uint16)
    low_byte = rgb_image[:, :, 1].astype(np.uint16)
    depth_mm = (high_byte << 8) | low_byte
    
    return depth_mm


def depth_to_colormap(depth_mm: np.ndarray, min_depth: int = 200, max_depth: int = 5000) -> np.ndarray:
    """将深度图转换为彩色可视化 (蓝色=近, 红色=远)"""
    depth_clipped = np.clip(depth_mm, min_depth, max_depth)
    depth_normalized = ((depth_clipped - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    depth_colored[depth_mm == 0] = [0, 0, 0]  # 无效深度设为黑色
    return depth_colored


def convert_video_to_depth(video_path: str, output_dir: str = None, save_frames: bool = True, 
                           save_visualization: bool = True) -> dict:
    """
    将 depth_raw 视频转换为深度数据
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录，None 则使用视频同目录
        save_frames: 是否保存每帧的 .npy 文件
        save_visualization: 是否保存彩色可视化
    
    Returns:
        dict: 包含统计信息的字典
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    if output_dir is None:
        output_dir = video_path.parent / f"{video_path.stem}_depth"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "depth_frames"
    frames_dir.mkdir(exist_ok=True)
    
    # 打开视频
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"视频信息: {width}x{height}, {total_frames} 帧, {fps:.1f} FPS")
    print(f"输出目录: {output_dir}")
    
    all_depths = []
    depth_stats = {
        "video_path": str(video_path),
        "total_frames": total_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "min_depth_mm": float('inf'),
        "max_depth_mm": 0,
        "mean_depth_mm": 0,
        "valid_pixel_ratio": 0,
    }
    
    frame_idx = 0
    total_valid_pixels = 0
    total_pixels = 0
    sum_depth = 0
    
    pbar = tqdm(total=total_frames, desc="转换深度帧")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # OpenCV 读取为 BGR，转换为 RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 解码深度
        depth_mm = decode_depth_image(frame_rgb)
        all_depths.append(depth_mm)
        
        # 统计
        valid_mask = depth_mm > 0
        valid_pixels = np.sum(valid_mask)
        total_valid_pixels += valid_pixels
        total_pixels += depth_mm.size
        
        if valid_pixels > 0:
            frame_min = depth_mm[valid_mask].min()
            frame_max = depth_mm[valid_mask].max()
            sum_depth += depth_mm[valid_mask].sum()
            
            depth_stats["min_depth_mm"] = min(depth_stats["min_depth_mm"], int(frame_min))
            depth_stats["max_depth_mm"] = max(depth_stats["max_depth_mm"], int(frame_max))
        
        # 保存帧
        if save_frames:
            np.save(frames_dir / f"frame_{frame_idx:05d}.npy", depth_mm)
        
        if save_visualization:
            vis = depth_to_colormap(depth_mm)
            cv2.imwrite(str(frames_dir / f"frame_{frame_idx:05d}_vis.png"), vis)
        
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    # 计算统计
    if total_valid_pixels > 0:
        depth_stats["mean_depth_mm"] = float(sum_depth / total_valid_pixels)
    depth_stats["valid_pixel_ratio"] = float(total_valid_pixels / total_pixels)
    
    if depth_stats["min_depth_mm"] == float('inf'):
        depth_stats["min_depth_mm"] = 0
    
    # 保存所有深度数据为单个文件
    all_depths = np.stack(all_depths, axis=0)
    np.save(output_dir / "depth_all.npy", all_depths)
    print(f"保存完整深度数据: {output_dir / 'depth_all.npy'} (shape: {all_depths.shape})")
    
    # 保存统计信息
    with open(output_dir / "depth_stats.json", 'w') as f:
        json.dump(depth_stats, f, indent=2)
    print(f"保存统计信息: {output_dir / 'depth_stats.json'}")
    
    return depth_stats


def convert_dataset_depths(dataset_path: str, output_base_dir: str = None):
    """转换整个数据集中的所有深度视频"""
    dataset_path = Path(dataset_path)
    videos_dir = dataset_path / "videos"
    
    if not videos_dir.exists():
        raise FileNotFoundError(f"视频目录不存在: {videos_dir}")
    
    # 查找所有深度视频目录
    depth_dirs = list(videos_dir.glob("*depth*")) + list(videos_dir.glob("*orbbec*depth*"))
    
    if not depth_dirs:
        print("未找到深度视频目录")
        print("提示: 深度视频目录名称应包含 'depth'")
        print(f"可用目录: {[d.name for d in videos_dir.iterdir() if d.is_dir()]}")
        return
    
    print(f"找到 {len(depth_dirs)} 个深度视频目录")
    
    for depth_dir in depth_dirs:
        print(f"\n处理目录: {depth_dir.name}")
        
        # 查找所有视频文件
        video_files = list(depth_dir.rglob("*.mp4"))
        
        for video_file in video_files:
            print(f"  转换: {video_file.relative_to(dataset_path)}")
            
            if output_base_dir:
                output_dir = Path(output_base_dir) / video_file.relative_to(videos_dir).parent
            else:
                output_dir = None
            
            try:
                stats = convert_video_to_depth(video_file, output_dir)
                print(f"    深度范围: {stats['min_depth_mm']} - {stats['max_depth_mm']} mm")
                print(f"    有效像素: {stats['valid_pixel_ratio']*100:.1f}%")
            except Exception as e:
                print(f"    错误: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="将 LeRobot 的 depth_raw 视频转换为深度数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 转换单个视频
    python convert_depth_video.py video.mp4
    
    # 转换整个数据集
    python convert_depth_video.py ~/.cache/huggingface/lerobot/robodata/test --dataset
    
    # 指定输出目录
    python convert_depth_video.py video.mp4 -o ./output

深度数据恢复公式:
    depth_mm = frame[:, :, 0] * 256 + frame[:, :, 1]
        """
    )
    
    parser.add_argument("input", help="视频文件路径或数据集目录")
    parser.add_argument("-o", "--output", help="输出目录")
    parser.add_argument("--dataset", action="store_true", help="转换整个数据集的深度视频")
    parser.add_argument("--no-frames", action="store_true", help="不保存单帧 .npy 文件")
    parser.add_argument("--no-vis", action="store_true", help="不保存可视化图像")
    
    args = parser.parse_args()
    
    if args.dataset:
        convert_dataset_depths(args.input, args.output)
    else:
        stats = convert_video_to_depth(
            args.input, 
            args.output,
            save_frames=not args.no_frames,
            save_visualization=not args.no_vis
        )
        
        print("\n=== 深度统计 ===")
        print(f"帧数: {stats['total_frames']}")
        print(f"深度范围: {stats['min_depth_mm']} - {stats['max_depth_mm']} mm")
        print(f"平均深度: {stats['mean_depth_mm']:.1f} mm")
        print(f"有效像素比例: {stats['valid_pixel_ratio']*100:.1f}%")


if __name__ == "__main__":
    main()








