#!/usr/bin/env python3
"""
从深度视频生成并可视化点云

使用方法:
    python visualize_pointcloud.py <深度视频路径> [选项]

示例:
    # 可视化单帧
    python visualize_pointcloud.py video.mp4 --frame 0
    
    # 可视化并播放所有帧
    python visualize_pointcloud.py video.mp4 --play
    
    # 保存点云为 PLY 文件
    python visualize_pointcloud.py video.mp4 --frame 50 --save output.ply

依赖:
    pip install open3d av numpy
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# 尝试导入 PyAV (更好的视频解码支持，特别是 AV1)
try:
    import av
    HAS_PYAV = True
except ImportError:
    HAS_PYAV = False
    import cv2
    print("警告: 未安装 PyAV，使用 OpenCV 读取视频")
    print("如果遇到 AV1 解码问题，请安装: pip install av")

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("警告: 未安装 open3d，将使用 matplotlib 进行简化可视化")
    print("建议安装: pip install open3d")

# Orbbec Astra 相机内参
FX = 570.3422
FY = 570.3422
CX = 320.0
CY = 240.0


def decode_depth_image(rgb_image: np.ndarray, smooth: bool = False) -> np.ndarray:
    """
    将 depth_raw 编码的 RGB 图像解码为深度值（毫米）
    
    编码方式: R = 高8位, G = 低8位, B = 0
    解码公式: depth_mm = R * 256 + G
    
    注意: 如果视频使用有损压缩（如 AV1），深度数据会有量化误差，
    建议使用 smooth=True 进行滤波处理。
    """
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError(f"期望 RGB 图像 (H, W, 3), 实际: {rgb_image.shape}")
    
    high_byte = rgb_image[:, :, 0].astype(np.uint16)
    low_byte = rgb_image[:, :, 1].astype(np.uint16)
    depth_mm = (high_byte << 8) | low_byte
    
    if smooth:
        depth_mm = smooth_depth(depth_mm)
    
    return depth_mm


def smooth_depth(depth_mm: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    对深度图进行平滑处理，减少视频压缩带来的量化噪声
    
    使用双边滤波：保留边缘的同时平滑噪声
    """
    import cv2
    
    # 保存原始无效区域
    invalid_mask = depth_mm == 0
    
    # 转换为 float32 进行滤波
    depth_float = depth_mm.astype(np.float32)
    
    # 双边滤波：d=5, sigmaColor=50, sigmaSpace=50
    # 这些参数可以保留边缘同时平滑噪声
    depth_smooth = cv2.bilateralFilter(depth_float, d=kernel_size, sigmaColor=50, sigmaSpace=50)
    
    # 恢复为 uint16
    depth_smooth = depth_smooth.astype(np.uint16)
    
    # 恢复无效区域
    depth_smooth[invalid_mask] = 0
    
    return depth_smooth


class VideoReader:
    """视频读取器，优先使用 PyAV，回退到 OpenCV"""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.use_pyav = HAS_PYAV
        self._container = None
        self._cap = None
        self._total_frames = 0
        self._current_frame = 0
        
        if self.use_pyav:
            self._init_pyav()
        else:
            self._init_opencv()
    
    def _init_pyav(self):
        """使用 PyAV 初始化"""
        self._container = av.open(self.video_path)
        self._stream = self._container.streams.video[0]
        self._total_frames = self._stream.frames
        # 如果帧数为0，尝试通过duration估算
        if self._total_frames == 0:
            duration = float(self._stream.duration * self._stream.time_base)
            fps = float(self._stream.average_rate)
            self._total_frames = int(duration * fps)
        self._frames_iter = None
    
    def _init_opencv(self):
        """使用 OpenCV 初始化"""
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            raise RuntimeError(f"无法打开视频: {self.video_path}")
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    @property
    def total_frames(self) -> int:
        return self._total_frames
    
    def seek(self, frame_idx: int):
        """跳转到指定帧"""
        if self.use_pyav:
            # PyAV 需要重新打开并跳转
            if self._container:
                self._container.close()
            self._container = av.open(self.video_path)
            self._stream = self._container.streams.video[0]
            
            # 计算时间戳并 seek
            if self._stream.average_rate:
                fps = float(self._stream.average_rate)
                target_ts = int(frame_idx / fps / self._stream.time_base)
                self._container.seek(target_ts, stream=self._stream)
            
            self._frames_iter = self._container.decode(video=0)
            self._current_frame = frame_idx
        else:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            self._current_frame = frame_idx
    
    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        """读取下一帧，返回 (success, rgb_frame)"""
        if self.use_pyav:
            if self._frames_iter is None:
                self._frames_iter = self._container.decode(video=0)
            
            try:
                frame = next(self._frames_iter)
                rgb_frame = frame.to_ndarray(format='rgb24')
                self._current_frame += 1
                return True, rgb_frame
            except StopIteration:
                return False, None
        else:
            ret, frame = self._cap.read()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self._current_frame += 1
                return True, rgb_frame
            return False, None
    
    def read_frame_at(self, frame_idx: int) -> tuple[bool, np.ndarray | None]:
        """读取指定帧"""
        self.seek(frame_idx)
        return self.read_frame()
    
    def __iter__(self):
        """迭代所有帧"""
        self.seek(0)
        return self
    
    def __next__(self) -> np.ndarray:
        ret, frame = self.read_frame()
        if not ret:
            raise StopIteration
        return frame
    
    def close(self):
        """关闭视频"""
        if self._container:
            self._container.close()
            self._container = None
        if self._cap:
            self._cap.release()
            self._cap = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


def depth_to_pointcloud(depth_mm: np.ndarray, fx: float = FX, fy: float = FY, 
                        cx: float = CX, cy: float = CY, 
                        min_depth: float = 100, max_depth: float = 5000) -> np.ndarray:
    """
    将深度图转换为点云
    
    Args:
        depth_mm: 深度图 (H, W)，单位毫米
        fx, fy: 焦距
        cx, cy: 主点
        min_depth: 最小有效深度 (mm)
        max_depth: 最大有效深度 (mm)
    
    Returns:
        points: 点云 (N, 3)，单位米
    """
    height, width = depth_mm.shape
    
    # 创建像素坐标网格
    u = np.arange(width)
    v = np.arange(height)
    u, v = np.meshgrid(u, v)
    
    # 转换为米
    depth_m = depth_mm.astype(np.float32) / 1000.0
    
    # 过滤无效深度
    valid_mask = (depth_mm > min_depth) & (depth_mm < max_depth)
    
    # 反投影到3D
    z = depth_m
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # 组合点云
    points = np.stack([x, y, z], axis=-1)  # (H, W, 3)
    points = points[valid_mask]  # (N, 3)
    
    return points, valid_mask


def depth_to_colors(depth_mm: np.ndarray, valid_mask: np.ndarray, 
                    min_depth: int = 200, max_depth: int = 3000) -> np.ndarray:
    """
    为点云生成颜色（基于深度）
    
    Args:
        depth_mm: 深度图 (H, W)
        valid_mask: 有效像素掩码 (H, W)
        min_depth, max_depth: 颜色映射的深度范围
    
    Returns:
        colors: 颜色数组 (N, 3)，范围 [0, 1]
    """
    import matplotlib.pyplot as plt
    
    depth_clipped = np.clip(depth_mm, min_depth, max_depth)
    depth_normalized = (depth_clipped - min_depth) / (max_depth - min_depth)
    
    # 使用 jet colormap
    cmap = plt.cm.jet
    colors_full = cmap(depth_normalized)[:, :, :3]  # (H, W, 3), RGB
    colors = colors_full[valid_mask]  # (N, 3)
    
    return colors


def rgb_to_colors(rgb_image: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """
    从 RGB 图像提取点云颜色
    
    Args:
        rgb_image: RGB 图像 (H, W, 3)，范围 [0, 255]
        valid_mask: 有效像素掩码 (H, W)
    
    Returns:
        colors: 颜色数组 (N, 3)，范围 [0, 1]
    """
    # 归一化到 [0, 1]
    colors_full = rgb_image.astype(np.float32) / 255.0
    colors = colors_full[valid_mask]  # (N, 3)
    return colors


def visualize_pointcloud_open3d(points: np.ndarray, colors: np.ndarray = None, 
                                 window_name: str = "Point Cloud"):
    """使用 Open3D 可视化点云（需要图形界面）"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 创建坐标系
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    
    # 可视化
    o3d.visualization.draw_geometries(
        [pcd, coord_frame],
        window_name=window_name,
        width=1280,
        height=720,
        point_show_normal=False
    )


def render_pointcloud_to_image(points: np.ndarray, colors: np.ndarray = None,
                                output_path: str = "pointcloud.png",
                                width: int = 1280, height: int = 720):
    """离线渲染点云为图片（不需要图形界面）"""
    if not HAS_OPEN3D:
        print("错误: 需要 open3d 进行渲染")
        return False
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 创建坐标系
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    
    try:
        # 尝试使用离线渲染器
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=width, height=height)
        vis.add_geometry(pcd)
        vis.add_geometry(coord_frame)
        
        # 设置视角
        ctr = vis.get_view_control()
        ctr.set_zoom(0.5)
        ctr.set_front([0, 0, -1])
        ctr.set_lookat([0, 0, 1])
        ctr.set_up([0, -1, 0])
        
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(output_path)
        vis.destroy_window()
        print(f"点云图片已保存到: {output_path}")
        return True
    except Exception as e:
        print(f"Open3D 离线渲染失败: {e}")
        print("尝试使用 matplotlib 渲染...")
        return False


def visualize_pointcloud_matplotlib(points: np.ndarray, colors: np.ndarray = None,
                                    subsample: int = 10, save_path: str = None,
                                    show: bool = True, view: str = "camera"):
    """
    使用 matplotlib 可视化点云（备用方案，支持无图形界面）
    
    Args:
        view: 视角模式
            - "camera": 从相机视角看（推荐，看到的是场景）
            - "3d": 斜45度3D视角
            - "all": 显示所有视角
    """
    import matplotlib
    if save_path and not show:
        matplotlib.use('Agg')  # 无图形界面后端
    import matplotlib.pyplot as plt
    
    # 降采样以提高性能
    indices = np.arange(0, len(points), subsample)
    points_sub = points[indices]
    
    # 视角定义: (elev, azim, title)
    # elev: 仰角，azim: 方位角
    # 从相机视角看：相机在原点看向+Z方向，所以我们要从+Z方向往回看
    view_configs = {
        "camera": [(-90, -90, "Camera View (looking from camera)")],  # 从相机视角
        "3d": [(30, -60, "3D View")],
        "top": [(90, -90, "Top View (Bird's Eye)")],
        "front": [(0, -90, "Front View")],
        "side": [(0, 0, "Side View")],
        "all": [
            (221, -90, -90, "Camera View"),      # 相机视角（主要）
            (222, 30, -60, "3D View"),           # 3D斜视角
            (223, 90, -90, "Top View"),          # 俯视
            (224, 0, 0, "Side View"),            # 侧视
        ],
    }
    
    if view == "all" or save_path:
        # 多视角布局
        fig = plt.figure(figsize=(14, 12))
        views = view_configs["all"]
        for subplot, elev, azim, title in views:
            ax = fig.add_subplot(subplot, projection='3d')
            _plot_pointcloud_3d(ax, points_sub, colors, indices, elev, azim, title, len(points), subsample)
    else:
        # 单视角
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        elev, azim, title = view_configs.get(view, view_configs["camera"])[0]
        _plot_pointcloud_3d(ax, points_sub, colors, indices, elev, azim, title, len(points), subsample)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"点云图片已保存到: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def _plot_pointcloud_3d(ax, points_sub, colors, indices, elev, azim, title, total_points, subsample):
    """辅助函数：在3D轴上绘制点云"""
    if colors is not None:
        colors_sub = colors[indices]
        ax.scatter(points_sub[:, 0], points_sub[:, 2], -points_sub[:, 1],  # 交换Y和Z，翻转Y
                   c=colors_sub, s=1, alpha=0.6)
    else:
        ax.scatter(points_sub[:, 0], points_sub[:, 2], -points_sub[:, 1], 
                   c=points_sub[:, 2], cmap='jet', s=1, alpha=0.6)
    
    ax.set_xlabel('X (m) - Left/Right')
    ax.set_ylabel('Z (m) - Depth')
    ax.set_zlabel('Y (m) - Up/Down')
    ax.set_title(f'{title}\n({total_points} points, 1/{subsample} shown)')
    ax.view_init(elev=elev, azim=azim)
    
    # 设置相等的坐标轴比例
    if len(points_sub) > 0:
        max_range = max(
            points_sub[:, 0].max() - points_sub[:, 0].min(),
            points_sub[:, 1].max() - points_sub[:, 1].min(),
            points_sub[:, 2].max() - points_sub[:, 2].min()
        ) / 2
        mid_x = (points_sub[:, 0].max() + points_sub[:, 0].min()) / 2
        mid_y = (points_sub[:, 1].max() + points_sub[:, 1].min()) / 2
        mid_z = (points_sub[:, 2].max() + points_sub[:, 2].min()) / 2
        
        ax.set_xlim([mid_x - max_range, mid_x + max_range])
        ax.set_ylim([mid_z - max_range, mid_z + max_range])  # Z轴作为Y显示
        ax.set_zlim([-mid_y - max_range, -mid_y + max_range])  # Y轴翻转作为Z显示


def save_pointcloud(points: np.ndarray, colors: np.ndarray, output_path: str):
    """保存点云为 PLY 文件"""
    if not HAS_OPEN3D:
        print("错误: 需要 open3d 来保存 PLY 文件")
        return
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"点云已保存到: {output_path}")


def play_pointcloud_video(video_path: str, fps: float = 10, 
                          fx: float = FX, fy: float = FY,
                          cx: float = CX, cy: float = CY,
                          min_depth: float = 100, max_depth: float = 5000):
    """播放点云视频"""
    if not HAS_OPEN3D:
        print("错误: 需要 open3d 来播放点云视频")
        return
    
    reader = VideoReader(video_path)
    total_frames = reader.total_frames
    print(f"视频共 {total_frames} 帧，关闭窗口退出")
    
    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud Video", width=1280, height=720)
    
    # 添加坐标系
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    vis.add_geometry(coord_frame)
    
    # 创建初始点云
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    
    # 设置视角
    ctr = vis.get_view_control()
    ctr.set_zoom(0.5)
    ctr.set_front([0, 0, -1])
    ctr.set_lookat([0, 0, 1])
    ctr.set_up([0, -1, 0])
    
    frame_idx = 0
    
    try:
        while True:
            ret, frame_rgb = reader.read_frame()
            if not ret:
                # 循环播放
                reader.seek(0)
                ret, frame_rgb = reader.read_frame()
                if not ret:
                    break
                frame_idx = 0
            
            # 解码深度
            depth_mm = decode_depth_image(frame_rgb)
            
            # 转换为点云
            points, valid_mask = depth_to_pointcloud(depth_mm, fx, fy, cx, cy, min_depth, max_depth)
            colors = depth_to_colors(depth_mm, valid_mask)
            
            # 更新点云
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            vis.update_geometry(pcd)
            
            print(f"\r帧 {frame_idx}/{total_frames}, 点数: {len(points)}", end="")
            frame_idx += 1
            
            # 更新可视化
            if not vis.poll_events():
                break
            vis.update_renderer()
    finally:
        print()
        reader.close()
        vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(
        description="从深度视频生成并可视化点云",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 可视化第一帧 (需要图形界面)
    python visualize_pointcloud.py depth.mp4
    
    # 带 RGB 颜色的点云 (推荐)
    python visualize_pointcloud.py depth.mp4 --rgb rgb.mp4 --render pointcloud.png
    
    # 可视化指定帧
    python visualize_pointcloud.py depth.mp4 --frame 50
    
    # 渲染点云为图片 (无需图形界面，适合 SSH 远程)
    python visualize_pointcloud.py depth.mp4 --render pointcloud.png
    python visualize_pointcloud.py depth.mp4 -f 50 -r frame50.png
    
    # 播放点云视频
    python visualize_pointcloud.py depth.mp4 --play
    
    # 保存点云为 PLY 文件
    python visualize_pointcloud.py depth.mp4 --frame 0 --save cloud.ply

相机内参 (Orbbec Astra):
    fx = fy = 570.3422
    cx = 320.0, cy = 240.0
        """
    )
    
    parser.add_argument("video", help="深度视频路径")
    parser.add_argument("--rgb", type=str, help="RGB 视频路径 (可选，用于给点云上色)")
    parser.add_argument("--frame", "-f", type=int, default=0, help="要可视化的帧索引 (默认: 0)")
    parser.add_argument("--play", "-p", action="store_true", help="播放点云视频")
    parser.add_argument("--save", "-s", type=str, help="保存点云为 PLY 文件")
    parser.add_argument("--render", "-r", type=str, help="渲染点云为图片 (无需图形界面)")
    parser.add_argument("--fx", type=float, default=FX, help=f"焦距 fx (默认: {FX})")
    parser.add_argument("--fy", type=float, default=FY, help=f"焦距 fy (默认: {FY})")
    parser.add_argument("--cx", type=float, default=CX, help=f"主点 cx (默认: {CX})")
    parser.add_argument("--cy", type=float, default=CY, help=f"主点 cy (默认: {CY})")
    parser.add_argument("--min-depth", type=float, default=100, help="最小有效深度 mm (默认: 100)")
    parser.add_argument("--max-depth", type=float, default=600, help="最大有效深度 mm (默认: 1000, 即1米)")
    parser.add_argument("--subsample", type=int, default=5, help="matplotlib 渲染时的降采样率 (默认: 5)")
    parser.add_argument("--smooth", action="store_true", 
                        help="对深度图进行平滑滤波，减少视频压缩带来的量化噪声（推荐开启）")
    parser.add_argument("--view", "-v", type=str, default="camera", 
                        choices=["camera", "3d", "top", "front", "side", "all"],
                        help="视角: camera(相机视角,推荐), 3d, top, front, side, all (默认: camera)")
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"错误: 视频文件不存在: {video_path}")
        sys.exit(1)
    
    if args.play:
        # 播放模式
        play_pointcloud_video(str(video_path), fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
                              min_depth=args.min_depth, max_depth=args.max_depth)
    else:
        # 单帧模式
        try:
            reader = VideoReader(str(video_path))
        except Exception as e:
            print(f"错误: 无法打开视频: {video_path}")
            print(f"详情: {e}")
            sys.exit(1)
        
        total_frames = reader.total_frames
        print(f"视频共 {total_frames} 帧")
        
        if total_frames > 0 and args.frame >= total_frames:
            print(f"错误: 帧索引 {args.frame} 超出范围 (总帧数: {total_frames})")
            reader.close()
            sys.exit(1)
        
        # 读取指定帧
        ret, frame_rgb = reader.read_frame_at(args.frame)
        reader.close()
        
        if not ret:
            print(f"错误: 无法读取帧 {args.frame}")
            sys.exit(1)
        
        # 解码深度
        depth_mm = decode_depth_image(frame_rgb, smooth=args.smooth)
        
        print(f"深度图大小: {depth_mm.shape}")
        if args.smooth:
            print("已启用深度平滑滤波")
        print(f"深度范围: {depth_mm[depth_mm > 0].min()} - {depth_mm.max()} mm")
        
        # 转换为点云
        points, valid_mask = depth_to_pointcloud(
            depth_mm, args.fx, args.fy, args.cx, args.cy,
            args.min_depth, args.max_depth
        )
        
        # 获取颜色
        rgb_frame = None
        if args.rgb:
            # 从 RGB 视频获取颜色
            rgb_path = Path(args.rgb)
            if not rgb_path.exists():
                print(f"警告: RGB 视频不存在: {rgb_path}，使用深度着色")
            else:
                try:
                    rgb_reader = VideoReader(str(rgb_path))
                    ret_rgb, rgb_frame = rgb_reader.read_frame_at(args.frame)
                    rgb_reader.close()
                    
                    if ret_rgb:
                        # 检查尺寸是否匹配
                        if rgb_frame.shape[:2] != depth_mm.shape:
                            print(f"警告: RGB 图像尺寸 {rgb_frame.shape[:2]} 与深度图 {depth_mm.shape} 不匹配")
                            # 尝试调整大小
                            import cv2
                            rgb_frame = cv2.resize(rgb_frame, (depth_mm.shape[1], depth_mm.shape[0]))
                            print(f"已调整 RGB 图像尺寸为 {rgb_frame.shape[:2]}")
                        
                        print(f"RGB 图像大小: {rgb_frame.shape}")
                    else:
                        print(f"警告: 无法读取 RGB 帧 {args.frame}，使用深度着色")
                        rgb_frame = None
                except Exception as e:
                    print(f"警告: 读取 RGB 视频失败: {e}，使用深度着色")
                    rgb_frame = None
        
        if rgb_frame is not None:
            colors = rgb_to_colors(rgb_frame, valid_mask)
            print("使用 RGB 颜色")
        else:
            colors = depth_to_colors(depth_mm, valid_mask)
            print("使用深度颜色")
        
        print(f"点云点数: {len(points)}")
        print(f"点云范围:")
        print(f"  X: {points[:, 0].min():.3f} ~ {points[:, 0].max():.3f} m")
        print(f"  Y: {points[:, 1].min():.3f} ~ {points[:, 1].max():.3f} m")
        print(f"  Z: {points[:, 2].min():.3f} ~ {points[:, 2].max():.3f} m")
        
        # 保存点云
        if args.save:
            save_pointcloud(points, colors, args.save)
        
        # 渲染/可视化
        if args.render:
            # 直接保存图片，不需要显示窗口
            visualize_pointcloud_matplotlib(points, colors, subsample=args.subsample,
                                            save_path=args.render, show=False, view=args.view)
        else:
            # 尝试交互式可视化
            if HAS_OPEN3D:
                try:
                    visualize_pointcloud_open3d(points, colors, 
                                                window_name=f"Frame {args.frame} - {len(points)} points")
                except Exception as e:
                    print(f"Open3D 可视化失败: {e}")
                    print("自动保存点云图片...")
                    output_img = f"pointcloud_frame{args.frame}.png"
                    visualize_pointcloud_matplotlib(points, colors, subsample=args.subsample,
                                                    save_path=output_img, show=False, view=args.view)
            else:
                try:
                    visualize_pointcloud_matplotlib(points, colors, subsample=args.subsample, 
                                                    view=args.view)
                except Exception as e:
                    print(f"matplotlib 显示失败: {e}")
                    print("自动保存点云图片...")
                    output_img = f"pointcloud_frame{args.frame}.png"
                    visualize_pointcloud_matplotlib(points, colors, subsample=args.subsample,
                                                    save_path=output_img, show=False, view=args.view)


if __name__ == "__main__":
    main()

