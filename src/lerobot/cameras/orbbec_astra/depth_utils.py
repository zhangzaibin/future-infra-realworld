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
Utility functions for working with Orbbec Astra depth data.

When using output_mode="depth_raw", depth values are encoded into RGB channels:
- R channel = high 8 bits of depth (mm)
- G channel = low 8 bits of depth (mm)
- B channel = unused (0)

This allows lossless storage of 16-bit depth data in standard 8-bit RGB images.
"""

import numpy as np
from numpy.typing import NDArray


def decode_depth_image(rgb_image: NDArray) -> NDArray:
    """
    Decode a depth_raw encoded RGB image back to depth values in millimeters.
    
    Args:
        rgb_image: RGB image with shape (H, W, 3) where depth is encoded as:
                   R = high 8 bits, G = low 8 bits
    
    Returns:
        np.ndarray: Depth map with shape (H, W), dtype=uint16, values in millimeters
    
    Example:
        ```python
        import cv2
        from lerobot.cameras.orbbec_astra.depth_utils import decode_depth_image
        
        # Load the saved depth_raw image
        img = cv2.imread("depth_raw_frame.png")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Decode to depth in millimeters
        depth_mm = decode_depth_image(img_rgb)
        
        # Now you can use the depth data
        print(f"Depth range: {depth_mm.min()} - {depth_mm.max()} mm")
        ```
    """
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {rgb_image.shape}")
    
    # Extract high and low bytes
    high_byte = rgb_image[:, :, 0].astype(np.uint16)
    low_byte = rgb_image[:, :, 1].astype(np.uint16)
    
    # Reconstruct 16-bit depth
    depth_mm = (high_byte << 8) | low_byte
    
    return depth_mm


def encode_depth_to_rgb(depth_mm: NDArray) -> NDArray:
    """
    Encode 16-bit depth data into RGB image for lossless storage.
    
    Args:
        depth_mm: Depth map with shape (H, W), dtype=uint16, values in millimeters
    
    Returns:
        np.ndarray: RGB image with shape (H, W, 3), dtype=uint8
    
    Example:
        ```python
        from lerobot.cameras.orbbec_astra.depth_utils import encode_depth_to_rgb
        
        # Encode depth for saving
        depth_rgb = encode_depth_to_rgb(depth_mm)
        cv2.imwrite("depth_encoded.png", cv2.cvtColor(depth_rgb, cv2.COLOR_RGB2BGR))
        ```
    """
    if depth_mm.ndim != 2:
        raise ValueError(f"Expected 2D depth map, got shape {depth_mm.shape}")
    
    depth_mm = depth_mm.astype(np.uint16)
    
    # Split into high and low bytes
    high_byte = (depth_mm >> 8).astype(np.uint8)
    low_byte = (depth_mm & 0xFF).astype(np.uint8)
    
    # Create RGB image
    rgb = np.stack([high_byte, low_byte, np.zeros_like(high_byte)], axis=-1)
    
    return rgb


def depth_to_colormap(depth_mm: NDArray, min_depth: int = 200, max_depth: int = 5000) -> NDArray:
    """
    Convert depth map to colorized visualization (blue=near, red=far).
    
    Args:
        depth_mm: Depth map in millimeters
        min_depth: Minimum depth for colormap (mm)
        max_depth: Maximum depth for colormap (mm)
    
    Returns:
        np.ndarray: Colorized RGB image
    """
    import cv2
    
    # Normalize to 0-255
    depth_clipped = np.clip(depth_mm, min_depth, max_depth)
    depth_normalized = ((depth_clipped - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)
    
    # Apply colormap
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
    
    # Set invalid depth to black
    depth_colored[depth_mm == 0] = [0, 0, 0]
    
    return depth_colored


def depth_to_meters(depth_mm: NDArray) -> NDArray:
    """Convert depth from millimeters to meters."""
    return depth_mm.astype(np.float32) / 1000.0


def depth_to_point_cloud(depth_mm: NDArray, fx: float = 570.0, fy: float = 570.0, 
                         cx: float = 320.0, cy: float = 240.0) -> NDArray:
    """
    Convert depth map to 3D point cloud.
    
    Args:
        depth_mm: Depth map in millimeters, shape (H, W)
        fx, fy: Focal lengths in pixels (default values are approximate for Astra)
        cx, cy: Principal point (default is image center for 640x480)
    
    Returns:
        np.ndarray: Point cloud with shape (H, W, 3) containing (X, Y, Z) in meters
    
    Example:
        ```python
        from lerobot.cameras.orbbec_astra.depth_utils import decode_depth_image, depth_to_point_cloud
        
        depth_mm = decode_depth_image(rgb_image)
        points = depth_to_point_cloud(depth_mm)
        
        # Get valid points
        valid_mask = depth_mm > 0
        valid_points = points[valid_mask]  # Shape: (N, 3)
        ```
    """
    h, w = depth_mm.shape
    
    # Create pixel coordinate grids
    u = np.arange(w)
    v = np.arange(h)
    u, v = np.meshgrid(u, v)
    
    # Convert to meters
    z = depth_mm.astype(np.float32) / 1000.0
    
    # Back-project to 3D
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # Stack into point cloud
    points = np.stack([x, y, z], axis=-1)
    
    return points


