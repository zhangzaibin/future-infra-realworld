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

from dataclasses import dataclass

from ..configs import CameraConfig, ColorMode, Cv2Rotation

__all__ = ["OrbbecAstraCameraConfig", "ColorMode", "Cv2Rotation"]


class OrbbecOutputMode:
    """Output mode for Orbbec camera.
    
    - COLOR: Output RGB color image
    - DEPTH: Output depth as colorized image (blue=near, red=far) for visualization
    - DEPTH_RAW: Output raw depth encoded in RGB channels (lossless, can be decoded)
    
    To decode DEPTH_RAW back to millimeters:
        depth_mm = image[:, :, 0].astype(np.uint16) * 256 + image[:, :, 1].astype(np.uint16)
    """
    COLOR = "color"
    DEPTH = "depth"
    DEPTH_RAW = "depth_raw"


@CameraConfig.register_subclass("orbbec_astra")
@dataclass
class OrbbecAstraCameraConfig(CameraConfig):
    """Configuration class for Orbbec Astra cameras with RGB and depth support.

    This class provides configuration options for Orbbec Astra cameras accessed
    via shared memory from the orbbec_shm_capture program.

    Example configurations:
    ```python
    # RGB only camera
    OrbbecAstraCameraConfig(device_index=0, fps=30, width=640, height=480, output_mode="color")

    # Depth camera (colorized for visualization)
    OrbbecAstraCameraConfig(device_index=0, fps=30, width=640, height=480, output_mode="depth")
    ```

    For data collection, configure two cameras:
    ```
    --robot.cameras="{
      orbbec_rgb: {type: orbbec_astra, width: 640, height: 480, fps: 30, output_mode: color},
      orbbec_depth: {type: orbbec_astra, width: 640, height: 480, fps: 30, output_mode: depth}
    }"
    ```

    Attributes:
        device_index: The device index (usually 0).
        fps: Requested frames per second.
        width: Frame width in pixels.
        height: Frame height in pixels.
        color_mode: Color mode for RGB output (RGB or BGR). Defaults to RGB.
        output_mode: What to output - "color" for RGB, "depth" for colorized depth. Defaults to "color".
        use_depth: Internal flag, set automatically based on output_mode.
        rotation: Image rotation setting. Defaults to no rotation.
        warmup_s: Warmup time in seconds.

    Note:
        - Requires orbbec_shm_capture to be running first.
        - Depth mode outputs a colorized depth map (blue=near, red=far) as RGB.
    """

    device_index: int | str = 0
    color_mode: ColorMode = ColorMode.RGB
    output_mode: str = "color"  # "color", "depth", or "depth_raw"
    use_depth: bool = True
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 2

    def __post_init__(self) -> None:
        if self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"`color_mode` is expected to be {ColorMode.RGB.value} or {ColorMode.BGR.value}, but {self.color_mode} is provided."
            )

        if self.rotation not in (
            Cv2Rotation.NO_ROTATION,
            Cv2Rotation.ROTATE_90,
            Cv2Rotation.ROTATE_180,
            Cv2Rotation.ROTATE_270,
        ):
            raise ValueError(
                f"`rotation` is expected to be in {(Cv2Rotation.NO_ROTATION, Cv2Rotation.ROTATE_90, Cv2Rotation.ROTATE_180, Cv2Rotation.ROTATE_270)}, but {self.rotation} is provided."
            )

