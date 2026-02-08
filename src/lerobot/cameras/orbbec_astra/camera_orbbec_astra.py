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
Provides the OrbbecAstraCamera class for capturing frames from Orbbec Astra cameras.
Uses shared memory to receive frames from the orbbec_shm_capture C program.

To use this camera:
1. Build the orbbec_shm_capture program from ~/AstraSDK/samples/lerobot_bridge/
2. Run orbbec_shm_capture in a separate terminal before starting data collection
3. Configure the camera in lerobot with type: orbbec_astra
"""

import ctypes
import logging
import mmap
import os
import struct
import time
from threading import Event, Lock, Thread
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..camera import Camera
from ..configs import ColorMode
from ..utils import get_cv2_rotation
from .configuration_orbbec_astra import OrbbecAstraCameraConfig, OrbbecOutputMode

logger = logging.getLogger(__name__)

# Depth colorization parameters
DEPTH_MIN_MM = 200   # Minimum depth in mm (closer = blue)
DEPTH_MAX_MM = 5000  # Maximum depth in mm (farther = red)

# Shared memory names (must match C program)
SHM_NAME_COLOR = "/orbbec_color"
SHM_NAME_DEPTH = "/orbbec_depth"
SHM_NAME_META = "/orbbec_meta"

# Default dimensions
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
COLOR_SIZE = DEFAULT_WIDTH * DEFAULT_HEIGHT * 3  # RGB
DEPTH_SIZE = DEFAULT_WIDTH * DEFAULT_HEIGHT * 2  # uint16


class FrameMeta(ctypes.Structure):
    """Metadata structure matching the C program"""
    _fields_ = [
        ("frame_id", ctypes.c_uint64),
        ("timestamp_ns", ctypes.c_uint64),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("color_ready", ctypes.c_int),
        ("depth_ready", ctypes.c_int),
        ("running", ctypes.c_int),
    ]


class OrbbecAstraCamera(Camera):
    """
    Manages camera interactions with Orbbec Astra cameras via shared memory.

    This camera requires the orbbec_shm_capture program to be running separately.
    The C program captures frames from the Astra camera and writes them to shared
    memory, which this class reads.

    Setup:
        1. Build the capture program:
           cd ~/AstraSDK/samples/lerobot_bridge
           mkdir build && cd build
           cmake .. && make
        
        2. Run the capture program (in a separate terminal):
           export LD_LIBRARY_PATH=~/AstraSDK/lib:$LD_LIBRARY_PATH
           ~/AstraSDK/samples/lerobot_bridge/build/orbbec_shm_capture
        
        3. Use in lerobot-record:
           --robot.cameras="{ orbbec: {type: orbbec_astra, width: 640, height: 480, fps: 30} }"

    Example:
        ```python
        from lerobot.cameras.orbbec_astra import OrbbecAstraCamera, OrbbecAstraCameraConfig

        config = OrbbecAstraCameraConfig(fps=30, width=640, height=480)
        camera = OrbbecAstraCamera(config)
        camera.connect()

        # Read RGB frame
        color_image = camera.read()
        print(color_image.shape)

        # Read depth frame
        depth_map = camera.read_depth()
        print(depth_map.shape)

        camera.disconnect()
        ```
    """

    def __init__(self, config: OrbbecAstraCameraConfig):
        """Initialize the OrbbecAstraCamera instance."""
        super().__init__(config)

        self.config = config
        self.device_index = config.device_index

        self.fps = config.fps
        self.color_mode = config.color_mode
        self.output_mode = config.output_mode
        self.use_depth = config.use_depth
        self.warmup_s = config.warmup_s

        # Shared memory file descriptors and mappings
        self.shm_color_fd = None
        self.shm_depth_fd = None
        self.shm_meta_fd = None
        self.shm_color = None
        self.shm_depth = None
        self.shm_meta = None

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: NDArray[Any] | None = None
        self.latest_depth: NDArray[Any] | None = None
        self.new_frame_event: Event = Event()

        self.rotation: int | None = get_cv2_rotation(config.rotation)

        self._connected = False
        self._last_frame_id = -1

        if self.height and self.width:
            self.capture_width, self.capture_height = self.width, self.height
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.capture_width, self.capture_height = self.height, self.width

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(shm)"

    @property
    def is_connected(self) -> bool:
        """Check if connected to shared memory."""
        return self._connected and self.shm_meta is not None

    def connect(self, warmup: bool = True) -> None:
        """
        Connect to the shared memory created by orbbec_shm_capture.

        Raises:
            DeviceAlreadyConnectedError: If already connected.
            ConnectionError: If shared memory is not available (capture program not running).
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        try:
            # Open shared memory for color
            self.shm_color_fd = os.open(f"/dev/shm{SHM_NAME_COLOR}", os.O_RDONLY)
            self.shm_color = mmap.mmap(self.shm_color_fd, COLOR_SIZE, mmap.MAP_SHARED, mmap.PROT_READ)

            # Open shared memory for depth
            self.shm_depth_fd = os.open(f"/dev/shm{SHM_NAME_DEPTH}", os.O_RDONLY)
            self.shm_depth = mmap.mmap(self.shm_depth_fd, DEPTH_SIZE, mmap.MAP_SHARED, mmap.PROT_READ)

            # Open shared memory for metadata
            self.shm_meta_fd = os.open(f"/dev/shm{SHM_NAME_META}", os.O_RDONLY)
            self.shm_meta = mmap.mmap(self.shm_meta_fd, ctypes.sizeof(FrameMeta), mmap.MAP_SHARED, mmap.PROT_READ)

        except FileNotFoundError:
            self._cleanup_shm()
            raise ConnectionError(
                f"Failed to connect to {self}. Make sure orbbec_shm_capture is running.\n"
                "Steps to start the capture program:\n"
                "  1. cd ~/AstraSDK/samples/lerobot_bridge/build\n"
                "  2. export LD_LIBRARY_PATH=~/AstraSDK/lib:$LD_LIBRARY_PATH\n"
                "  3. ./orbbec_shm_capture"
            )

        # Verify the capture program is running
        meta = self._read_meta()
        if meta is None or not meta.running:
            self._cleanup_shm()
            raise ConnectionError(
                f"Shared memory exists but orbbec_shm_capture is not running. Please start it first."
            )

        self._connected = True

        # Set dimensions from metadata
        if self.width is None:
            self.width = meta.width
        if self.height is None:
            self.height = meta.height
        self.capture_width = self.width
        self.capture_height = self.height

        if warmup:
            start_time = time.time()
            while time.time() - start_time < self.warmup_s:
                try:
                    self.read()
                    if self.use_depth:
                        self.read_depth()
                except RuntimeError:
                    pass
                time.sleep(0.1)

        logger.info(f"{self} connected.")

    def _cleanup_shm(self):
        """Clean up shared memory mappings."""
        if self.shm_color:
            self.shm_color.close()
            self.shm_color = None
        if self.shm_depth:
            self.shm_depth.close()
            self.shm_depth = None
        if self.shm_meta:
            self.shm_meta.close()
            self.shm_meta = None

        if self.shm_color_fd is not None:
            os.close(self.shm_color_fd)
            self.shm_color_fd = None
        if self.shm_depth_fd is not None:
            os.close(self.shm_depth_fd)
            self.shm_depth_fd = None
        if self.shm_meta_fd is not None:
            os.close(self.shm_meta_fd)
            self.shm_meta_fd = None

    def _read_meta(self) -> FrameMeta | None:
        """Read metadata from shared memory."""
        if self.shm_meta is None:
            return None
        self.shm_meta.seek(0)
        data = self.shm_meta.read(ctypes.sizeof(FrameMeta))
        return FrameMeta.from_buffer_copy(data)

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """Check if the shared memory capture is available."""
        found = []
        try:
            if os.path.exists(f"/dev/shm{SHM_NAME_META}"):
                found.append({
                    "name": "Orbbec Astra (shared memory)",
                    "type": "OrbbecAstra",
                    "id": "shm",
                    "default_stream_profile": {
                        "width": DEFAULT_WIDTH,
                        "height": DEFAULT_HEIGHT,
                        "fps": 30,
                    },
                })
        except Exception:
            pass
        return found

    def read_depth(self, timeout_ms: int = 200) -> NDArray[Any]:
        """
        Read depth frame from shared memory.

        Returns:
            np.ndarray: Depth map as uint16 (millimeters).
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if not self.use_depth:
            raise RuntimeError(f"Depth stream is not enabled for {self}.")

        start_time = time.perf_counter()
        timeout_s = timeout_ms / 1000.0

        while time.perf_counter() - start_time < timeout_s:
            meta = self._read_meta()
            if meta and meta.depth_ready and meta.frame_id != self._last_frame_id:
                self.shm_depth.seek(0)
                depth_data = np.frombuffer(self.shm_depth.read(DEPTH_SIZE), dtype=np.uint16)
                depth_map = depth_data.reshape((DEFAULT_HEIGHT, DEFAULT_WIDTH))

                if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
                    depth_map = cv2.rotate(depth_map, self.rotation)

                return depth_map.copy()
            time.sleep(0.001)

        raise RuntimeError(f"{self} depth read timeout.")

    def _colorize_depth(self, depth_map: NDArray[Any]) -> NDArray[Any]:
        """
        Convert depth map to colorized RGB image for visualization.
        Uses a blue-to-red colormap where blue=near, red=far.
        """
        # Normalize depth to 0-255 range
        depth_clipped = np.clip(depth_map, DEPTH_MIN_MM, DEPTH_MAX_MM)
        depth_normalized = ((depth_clipped - DEPTH_MIN_MM) / (DEPTH_MAX_MM - DEPTH_MIN_MM) * 255).astype(np.uint8)
        
        # Apply colormap (COLORMAP_JET: blue=near, red=far)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        
        # Convert BGR to RGB
        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
        
        # Set invalid depth (0) to black
        depth_colored[depth_map == 0] = [0, 0, 0]
        
        return depth_colored

    def read(self, color_mode: ColorMode | None = None) -> NDArray[Any]:
        """
        Read frame from shared memory based on output_mode.

        Returns:
            np.ndarray: 
                - If output_mode="color": RGB/BGR color image
                - If output_mode="depth": Colorized depth map as RGB image
                - If output_mode="depth_raw": Raw depth as grayscale (scaled to uint8)
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start_time = time.perf_counter()
        timeout_s = 0.2  # 200ms timeout

        while time.perf_counter() - start_time < timeout_s:
            meta = self._read_meta()
            if meta and meta.frame_id != self._last_frame_id:
                # Check if the required data is ready
                if self.output_mode == OrbbecOutputMode.COLOR and not meta.color_ready:
                    time.sleep(0.001)
                    continue
                if self.output_mode in [OrbbecOutputMode.DEPTH, OrbbecOutputMode.DEPTH_RAW] and not meta.depth_ready:
                    time.sleep(0.001)
                    continue

                self._last_frame_id = meta.frame_id

                if self.output_mode == OrbbecOutputMode.COLOR:
                    # Return RGB color image
                    self.shm_color.seek(0)
                    color_data = np.frombuffer(self.shm_color.read(COLOR_SIZE), dtype=np.uint8)
                    frame = color_data.reshape((DEFAULT_HEIGHT, DEFAULT_WIDTH, 3))

                    # Apply color mode conversion
                    requested_mode = self.color_mode if color_mode is None else color_mode
                    if requested_mode == ColorMode.BGR:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                elif self.output_mode == OrbbecOutputMode.DEPTH:
                    # Return colorized depth map as RGB image
                    self.shm_depth.seek(0)
                    depth_data = np.frombuffer(self.shm_depth.read(DEPTH_SIZE), dtype=np.uint16)
                    depth_map = depth_data.reshape((DEFAULT_HEIGHT, DEFAULT_WIDTH))
                    frame = self._colorize_depth(depth_map)

                elif self.output_mode == OrbbecOutputMode.DEPTH_RAW:
                    # Encode 16-bit depth into RGB channels for lossless storage
                    # R channel = high 8 bits, G channel = low 8 bits, B = 0
                    # To recover: depth_mm = R * 256 + G
                    self.shm_depth.seek(0)
                    depth_data = np.frombuffer(self.shm_depth.read(DEPTH_SIZE), dtype=np.uint16)
                    depth_map = depth_data.reshape((DEFAULT_HEIGHT, DEFAULT_WIDTH))
                    
                    # Split into high and low bytes
                    high_byte = (depth_map >> 8).astype(np.uint8)  # High 8 bits
                    low_byte = (depth_map & 0xFF).astype(np.uint8)  # Low 8 bits
                    
                    # Create RGB image: R=high, G=low, B=0
                    frame = np.stack([high_byte, low_byte, np.zeros_like(high_byte)], axis=-1)

                else:
                    raise ValueError(f"Unknown output_mode: {self.output_mode}")

                # Apply rotation
                if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
                    frame = cv2.rotate(frame, self.rotation)

                return frame.copy()
            time.sleep(0.001)

        raise RuntimeError(f"{self} read timeout (output_mode={self.output_mode}).")

    def _read_loop(self) -> None:
        """Background read loop for async reading."""
        if self.stop_event is None:
            raise RuntimeError(f"{self}: stop_event not initialized.")

        while not self.stop_event.is_set():
            try:
                color_image = self.read()
                depth_image = None
                if self.use_depth:
                    try:
                        depth_image = self.read_depth()
                    except RuntimeError:
                        pass

                with self.frame_lock:
                    self.latest_frame = color_image
                    if depth_image is not None:
                        self.latest_depth = depth_image
                self.new_frame_event.set()

            except DeviceNotConnectedError:
                break
            except Exception as e:
                logger.warning(f"Error in background thread for {self}: {e}")

    def _start_read_thread(self) -> None:
        """Start background read thread."""
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.1)
        if self.stop_event is not None:
            self.stop_event.set()

        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, name=f"{self}_read_loop")
        self.thread.daemon = True
        self.thread.start()

    def _stop_read_thread(self) -> None:
        """Stop background read thread."""
        if self.stop_event is not None:
            self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.thread = None
        self.stop_event = None

    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        """Read the latest frame asynchronously."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(f"Timeout waiting for frame from {self}.")

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"No frame available from {self}.")

        return frame

    def async_read_depth(self, timeout_ms: float = 200) -> NDArray[Any]:
        """Read the latest depth frame asynchronously."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if not self.use_depth:
            raise RuntimeError(f"Depth not enabled for {self}.")

        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(f"Timeout waiting for depth from {self}.")

        with self.frame_lock:
            depth = self.latest_depth
            self.new_frame_event.clear()

        if depth is None:
            raise RuntimeError(f"No depth frame available from {self}.")

        return depth

    def disconnect(self) -> None:
        """Disconnect from shared memory."""
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(f"{self} not connected.")

        if self.thread is not None:
            self._stop_read_thread()

        self._cleanup_shm()
        self._connected = False

        logger.info(f"{self} disconnected.")
