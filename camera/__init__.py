"""Camera adapters for real and future simulated image sources."""

from .camera_base import CameraBase, CameraInfo
from .calibration import LensUndistorter
from .camera_pair import CameraPairInfo, FramePair, OpenCVCameraPair
from .opencv_camera import OpenCVCamera, CameraOpenError, discover_cameras

__all__ = [
    "CameraBase",
    "CameraInfo",
    "LensUndistorter",
    "CameraPairInfo",
    "FramePair",
    "OpenCVCameraPair",
    "OpenCVCamera",
    "CameraOpenError",
    "discover_cameras",
]
