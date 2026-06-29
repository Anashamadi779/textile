import cv2
import numpy as np
from loguru import logger


class Camera:
    def __init__(self, device_index: int = 0) -> None:
        self._cap = cv2.VideoCapture(device_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera at device index {device_index}")
        logger.info(f"Camera opened at device index {device_index}")

    def capture_frame(self) -> np.ndarray:
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame from camera")
        return frame

    def release(self) -> None:
        self._cap.release()
        logger.info("Camera released")

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *_) -> None:
        self.release()
