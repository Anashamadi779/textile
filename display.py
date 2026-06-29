import sys

import cv2
import numpy as np
from loguru import logger

from interpreter import InterpretedResult

_GREEN = (0, 200, 0)
_RED = (0, 0, 220)
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_SCALE = 0.85
_THICKNESS = 2


def _check_gui() -> None:
    try:
        cv2.namedWindow("__test__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__test__")
    except cv2.error:
        logger.error(
            "OpenCV was built without GUI support (headless build detected).\n"
            "Fix: pip uninstall opencv-python opencv-python-headless -y  &&  pip install opencv-python"
        )
        sys.exit(1)


def overlay_result(frame: np.ndarray, result: InterpretedResult) -> np.ndarray:
    prefix = "OK " if result.is_ok else "DEFECT "
    text = prefix + str(result)
    color = _GREEN if result.is_ok else _RED

    cv2.putText(frame, text, (11, 36), _FONT, _SCALE, (0, 0, 0), _THICKNESS + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (10, 35), _FONT, _SCALE, color, _THICKNESS, cv2.LINE_AA)
    return frame


def show(window_name: str, frame: np.ndarray) -> None:
    cv2.imshow(window_name, frame)


def wait_key(ms: int = 1) -> int:
    return cv2.waitKey(ms) & 0xFF


def destroy_windows() -> None:
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass
