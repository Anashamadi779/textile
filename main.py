import time

from loguru import logger

from camera import Camera
from display import _check_gui, destroy_windows, overlay_result, show, wait_key
from interpreter import InterpretedResult, interpret
from predictor import FabricPredictor
from preprocessor import preprocess
from settings import (
    API_KEY,
    CAPTURE_INTERVAL_SEC,
    ENDPOINT_ID,
    FRAME_WIDTH,
    WINDOW_NAME,
)


def main() -> None:
    logger.info("Starting Fabric Quality Agent")
    _check_gui()

    predictor = FabricPredictor(ENDPOINT_ID, API_KEY)

    with Camera(device_index=1) as cam:
        result            = InterpretedResult(is_ok=True, label="ok", confidence=1.0)
        last_inference_at = 0.0

        try:
            while True:
                raw_frame = cam.capture_frame()
                display_frame, pil_image = preprocess(raw_frame, target_width=FRAME_WIDTH)

                now = time.monotonic()
                if now - last_inference_at >= CAPTURE_INTERVAL_SEC:
                    predictions = predictor.predict(pil_image)
                    result      = interpret(predictions)
                    logger.info(str(result))
                    last_inference_at = now

                annotated = overlay_result(display_frame.copy(), result)
                show(WINDOW_NAME, annotated)

                if wait_key(1) == ord("q"):
                    logger.info("Quit key pressed — stopping")
                    break

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt — shutting down")
        finally:
            destroy_windows()

    logger.info("Fabric Quality Agent stopped")


if __name__ == "__main__":
    main()
