from PIL import Image
from landingai.predict import Predictor
from loguru import logger


class FabricPredictor:
    def __init__(self, endpoint_id: str, api_key: str) -> None:
        self._predictor = Predictor(endpoint_id, api_key=api_key)
        logger.info(f"LandingLens Predictor initialised for endpoint: {endpoint_id}")

    def predict(self, image: Image.Image) -> list:
        try:
            predictions = self._predictor.predict(image)
            logger.debug(f"Raw predictions received: {predictions}")
            return predictions
        except Exception as exc:
            logger.error(f"Prediction request failed: {exc}")
            return []
