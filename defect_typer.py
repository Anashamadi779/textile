"""Second-stage fabric inspection via the Mistral API (Pixtral vision model).

Mistral acts as a SECOND FILTER after LandingLens: for every image it returns its
own verdict (conforming / defective) with a confidence, AND — when defective — the
defect type from settings.DEFECT_TYPES (zero-shot, no training). The caller fuses
this verdict with LandingLens by keeping the higher-confidence answer.
"""

import base64
import io
import json
import re
from dataclasses import dataclass
from typing import Optional

from PIL import Image
from loguru import logger

try:
    # SDK >= 2.x moved the client under mistralai.client; older 1.x exposes it
    # at the top level. Support both.
    try:
        from mistralai import Mistral
    except ImportError:
        from mistralai.client import Mistral
except ImportError:  # SDK not installed at all
    Mistral = None


@dataclass
class MistralVerdict:
    is_ok:       bool
    defect_type: Optional[str]
    confidence:  float


class DefectTyper:
    def __init__(self, api_key: str, model_name: str, defect_types: list[str]) -> None:
        self._types = defect_types
        self._model_name = model_name
        self._enabled = bool(api_key) and Mistral is not None

        if not api_key:
            logger.warning("MISTRAL_API_KEY not set — Mistral filter disabled.")
            self._client = None
            return
        if Mistral is None:
            logger.warning("mistralai not installed — Mistral filter disabled. "
                           "Run: pip install mistralai")
            self._client = None
            return

        self._client = Mistral(api_key=api_key)
        logger.info(f"Mistral inspector ready (model: {model_name})")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _build_type_prompt(self) -> str:
        """Prompt FORCÉ : le tissu est supposé défectueux, renvoyer seulement le type."""
        options = ", ".join(self._types)
        return (
            "You are a textile quality-control inspector. The fabric in this image "
            "has been flagged as DEFECTIVE. Identify the single most likely type of "
            f"defect.\n\nRespond with EXACTLY ONE of these labels and nothing else: "
            f"{options}.\nDo not add any explanation, punctuation, or extra words."
        )

    def _build_prompt(self) -> str:
        options = ", ".join(self._types)
        return (
            "You are a textile quality-control inspector. Examine the fabric in this "
            "image and decide whether it is DEFECTIVE or CONFORMING (no visible "
            "defect). If it is defective, identify the single most likely defect "
            f"type among: {options}.\n\n"
            "Respond ONLY with a compact JSON object and nothing else, exactly:\n"
            '{"defect": true|false, "type": "<one of the types or null>", '
            '"confidence": <number between 0 and 1>}\n\n'
            "Rules:\n"
            '- "defect": true if a defect is visible, false if the fabric looks fine.\n'
            '- "type": the defect type if "defect" is true, otherwise null.\n'
            '- "confidence": how confident you are in this verdict, from 0 to 1.'
        )

    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        """PIL image -> base64 JPEG data URI for Mistral's image_url content."""
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def _normalize_type(self, text: Optional[str]) -> Optional[str]:
        """Map a free-text type onto one of the known DEFECT_TYPES."""
        if not text:
            return None
        cleaned = str(text).strip().lower().replace(" ", "_").strip(".,!\"' ")
        if cleaned in self._types:
            return cleaned
        for t in self._types:
            if t in cleaned:
                return t
        return None

    def inspect(self, image: Image.Image) -> Optional[MistralVerdict]:
        """Verdict de Mistral : conforme/défaut + type + confiance. None si échec."""
        if not self._enabled or self._client is None:
            return None
        try:
            response = self._client.chat.complete(
                model=self._model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self._build_prompt()},
                            {"type": "image_url", "image_url": self._encode_image(image)},
                        ],
                    }
                ],
                temperature=0.0,
            )
            raw = (response.choices[0].message.content or "").strip()
            logger.debug(f"Mistral raw reply: {raw!r}")
            data = self._parse_json(raw)
            if data is None:
                logger.warning(f"Mistral reply not parseable as JSON: {raw!r}")
                return None

            is_defect = bool(data.get("defect"))
            confidence = float(data.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))  # clamp 0..1
            defect_type = self._normalize_type(data.get("type")) if is_defect else None
            return MistralVerdict(
                is_ok=not is_defect,
                defect_type=defect_type,
                confidence=confidence,
            )
        except Exception as exc:
            logger.error(f"Mistral inspection failed: {exc}")
            return None

    @staticmethod
    def _parse_json(raw: str) -> Optional[dict]:
        """Extract the JSON object from Mistral's reply (handles ```json fences)."""
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    def classify(self, image: Image.Image) -> Optional[str]:
        """Type de défaut FORCÉ (suppose le tissu défectueux) → un des DEFECT_TYPES.

        Utilisé quand la décision finale est « défaut » : on veut toujours un type,
        même si Mistral n'a pas été consulté (ou pensait que c'était conforme).
        Renvoie None seulement si Mistral est indisponible / l'appel échoue.
        """
        if not self._enabled or self._client is None:
            return None
        try:
            response = self._client.chat.complete(
                model=self._model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self._build_type_prompt()},
                            {"type": "image_url", "image_url": self._encode_image(image)},
                        ],
                    }
                ],
                temperature=0.0,
            )
            raw = (response.choices[0].message.content or "").strip()
            logger.debug(f"Mistral forced-type raw reply: {raw!r}")
            defect_type = self._normalize_type(raw)
            if defect_type is None:
                logger.warning(f"Mistral forced-type reply unrecognized: {raw!r}")
            return defect_type
        except Exception as exc:
            logger.error(f"Mistral type classification failed: {exc}")
            return None
