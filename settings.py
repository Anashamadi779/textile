import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ── LandingLens ───────────────────────────────────────────────────────────────
ENDPOINT_ID: str = os.environ["LANDINGLENS_ENDPOINT_ID"].strip()
API_KEY: str     = os.environ["LANDINGLENS_API_KEY"].strip()

# ── Camera / inference ────────────────────────────────────────────────────────
FRAME_WIDTH:          int   = 640
CONFIDENCE_THRESHOLD: float = 0.75
CAPTURE_INTERVAL_SEC: float = 1.0
WINDOW_NAME:          str   = "Fabric Quality Agent"

# Only labels that mean a defect. "ok" / "no_defect" must NOT be here.
DEFECT_LABELS: set[str] = {"Defect"}

