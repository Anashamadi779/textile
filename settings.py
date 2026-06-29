import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ── LandingLens ───────────────────────────────────────────────────────────────
ENDPOINT_ID: str = os.environ["LANDINGLENS_ENDPOINT_ID"].strip()
API_KEY: str     = os.environ["LANDINGLENS_API_KEY"].strip()

# ── Mistral (zero-shot defect typing) ─────────────────────────────────────────
# Used only when LandingLens flags a defect, to name the defect type. Optional:
# if the key is missing, typing is skipped and the rest of the app still works.
MISTRAL_API_KEY: str = os.environ.get("MISTRAL_API_KEY", "").strip()
MISTRAL_MODEL: str   = os.environ.get("MISTRAL_MODEL", "pixtral-12b-latest").strip()

# The candidate defect types Mistral chooses from (zero-shot, no training).
DEFECT_TYPES: list[str] = [
    "hole",
    "stain",
    "crease",
]

# ── Email (envoi automatique du rapport PDF via l'API HTTP Brevo) ─────────────
# Beaucoup de réseaux (campus/ESITH) bloquent le SMTP sortant (ports 587/465/2525).
# On utilise donc l'API HTTPS de Brevo (port 443, qui passe). BREVO_API_KEY = clé
# API v3 (format xkeysib-...), à placer dans .env — jamais dans le code.
# MAIL_FROM doit être un expéditeur VÉRIFIÉ dans le compte Brevo.
# (Repli sur SMTP_PASSWORD si la clé a déjà été mise sous ce nom.)
BREVO_API_KEY: str = (os.environ.get("BREVO_API_KEY")
                      or os.environ.get("SMTP_PASSWORD", "")).strip()
MAIL_FROM: str     = os.environ.get("MAIL_FROM", "anas.hamadi@esith.net").strip()
MAIL_TO: str       = os.environ.get("MAIL_TO", "anashamadi522@gmail.com").strip()

# ── Camera / inference ────────────────────────────────────────────────────────
FRAME_WIDTH:          int   = 640
CONFIDENCE_THRESHOLD: float = 0.75
CAPTURE_INTERVAL_SEC: float = 5.0
WINDOW_NAME:          str   = "Fabric Quality Agent"

# Only labels that mean a defect. "ok" / "no_defect" must NOT be here.
DEFECT_LABELS: set[str] = {"Defect"}

