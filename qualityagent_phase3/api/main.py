"""
api/main.py
===========
API FastAPI — détection de défauts textiles via YOLOv8.

Endpoint principal :
    POST /analyze  — reçoit une image, retourne les défauts détectés en JSON.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO

# ─── Constantes ───────────────────────────────────────────────────────────────

MODEL_PATH = Path(__file__).resolve().parents[2] / (
    "qualityagent_phase2/runs/quality_agent/run_v1/weights/best.pt"
)

DEFECT_CLASSES = [
    "couture_lache",
    "accroc",
    "trou",
    "bouton_manquant",
    "couleur_aberrante",
    "fil_apparent",
    "pli_permanent",
]

# ─── Application ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="QualityAgent — Défauts Textiles",
    description="Détection automatique de défauts en confection via YOLOv8.",
    version="3.0.0",
)

# Modèle chargé une seule fois au démarrage (singleton)
_model: Optional[YOLO] = None


FALLBACK_MODEL = "yolov8n.pt"  # téléchargé automatiquement par ultralytics si absent

_model_source: str = "none"


def get_model() -> YOLO:
    global _model, _model_source
    if _model is None:
        if MODEL_PATH.exists():
            _model = YOLO(str(MODEL_PATH))
            _model_source = str(MODEL_PATH)
            print(f"[INFO] Modèle chargé : {MODEL_PATH}")
        else:
            print(f"[WARNING] {MODEL_PATH} introuvable — fallback sur {FALLBACK_MODEL}")
            _model = YOLO(FALLBACK_MODEL)
            _model_source = FALLBACK_MODEL
    return _model


@app.on_event("startup")
def load_model_on_startup() -> None:
    """Pré-charge le modèle au démarrage pour éviter la latence au premier appel."""
    get_model()


# ─── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(..., description="Image à analyser (JPEG, PNG, BMP…)"),
    conf: float = Query(0.25, ge=0.01, le=1.0, description="Seuil de confiance"),
    iou: float = Query(0.45, ge=0.01, le=1.0, description="Seuil IoU pour NMS"),
) -> JSONResponse:
    """
    Analyse une image et retourne les défauts textiles détectés.

    Retourne une liste de détections, chacune contenant :
    - `classe`      : nom du défaut
    - `classe_id`   : index de la classe (0–6)
    - `confiance`   : score de confiance (0.0–1.0)
    - `bbox`        : bounding box en pixels `{x1, y1, x2, y2}`
    - `bbox_norm`   : bounding box normalisée (0.0–1.0)
    """
    # Validation du type de fichier
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail=f"Type de fichier non supporté : {file.content_type}. Envoyez une image.",
        )

    # Lecture et décodage de l'image
    try:
        raw = await file.read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Impossible de décoder l'image : {exc}")

    img_w, img_h = image.size
    img_array = np.array(image)

    # Inférence YOLOv8
    try:
        model = get_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    results = model.predict(
        source=img_array,
        conf=conf,
        iou=iou,
        verbose=False,
    )

    # Construction de la réponse
    detections: list[dict] = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = DEFECT_CLASSES[cls_id] if cls_id < len(DEFECT_CLASSES) else str(cls_id)
            confidence = round(float(box.conf[0]), 4)

            x1, y1, x2, y2 = (round(float(v), 2) for v in box.xyxy[0])

            detections.append(
                {
                    "classe": result.names.get(cls_id, cls_name),
                    "classe_id": cls_id,
                    "confiance": confidence,
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "bbox_norm": {
                        "x1": round(x1 / img_w, 4),
                        "y1": round(y1 / img_h, 4),
                        "x2": round(x2 / img_w, 4),
                        "y2": round(y2 / img_h, 4),
                    },
                }
            )

    # Tri par confiance décroissante
    detections.sort(key=lambda d: d["confiance"], reverse=True)

    return JSONResponse(
        {
            "image": file.filename,
            "largeur_px": img_w,
            "hauteur_px": img_h,
            "nb_defauts": len(detections),
            "defauts": detections,
            "model_source": _model_source,
        }
    )
