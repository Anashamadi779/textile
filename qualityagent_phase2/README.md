# QualityAgent — Phase 2 : Modèle de Vision IA

Détection et classification de défauts visuels sur tissu.

## Structure
```
qualityagent_phase2/
├── data/
|   └── Fabric Defect Detection
|       └── test
|           └── defect
|           └── ok
|       └── train
|           └── defect
|           └── ok
│   └── confection.yaml          # Config dataset YOLO
├── models/
│   ├── yolov8_trainer.py        # Entraînement YOLOv8
│   └── dino_classifier.py       # Fine-tuning DINOv2
├── utils/
│   ├── augmentation.py          # Pipeline augmentation tissu
│   └── metrics.py               # Calcul rappel / précision / F1 / mAP
└── scripts/
    ├── train_yolo.py            # Script principal YOLOv8
    ├── train_dino.py            # Script principal DINOv2
    └── evaluate.py              # Évaluation & rapport métriques 
```

## Installation
```bash
pip install ultralytics transformers torch torchvision albumentations scikit-learn matplotlib
```

## Lancement rapide
```bash
# 1. Entraîner YOLOv8
python scripts/train_yolo.py

# 2. Évaluer les métriques
python scripts/evaluate.py --model runs/quality_agent/run_v1/weights/best.pt

# 3. Fine-tuning DINOv2 (défauts fins)
python scripts/train_dino.py
```

## Classes de défauts
  0. defect
  1. ok
