"""
scripts/train_yolo.py
=====================
Script principal d'entraînement YOLOv8 pour QualityAgent.

Usage :
    python scripts/train_yolo.py
    python scripts/train_yolo.py --epochs 200 --batch 8 --size s
    python scripts/train_yolo.py --resume runs/quality_agent/run_v1
"""

import argparse
import sys
from pathlib import Path

# Rendre les modules du projet accessibles
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.yolov8_trainer import YOLOv8Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="QualityAgent — Entraînement YOLOv8")
    parser.add_argument('--data',    default='C:\\Users\\anash\\Downloads\\qualityagent_phase2\\qualityagent_phase2\\data\\confection.yaml', help='Dataset YAML')
    parser.add_argument('--size',    default='m', choices=['n','s','m','l','x'],
                        help='Taille modèle (n=nano, m=medium, x=extra-large)')
    parser.add_argument('--epochs',  type=int, default=100)
    parser.add_argument('--batch',   type=int, default=16)
    parser.add_argument('--imgsz',   type=int, default=640)
    parser.add_argument('--device',  default='', help="'cpu', '0', '0,1', '' = auto")
    parser.add_argument('--project', default='runs/quality_agent')
    parser.add_argument('--name',    default='run_v1')
    parser.add_argument('--patience',type=int, default=20, help='Early stopping patience')
    parser.add_argument('--resume',  default=None, help='Reprendre depuis un run existant')
    parser.add_argument('--export',  action='store_true', help='Exporter en ONNX après entraînement')
    parser.add_argument('--validate',action='store_true', help='Valider après entraînement')
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "="*60)
    print("  QualityAgent — Phase 2 : Entraînement YOLOv8")
    print("="*60)
    print(f"  Modèle   : YOLOv8{args.size}")
    print(f"  Dataset  : {args.data}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Batch    : {args.batch}")
    print(f"  Device   : {args.device or 'auto'}")
    print("="*60 + "\n")

    trainer = YOLOv8Trainer(
        model_size=args.size,
        project_dir=args.project,
        run_name=args.name
    )

    # Entraînement (ou reprise)
    if args.resume:
        print(f"[train_yolo] Reprise depuis : {args.resume}")
        from ultralytics import YOLO
        model = YOLO(f"{args.resume}/weights/last.pt")
        results = model.train(resume=True)
    else:
        results = trainer.train(
            data_yaml=args.data,
            epochs=args.epochs,
            batch=args.batch,
            image_size=args.imgsz,
            device=args.device,
            patience=args.patience
        )

    # Validation optionnelle
    if args.validate:
        print("\n[train_yolo] Validation du modèle ...")
        trainer.validate(data_yaml=args.data, split='val')

    # Export ONNX optionnel
    if args.export:
        print("\n[train_yolo] Export ONNX ...")
        export_path = trainer.export(format='onnx', half=True)
        print(f"  Modèle production : {export_path}")

    print("\n✅ Pipeline terminé.")
    print(f"   Weights : {args.project}/{args.name}/weights/best.pt")
    print(f"   Logs    : {args.project}/{args.name}/results.csv\n")


if __name__ == '__main__':
    main()
