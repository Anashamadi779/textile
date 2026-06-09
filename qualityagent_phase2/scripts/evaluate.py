"""
scripts/evaluate.py
====================
Évaluation complète d'un modèle QualityAgent.
Génère : métriques console + matrice de confusion + courbes PR + rapport JSON.

Usage :
    # Évaluer YOLOv8
    python scripts/evaluate.py --model runs/quality_agent/run_v1/weights/best.pt --type yolo

    # Évaluer DINOv2
    python scripts/evaluate.py --model checkpoints/dino_best.pth --type dino

    # Juste lire un results.csv YOLO
    python scripts/evaluate.py --csv runs/quality_agent/run_v1/results.csv
"""

import argparse
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="QualityAgent — Évaluation métriques")
    parser.add_argument('--model',  default=None, help='Chemin vers le modèle (.pt ou .pth)')
    parser.add_argument('--type',   default='yolo', choices=['yolo', 'dino'],
                        help='Type de modèle')
    parser.add_argument('--data',   default='data/confection.yaml',
                        help='Dataset YAML (pour YOLO)')
    parser.add_argument('--csv',    default=None,
                        help='Lire directement un results.csv YOLO')
    parser.add_argument('--split',  default='val', choices=['val', 'test'],
                        help='Jeu à évaluer')
    parser.add_argument('--conf',   type=float, default=0.25,
                        help='Seuil de confiance (YOLO)')
    parser.add_argument('--output', default='reports',
                        help='Dossier de sortie des rapports')
    return parser.parse_args()


def evaluate_yolo(model_path: str, data_yaml: str, conf: float, split: str, output_dir: str):
    """Évaluation complète d'un modèle YOLOv8."""
    from models.yolov8_trainer import YOLOv8Trainer
    from utils.metrics import summarize_yolo_results

    trainer = YOLOv8Trainer()
    metrics = trainer.validate(
        data_yaml=data_yaml,
        weights=model_path,
        conf=conf,
        split=split
    )

    # Chercher results.csv dans le dossier du modèle
    model_dir = Path(model_path).parent.parent
    results_csv = model_dir / 'results.csv'
    if results_csv.exists():
        summarize_yolo_results(str(results_csv))

    return metrics


def evaluate_dino(model_path: str, data_dir: str, output_dir: str):
    """Évaluation complète d'un modèle DINOv2 avec métriques détaillées."""
    import torch
    from models.dino_classifier import DINOTrainer, DEFECT_CLASSES
    from utils.metrics import compute_classification_metrics

    trainer = DINOTrainer(data_dir=data_dir)
    trainer.load(model_path)
    trainer.model.eval()

    all_labels, all_preds, all_scores = [], [], []

    with torch.no_grad():
        for images, labels in trainer.val_loader:
            images = images.to(trainer.device)
            logits = trainer.model(images)
            probs  = torch.softmax(logits, dim=1)
            preds  = logits.argmax(dim=1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_scores.append(probs.cpu().numpy())

    y_true   = np.array(all_labels)
    y_pred   = np.array(all_preds)
    y_scores = np.vstack(all_scores)

    report = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_scores=y_scores,
        class_names=DEFECT_CLASSES,
        output_dir=output_dir
    )
    return report


def main():
    args = parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    # ── Lecture directe d'un results.csv YOLO ─────────────────────────────
    if args.csv:
        from utils.metrics import summarize_yolo_results
        summarize_yolo_results(args.csv)
        return

    if not args.model:
        print("Erreur : --model ou --csv requis")
        return

    # ── Évaluation selon le type ──────────────────────────────────────────
    if args.type == 'yolo':
        evaluate_yolo(
            model_path=args.model,
            data_yaml=args.data,
            conf=args.conf,
            split=args.split,
            output_dir=args.output
        )

    elif args.type == 'dino':
        evaluate_dino(
            model_path=args.model,
            data_dir=Path(args.data).parent / 'confection_cls',
            output_dir=args.output
        )

    print(f"\n📁 Rapports générés dans : {args.output}/")


if __name__ == '__main__':
    main()
