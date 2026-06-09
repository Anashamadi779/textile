"""
scripts/train_dino.py
=====================
Script principal de fine-tuning DINOv2 pour défauts fins.

Usage :
    python scripts/train_dino.py
    python scripts/train_dino.py --data data/confection_cls --epochs 50 --device cuda
    python scripts/train_dino.py --load checkpoints/dino_best.pth --predict image.jpg
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.dino_classifier import DINOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="QualityAgent — Fine-tuning DINOv2")
    parser.add_argument('--data',    default='C:\\Users\\anash\\Downloads\\qualityagent_phase2\\qualityagent_phase2\\data\\confection.yaml', help='Dossier dataset classification')
    parser.add_argument('--epochs',  type=int, default=30)
    parser.add_argument('--batch',   type=int, default=32)
    parser.add_argument('--lr',      type=float, default=1e-4)
    parser.add_argument('--device',  default='auto', help="'cpu', 'cuda', 'auto'")
    parser.add_argument('--unfreeze',type=int, default=10, help='Epoch pour dégeler backbone')
    parser.add_argument('--load',    default=None, help='Charger des poids existants')
    parser.add_argument('--predict', default=None, help='C:\\Users\\anash\\Downloads\\qualityagent_phase2\\qualityagent_phase2\\data\\Fabric Defect Detection\\test\\defect\\defect_22.jpg')
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "="*60)
    print("  QualityAgent — Phase 2 : Fine-tuning DINOv2")
    print("="*60)
    print(f"  Dataset  : {args.data}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  LR       : {args.lr}")
    print(f"  Device   : {args.device}")
    print(f"  Unfreeze : epoch {args.unfreeze}")
    print("="*60 + "\n")

    trainer = DINOTrainer(
        data_dir=args.data,
        device=args.device,
        batch_size=args.batch
    )

    if args.load:
        trainer.load(args.load)

    if args.predict:
        result = trainer.predict(args.predict)
        print(f"\n  Résultat : {result['class']} ({result['confidence']:.2%})")
        return

    trainer.train(
        epochs=args.epochs,
        lr=args.lr,
        unfreeze_after=args.unfreeze
    )

    trainer.save('checkpoints/dino_final.pth')
    print("\n✅ Fine-tuning terminé.")
    print("   Meilleur modèle : checkpoints/dino_best.pth")
    print("   Modèle final    : checkpoints/dino_final.pth\n")


if __name__ == '__main__':
    main()
