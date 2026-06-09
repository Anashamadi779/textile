"""
models/yolov8_trainer.py
========================
Entraînement YOLOv8 pour la détection de défauts en confection.

Stratégie :
  - Backbone YOLOv8m (équilibre vitesse/précision)
  - Seuil de confiance bas (0.25) → favorise le rappel
  - Pondération de la loss cls augmentée → pénalise davantage les FN
  - Export ONNX half-precision pour inférence temps réel
"""

from pathlib import Path
from ultralytics import YOLO


# ─── Classes de défauts ──────────────────────────────────────────────────────
DEFECT_CLASSES = [
    "defect",
    "ok"
]


class YOLOv8Trainer:
    """
    Gestionnaire d'entraînement YOLOv8 pour QualityAgent.

    Exemple d'utilisation :
        trainer = YOLOv8Trainer(model_size='m', project_dir='runs/quality_agent')
        trainer.train(data_yaml='data/confection.yaml', epochs=100)
        trainer.export()
        trainer.validate()
    """

    def __init__(
        self,
        model_size: str = 'm',       # n / s / m / l / x
        project_dir: str = 'runs/quality_agent',
        run_name: str = 'run_v1',
        pretrained: bool = True
    ):
        """
        Args:
            model_size  : Taille du modèle YOLO (n=nano … x=extra-large)
            project_dir : Dossier de sauvegarde des runs
            run_name    : Nom du run (pour retrouver les weights)
            pretrained  : Partir des poids COCO pré-entraînés
        """
        self.model_size  = model_size
        self.project_dir = project_dir
        self.run_name    = run_name

        weights = f"yolov8{model_size}.pt" if pretrained else f"yolov8{model_size}.yaml"
        self.model = YOLO(weights)
        print(f"[YOLOv8Trainer] Modèle chargé : {weights}")

    # ── Entraînement ─────────────────────────────────────────────────────────
    def train(
        self,
        data_yaml: str = 'data/confection.yaml',
        epochs: int = 100,
        image_size: int = 640,
        batch: int = 16,
        lr0: float = 0.001,
        lrf: float = 0.01,
        warmup_epochs: int = 5,
        patience: int = 20,
        device: str = ''           # '' = auto (GPU si dispo sinon CPU)
    ) -> object:
        """
        Lance l'entraînement avec des hyperparamètres optimisés pour le rappel.

        Args:
            data_yaml     : Chemin vers confection.yaml
            epochs        : Nombre d'epochs max
            image_size    : Taille des images (pixels)
            batch         : Taille du batch (réduire si OOM)
            lr0           : Learning rate initial
            lrf           : Learning rate final (fraction de lr0)
            warmup_epochs : Epochs de montée en chauffe
            patience      : Early stopping si pas d'amélioration (epochs)
            device        : 'cpu', '0', '0,1', '' (auto)

        Returns:
            Objet Results ultralytics
        """
        print(f"\n[YOLOv8Trainer] Début entraînement — {epochs} epochs")
        print(f"  Dataset : {data_yaml}")
        print(f"  Classes : {DEFECT_CLASSES}\n")

        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=image_size,
            batch=batch,
            device=device,

            # ── Learning rate ────────────────────────────────────────────────
            lr0=lr0,
            lrf=lrf,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=warmup_epochs,
            warmup_momentum=0.8,

            # ── Loss — favorise le rappel ────────────────────────────────────
            # cls plus élevé = pénalise davantage les erreurs de classification
            cls=1.5,
            box=7.5,
            dfl=1.5,

            # ── Seuil bas = plus de détections = meilleur rappel ─────────────
            conf=0.25,
            iou=0.45,

            # ── Augmentations (en plus d'albumentations externe) ─────────────
            flipud=0.3,
            fliplr=0.5,
            mosaic=1.0,      # Mosaïque 4 images
            mixup=0.1,       # Mixup léger
            copy_paste=0.1,  # Copy-paste défauts
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            shear=2.0,
            perspective=0.0001,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,

            # ── Early stopping & sauvegarde ──────────────────────────────────
            patience=patience,
            save=True,
            save_period=10,   # Sauvegarde toutes les 10 epochs

            # ── Projet ──────────────────────────────────────────────────────
            project=self.project_dir,
            name=self.run_name,
            exist_ok=True,

            # ── Reproductibilité ─────────────────────────────────────────────
            seed=42,
            deterministic=True,

            # ── Logs ────────────────────────────────────────────────────────
            verbose=True,
            plots=True       # Génère les courbes automatiquement
        )

        self.results = results
        print(f"\n[YOLOv8Trainer] Entraînement terminé ✅")
        print(f"  Weights : {self.project_dir}/{self.run_name}/weights/best.pt")
        return results

    # ── Validation ───────────────────────────────────────────────────────────
    def validate(
        self,
        data_yaml: str = 'C:\\Users\\anash\\Downloads\\qualityagent_phase2\\qualityagent_phase2\\data\\confection.yaml',
        weights: str = None,
        conf: float = 0.25,
        split: str = 'val'
    ) -> dict:
        """
        Évalue le modèle sur le jeu de validation ou test.

        Args:
            data_yaml : Dataset à évaluer
            weights   : Chemin vers les poids (None = dernier run)
            conf      : Seuil de confiance
            split     : 'val' ou 'test'

        Returns:
            dict avec mAP50, mAP50-95, rappel, précision
        """
        if weights:
            model = YOLO(weights)
        else:
            best = Path(self.project_dir) / self.run_name / 'weights' / 'best.pt'
            model = YOLO(str(best))

        metrics = model.val(
            data=data_yaml,
            conf=conf,
            iou=0.45,
            split=split,
            plots=True,
            save_json=True
        )

        print("\n[YOLOv8Trainer] Résultats validation :")
        print(f"  mAP@0.5       : {metrics.box.map50:.4f}")
        print(f"  mAP@0.5:0.95  : {metrics.box.map:.4f}")
        print(f"  Précision     : {metrics.box.mp:.4f}")
        print(f"  Rappel        : {metrics.box.mr:.4f}")

        if metrics.box.mr < 0.90:
            print(f"\n  ⚠️  Rappel {metrics.box.mr:.2%} < 90% — Actions recommandées :")
            print("      1. Baisser le seuil de confiance (conf=0.15)")
            print("      2. Collecter plus d'images pour les classes faibles")
            print("      3. Augmenter le poids cls dans les hyperparamètres")

        return {
            'map50':     metrics.box.map50,
            'map':       metrics.box.map,
            'precision': metrics.box.mp,
            'recall':    metrics.box.mr,
        }

    # ── Export production ────────────────────────────────────────────────────
    def export(
        self,
        weights: str = None,
        format: str = 'onnx',
        half: bool = True,
        dynamic: bool = True,
        simplify: bool = True
    ) -> str:
        """
        Exporte le modèle pour inférence en production.

        Args:
            weights  : Chemin poids (None = best du dernier run)
            format   : 'onnx' | 'tensorrt' | 'coreml' | 'tflite'
            half     : FP16 pour GPU (réduit mémoire, accélère)
            dynamic  : Batch size dynamique
            simplify : Simplifier le graphe ONNX

        Returns:
            Chemin vers le fichier exporté
        """
        if weights:
            model = YOLO(weights)
        else:
            best = Path(self.project_dir) / self.run_name / 'weights' / 'best.pt'
            model = YOLO(str(best))

        export_path = model.export(
            format=format,
            half=half,
            dynamic=dynamic,
            simplify=simplify,
            opset=17          # ONNX opset compatible TensorRT 8+
        )

        print(f"\n[YOLOv8Trainer] Modèle exporté → {export_path}")
        return str(export_path)

    # ── Inférence sur une image ──────────────────────────────────────────────
    def predict(
        self,
        source,               # chemin image, dossier, URL, numpy array
        weights: str = None,
        conf: float = 0.25,
        save: bool = True
    ) -> list:
        """
        Prédit les défauts sur une ou plusieurs images.

        Args:
            source  : Image(s) à analyser
            weights : Poids à utiliser
            conf    : Seuil de confiance
            save    : Sauvegarder les images annotées

        Returns:
            Liste de Results ultralytics
        """
        if weights:
            model = YOLO(weights)
        else:
            best = Path(self.project_dir) / self.run_name / 'weights' / 'best.pt'
            model = YOLO(str(best))

        results = model.predict(
            source=source,
            conf=conf,
            iou=0.45,
            save=save,
            save_txt=True,
            save_conf=True,
            project=self.project_dir,
            name='predictions'
        )

        for r in results:
            if len(r.boxes):
                print(f"\n[DÉFAUTS DÉTECTÉS] {r.path}")
                for box in r.boxes:
                    cls_id = int(box.cls)
                    cls_name = DEFECT_CLASSES[cls_id] if cls_id < len(DEFECT_CLASSES) else str(cls_id)
                    conf_score = float(box.conf)
                    print(f"  → {cls_name:<22} | confiance: {conf_score:.2%}")
            else:
                print(f"  ✅ Aucun défaut détecté : {r.path}")

        return results
