"""
models/dino_classifier.py
=========================
Fine-tuning DINOv2 pour la classification de défauts fins en confection.

Quand utiliser DINOv2 plutôt que YOLOv8 ?
  - Défauts subtils difficiles à localiser (fil apparent, pli)
  - Peu de données annotées (DINOv2 = few-shot capable)
  - Besoin de représentations visuelles riches pour similarité

Architecture :
  - Backbone : facebook/dinov2-base (ViT-B/14, 86M params)
  - Tête : Linear(768 → 256 → nc)
  - Loss : Focal Loss (classes très déséquilibrées en production)
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel
from PIL import Image


# ─── Classes de défauts ──────────────────────────────────────────────────────
DEFECT_CLASSES = [
    "defect",
    "ok"
]
NUM_CLASSES = len(DEFECT_CLASSES)


# ─── Dataset ─────────────────────────────────────────────────────────────────

class DefectDataset(Dataset):

    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        self.root = Path(root_dir) / split
        self.transform = transform
        self.samples = []

        for cls_idx, cls_name in enumerate(DEFECT_CLASSES):
            cls_dir = self.root / cls_name
            if not cls_dir.exists():
                print(f"  [Dataset] Dossier manquant : {cls_dir}")
                continue
            images = list(cls_dir.glob('*.jpg')) + \
                     list(cls_dir.glob('*.jpeg')) + \
                     list(cls_dir.glob('*.png'))
            self.samples.extend([(str(img), cls_idx) for img in images])

        print(f"[DefectDataset] {split} : {len(self.samples)} images chargées")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = np.array(Image.open(img_path).convert('RGB'))

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label


# ─── Modèle ──────────────────────────────────────────────────────────────────

class DINOQualityClassifier(nn.Module):
    """
    Classificateur de défauts basé sur DINOv2.

    Le backbone est gelé par défaut (feature extraction).
    Pour un fine-tuning complet, appeler unfreeze_backbone().
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.3,
        freeze_backbone: bool = True
    ):
        super().__init__()

        print("[DINOv2] Chargement du backbone facebook/dinov2-base ...")
        self.backbone = AutoModel.from_pretrained("facebook/dinov2-base")

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("[DINOv2] Backbone gelé — seule la tête est entraînée")

        # Tête de classification
        hidden_dim = self.backbone.config.hidden_size  # 768 pour ViT-B
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=pixel_values)
        # Token [CLS] = représentation globale de l'image
        cls_token = outputs.last_hidden_state[:, 0]
        return self.classifier(cls_token)

    def unfreeze_backbone(self, last_n_layers: int = 4) -> None:
        """
        Dégèle les N derniers blocs du backbone pour un fine-tuning complet.

        Args:
            last_n_layers : Nombre de blocs à dégeler (depuis la fin)
        """
        total_layers = len(self.backbone.encoder.layer)
        for i, layer in enumerate(self.backbone.encoder.layer):
            if i >= total_layers - last_n_layers:
                for param in layer.parameters():
                    param.requires_grad = True
        print(f"[DINOv2] {last_n_layers} derniers blocs dégelés pour fine-tuning")

    def get_embeddings(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Retourne les embeddings [CLS] (utile pour similarité / anomalie).
        """
        with torch.no_grad():
            outputs = self.backbone(pixel_values=pixel_values)
        return outputs.last_hidden_state[:, 0]


# ─── Focal Loss ──────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss pour classes très déséquilibrées.
    Pénalise davantage les exemples faciles → force le modèle sur les défauts rares.

    Référence : Lin et al., "Focal Loss for Dense Object Detection", 2017.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            gamma : Facteur de mise au point (2.0 recommandé)
            alpha : Pondération des classes positives
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal.mean()


# ─── Trainer ─────────────────────────────────────────────────────────────────

class DINOTrainer:
    """
    Gestion complète de l'entraînement DINOv2.

    Exemple d'utilisation :
        trainer = DINOTrainer(data_dir='data/confection_cls', device='cuda')
        trainer.train(epochs=30)
        trainer.save('models/dino_quality.pth')
        trainer.predict('image.jpg')
    """

    def __init__(
        self,
        data_dir: str = 'C:\\Users\\anash\\Downloads\\qualityagent_phase2\\qualityagent_phase2\\data\\Fabric Defect Detection',
        device: str = 'auto',
        batch_size: int = 32,
        num_workers: int = 4
    ):
        self.data_dir   = data_dir
        self.batch_size = batch_size

        # Device auto-detection
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        print(f"[DINOTrainer] Device : {self.device}")

        # Processor HuggingFace (normalisation pour DINOv2)
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

        # Modèle
        self.model = DINOQualityClassifier().to(self.device)

        # DataLoaders (avec transform personnalisé)
        from utils.augmentation import get_classification_transform
        transforms = get_classification_transform(image_size=224)

        train_ds = DefectDataset(data_dir, split='train', transform=transforms['train'])
        val_ds   = DefectDataset(data_dir, split='val',   transform=transforms['val'])

        self.train_loader = DataLoader(
            train_ds, batch_size=batch_size,
            shuffle=True, num_workers=num_workers,
            pin_memory=self.device.type == 'cuda'
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=batch_size,
            shuffle=False, num_workers=num_workers,
            pin_memory=self.device.type == 'cuda'
        )

        self.history = {'train_loss': [], 'val_loss': [], 'val_recall': []}

    def train(
        self,
        epochs: int = 30,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        unfreeze_after: int = 10    # Epoch à partir de laquelle dégeler le backbone
    ) -> None:
        """
        Entraîne le classificateur DINOv2.

        Stratégie en 2 phases :
          Phase 1 (epochs 0→unfreeze_after) : Backbone gelé, seule la tête apprend
          Phase 2 (epochs unfreeze_after→end) : Fine-tuning partiel du backbone

        Args:
            epochs        : Nombre total d'epochs
            lr            : Learning rate
            weight_decay  : Régularisation L2
            unfreeze_after: Epoch pour dégeler le backbone
        """
        criterion = FocalLoss(gamma=2.0, alpha=0.25)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01
        )

        best_recall = 0.0

        for epoch in range(epochs):
            # Phase 2 : dégeler backbone
            if epoch == unfreeze_after:
                self.model.unfreeze_backbone(last_n_layers=4)
                # Réinitialiser optimizer avec lr plus bas
                optimizer = torch.optim.AdamW([
                    {'params': self.model.backbone.parameters(), 'lr': lr * 0.1},
                    {'params': self.model.classifier.parameters(), 'lr': lr}
                ], weight_decay=weight_decay)

            # ── Entraînement ─────────────────────────────────────────────────
            self.model.train()
            train_loss = 0.0
            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                logits = self.model(images)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(self.train_loader)

            # ── Validation ───────────────────────────────────────────────────
            val_loss, val_recall = self._evaluate(criterion)
            scheduler.step()

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_recall'].append(val_recall)

            status = "✅" if val_recall >= 0.90 else "⚠️"
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss train: {train_loss:.4f} | "
                  f"Loss val: {val_loss:.4f} | "
                  f"Rappel: {val_recall:.2%} {status}")

            # Sauvegarde du meilleur modèle (critère = rappel)
            if val_recall > best_recall:
                best_recall = val_recall
                self.save('checkpoints/dino_best.pth')

        print(f"\n[DINOTrainer] Entraînement terminé — meilleur rappel : {best_recall:.2%}")

    def _evaluate(self, criterion) -> Tuple[float, float]:
        """Évalue sur le jeu de validation. Retourne (val_loss, macro_recall)."""
        self.model.eval()
        all_preds, all_labels = [], []
        total_loss = 0.0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(images)
                loss = criterion(logits, labels)
                total_loss += loss.item()
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        from sklearn.metrics import recall_score
        macro_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        return total_loss / len(self.val_loader), macro_recall

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'class_names': DEFECT_CLASSES,
            'history': self.history
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        print(f"[DINOTrainer] Poids chargés depuis {path}")

    def predict(self, image_path: str, conf_threshold: float = 0.3) -> dict:
        """
        Prédit le type de défaut sur une image.

        Args:
            image_path     : Chemin vers l'image
            conf_threshold : Seuil de confiance minimum

        Returns:
            dict avec classe prédite et score de confiance
        """
        image = np.array(Image.open(image_path).convert('RGB'))
        from utils.augmentation import get_classification_transform
        tf = get_classification_transform()['val']
        tensor = tf(image=image)['image'].unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor)
            probs  = torch.softmax(logits, dim=1)[0]

        max_prob, pred_idx = probs.max(dim=0)
        pred_class = DEFECT_CLASSES[pred_idx.item()]
        confidence = max_prob.item()

        if confidence < conf_threshold:
            print(f"  ✅ Aucun défaut détecté (conf max: {confidence:.2%} < seuil {conf_threshold:.2%})")
            return {'class': 'ok', 'confidence': confidence, 'all_probs': probs.cpu().numpy()}

        print(f"  ⚠️  Défaut détecté : {pred_class} ({confidence:.2%})")
        return {'class': pred_class, 'confidence': confidence, 'all_probs': probs.cpu().numpy()}
