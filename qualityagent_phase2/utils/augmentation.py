"""
utils/augmentation.py
=====================
Pipeline d'augmentation adapté aux textures textiles.
Simule les conditions réelles d'un atelier de confection :
  - Variations d'éclairage (LED, naturel, mixte)
  - Légères variations de teinte tissu (teinture lot-à-lot)
  - Flou de mouvement (caméra ligne de production)
  - Bruit de capteur industriel
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ─── Moyennes/std ImageNet (backbone pré-entraîné) ─────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transform(image_size: int = 640) -> A.Compose:
    """
    Augmentations pour l'entraînement.
    Compatible annotations YOLO (bounding boxes format 'yolo').
    """
    return A.Compose([
        # ── Géométrie ───────────────────────────────────────────────────────
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.15,
            rotate_limit=10,
            border_mode=0,
            p=0.5
        ),

        # ── Éclairage atelier (LED / naturel / mixte) ───────────────────────
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.7
        ),
        A.RandomGamma(gamma_limit=(80, 120), p=0.4),
        A.CLAHE(clip_limit=2.0, p=0.3),         # Améliore contraste local

        # ── Variation de teinte tissu (teinture lot-à-lot) ──────────────────
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=10,
            p=0.5
        ),

        # ── Dégradations optiques ───────────────────────────────────────────
        A.OneOf([
            A.MotionBlur(blur_limit=5),          # Flou mouvement ligne
            A.GaussianBlur(blur_limit=3),        # Flou mise au point
            A.MedianBlur(blur_limit=3),          # Bruit impulsionnel
        ], p=0.3),

        # ── Bruit capteur industriel ─────────────────────────────────────────
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=0.2),

        # ── Occultation partielle (ombre, reflet) ────────────────────────────
        A.CoarseDropout(
            max_holes=4,
            max_height=32,
            max_width=32,
            min_holes=1,
            fill_value=0,
            p=0.2
        ),

        # ── Resize & normalisation ───────────────────────────────────────────
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=0
        ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()

    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.3   # Supprime les boxes trop petites après crop
    ))


def get_val_transform(image_size: int = 640) -> A.Compose:
    """
    Transformations pour validation/test.
    Pas d'augmentation, uniquement resize + normalisation.
    """
    return A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=0
        ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()

    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels']
    ))


def get_classification_transform(image_size: int = 224) -> dict:
    """
    Transformations pour DINOv2 / EfficientNet (classification).
    Retourne un dict {'train': ..., 'val': ...}.
    """
    train_tf = A.Compose([
        A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.7, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(0.3, 0.3, p=0.7),
        A.HueSaturationValue(10, 20, 10, p=0.5),
        A.GaussNoise(var_limit=(5, 30), p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])

    val_tf = A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])

    return {'train': train_tf, 'val': val_tf}
