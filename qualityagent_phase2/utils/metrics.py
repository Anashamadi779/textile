"""
utils/metrics.py
================
Calcul et affichage des métriques d'évaluation pour QualityAgent.
Priorité au RAPPEL : ne jamais manquer un défaut coûte moins cher
que de laisser une pièce défectueuse passer en production.

Métriques calculées :
  - Précision, Rappel, F1 par classe
  - mAP@0.5 et mAP@0.5:0.95 (détection)
  - Matrice de confusion
  - Courbe Précision-Rappel
  - Rapport HTML automatique
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')   # Sans interface graphique (serveur)

from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score
)
from typing import List, Dict, Optional
import json


CLASS_NAMES = [
    "defect",
    "ok"
]

# Seuil de rappel minimum acceptable en production
RECALL_THRESHOLD_MIN = 0.90


# ─── Métriques de classification ────────────────────────────────────────────

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray] = None,
    class_names: List[str] = CLASS_NAMES,
    output_dir: str = "reports"
) -> Dict:
    """
    Calcule et sauvegarde toutes les métriques de classification.

    Args:
        y_true    : Labels réels (int array)
        y_pred    : Prédictions (int array)
        y_scores  : Scores de confiance (float array, shape [N, C]) — optionnel
        class_names : Noms des classes
        output_dir  : Dossier de sauvegarde des rapports

    Returns:
        dict avec précision, rappel, F1 par classe + métriques globales
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── Rapport sklearn ──────────────────────────────────────────────────────
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    # ── Alerte si rappel insuffisant ─────────────────────────────────────────
    print("\n" + "="*60)
    print("  RAPPORT MÉTRIQUES — QualityAgent Phase 2")
    print("="*60)

    alerts = []
    for cls in class_names:
        if cls in report:
            rec = report[cls]['recall']
            prec = report[cls]['precision']
            f1  = report[cls]['f1-score']
            status = "✅" if rec >= RECALL_THRESHOLD_MIN else "⚠️ INSUFFISANT"
            print(f"  {cls:<22} | Rappel: {rec:.2%}  Préc: {prec:.2%}  F1: {f1:.2%}  {status}")
            if rec < RECALL_THRESHOLD_MIN:
                alerts.append(cls)

    macro = report.get('macro avg', {})
    print("-"*60)
    print(f"  Macro avg              | Rappel: {macro.get('recall',0):.2%}  "
          f"Préc: {macro.get('precision',0):.2%}  F1: {macro.get('f1-score',0):.2%}")

    if alerts:
        print(f"\n  ⚠️  Rappel < {RECALL_THRESHOLD_MIN:.0%} pour : {', '.join(alerts)}")
        print("     → Collecter plus d'images ou ajuster le seuil de confiance")
    print("="*60 + "\n")

    # ── Matrice de confusion ─────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    _plot_confusion_matrix(cm, class_names, output_dir)

    # ── Courbe Précision-Rappel (si scores disponibles) ──────────────────────
    if y_scores is not None:
        _plot_pr_curves(y_true, y_scores, class_names, output_dir)

    # ── Sauvegarde JSON ──────────────────────────────────────────────────────
    report_path = Path(output_dir) / "metrics_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  📄 Rapport JSON : {report_path}")

    return report


def _plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_dir: str
) -> None:
    """Génère et sauvegarde la matrice de confusion."""
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#0a0c10')
    ax.set_facecolor('#111318')

    # Normaliser par ligne (taux de rappel par classe)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    im = ax.imshow(cm_norm, cmap='Greens', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val_norm = cm_norm[i, j]
            val_abs  = cm[i, j]
            color = 'white' if val_norm < 0.5 else '#0a0c10'
            ax.text(j, i, f"{val_norm:.0%}\n({val_abs})",
                    ha='center', va='center', fontsize=8,
                    color=color, fontweight='bold' if i == j else 'normal')

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=40, ha='right',
                       color='#9ca3af', fontsize=9)
    ax.set_yticklabels(class_names, color='#9ca3af', fontsize=9)
    ax.set_xlabel("Prédit", color='#e8eaf0', labelpad=10)
    ax.set_ylabel("Réel", color='#e8eaf0', labelpad=10)
    ax.set_title("Matrice de confusion (normalisée par ligne = taux de rappel)",
                 color='#e8eaf0', pad=15)
    ax.tick_params(colors='#6b7280')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1f2937')

    plt.tight_layout()
    path = Path(output_dir) / "confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  📊 Matrice de confusion : {path}")


def _plot_pr_curves(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    class_names: List[str],
    output_dir: str
) -> None:
    """Génère les courbes Précision-Rappel par classe."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0a0c10')
    ax.set_facecolor('#111318')

    colors = ['#00e5a0', '#6c63ff', '#ff9f43', '#ff6b6b',
              '#54a0ff', '#48dbfb', '#ff9ff3']

    for i, cls_name in enumerate(class_names):
        # One-vs-rest pour chaque classe
        y_bin = (y_true == i).astype(int)
        if y_bin.sum() == 0:
            continue
        scores = y_scores[:, i]
        prec, rec, _ = precision_recall_curve(y_bin, scores)
        ap = average_precision_score(y_bin, scores)
        ax.plot(rec, prec, color=colors[i % len(colors)],
                linewidth=2, label=f"{cls_name} (AP={ap:.2f})")

    # Ligne de rappel minimum
    ax.axvline(x=RECALL_THRESHOLD_MIN, color='#ff6b6b', linestyle='--',
               linewidth=1.5, alpha=0.7, label=f"Rappel min ({RECALL_THRESHOLD_MIN:.0%})")
    ax.axhspan(0, 1, xmin=0, xmax=RECALL_THRESHOLD_MIN,
               alpha=0.05, color='#ff6b6b')

    ax.set_xlabel("Rappel", color='#e8eaf0', labelpad=8)
    ax.set_ylabel("Précision", color='#e8eaf0', labelpad=8)
    ax.set_title("Courbes Précision-Rappel par classe", color='#e8eaf0', pad=12)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.legend(loc='lower left', fontsize=8,
              facecolor='#181c24', edgecolor='#374151', labelcolor='#9ca3af')
    ax.grid(True, color='#1f2937', linewidth=0.5)
    ax.tick_params(colors='#6b7280')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1f2937')

    plt.tight_layout()
    path = Path(output_dir) / "pr_curves.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  📈 Courbes PR : {path}")


# ─── Métriques de détection (YOLO) ──────────────────────────────────────────

def summarize_yolo_results(results_path: str) -> Dict:
    """
    Lit les résultats d'un run YOLOv8 (results.csv) et affiche un résumé.

    Args:
        results_path : Chemin vers results.csv du run YOLO

    Returns:
        dict avec les meilleures métriques
    """
    import pandas as pd

    df = pd.read_csv(results_path, skipinitialspace=True)
    df.columns = df.columns.str.strip()

    best_epoch = df['metrics/mAP50(B)'].idxmax()
    best = df.loc[best_epoch]

    print("\n" + "="*60)
    print("  RÉSUMÉ RUN YOLOV8 — Meilleure epoch")
    print("="*60)
    print(f"  Epoch           : {int(best.get('epoch', best_epoch))}")
    print(f"  mAP@0.5         : {best.get('metrics/mAP50(B)', 0):.4f}")
    print(f"  mAP@0.5:0.95    : {best.get('metrics/mAP50-95(B)', 0):.4f}")
    print(f"  Précision (B)   : {best.get('metrics/precision(B)', 0):.4f}")
    print(f"  Rappel (B)      : {best.get('metrics/recall(B)', 0):.4f}")
    print(f"  Loss box        : {best.get('val/box_loss', 0):.4f}")
    print(f"  Loss cls        : {best.get('val/cls_loss', 0):.4f}")
    rec = best.get('metrics/recall(B)', 0)
    status = "✅ OK" if rec >= RECALL_THRESHOLD_MIN else f"⚠️  Rappel < {RECALL_THRESHOLD_MIN:.0%}"
    print(f"\n  Statut rappel   : {status}")
    print("="*60 + "\n")

    return best.to_dict()
