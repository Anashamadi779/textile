"""Quality scoring + PDF report generation for QualityAgent.

Pure helpers (no Streamlit dependency) so they can be reused from app.py, main.py
or tests. Builds a one-click PDF inspection report with reportlab.
"""

import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ── Domain knowledge (zero-shot defect types → cause / corrective action) ─────
DEFECT_FR: dict[str, str] = {"hole": "Trou", "stain": "Tache", "crease": "Pli"}

CAUSE_ACTION: dict[str, tuple[str, str]] = {
    "hole": (
        "Rupture de fil ou tension excessive sur le métier à tisser.",
        "Vérifier la tension des fils de chaîne et l'état des aiguilles / lames.",
    ),
    "stain": (
        "Contamination par huile, poussière ou résidus lors du tissage.",
        "Nettoyer les rouleaux et contrôler la propreté de la ligne de production.",
    ),
    "crease": (
        "Mauvais enroulement ou stockage inadéquat du tissu.",
        "Ajuster la tension d'enroulement et revoir les conditions de stockage.",
    ),
}


def defect_label_fr(defect_type: str | None) -> str:
    if not defect_type:
        return "—"
    return DEFECT_FR.get(defect_type, defect_type.replace("_", " ").title())


def severity_of(result) -> str:
    """Severity derived from the model's confidence on a defect."""
    if result.is_ok:
        return "—"
    c = result.confidence
    if c >= 0.90:
        return "Critique"
    if c >= 0.80:
        return "Majeure"
    return "Mineure"


def quality_score_of(result) -> int:
    """0–100 quality index: high when conforming, low when a confident defect."""
    if result.is_ok:
        return round(result.confidence * 100)
    return round((1 - result.confidence) * 100)


def conformity_of(result) -> str:
    return "Conforme" if result.is_ok else "Non conforme"


def cause_action_of(result) -> tuple[str, str]:
    if result.is_ok:
        return ("Aucune anomalie détectée — tissu conforme.", "Aucune action requise.")
    return CAUSE_ACTION.get(
        result.defect_type,
        ("Cause indéterminée.", "Inspection manuelle recommandée."),
    )


def build_record(result, name: str) -> dict:
    """One history row for the session table."""
    return {
        "time": datetime.now().strftime("%H:%M:%S"),
        "defect_type": defect_label_fr(result.defect_type) if not result.is_ok else "—",
        "severity": severity_of(result),
        "score": quality_score_of(result),
        "conformity": conformity_of(result),
        "ok": result.is_ok,
        "name": name,
    }


# ── PDF ───────────────────────────────────────────────────────────────────────
_NAVY = colors.HexColor("#0f3460")
_STEEL = colors.HexColor("#1e3a5f")
_GREY = colors.HexColor("#64748b")
_LINE = colors.HexColor("#cbd5e1")
_GREEN = colors.HexColor("#059669")
_RED = colors.HexColor("#dc2626")


def _footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica-Oblique", 8)
    canvas.setFillColor(_GREY)
    canvas.drawCentredString(
        A4[0] / 2, 12 * mm,
        "Généré automatiquement par QualityAgent — ESITH Casablanca",
    )
    canvas.restoreState()


def build_pdf(result, image, history: list[dict]) -> bytes:
    """Generate the PDF report for `result`/`image` plus the session `history`."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=18 * mm, bottomMargin=22 * mm,
        leftMargin=18 * mm, rightMargin=18 * mm,
        title="QualityAgent — Rapport de Contrôle Qualité",
    )

    base = getSampleStyleSheet()
    title_style = ParagraphStyle("t", parent=base["Title"], fontSize=18, textColor=_NAVY)
    meta_style = ParagraphStyle("m", parent=base["Normal"], fontSize=10, textColor=_GREY)
    h2 = ParagraphStyle("h2", parent=base["Heading2"], fontSize=12, textColor=_STEEL,
                        spaceBefore=4, spaceAfter=4)
    body = base["BodyText"]

    now = datetime.now()
    story = [
        Paragraph("QualityAgent — Rapport de Contrôle Qualité", title_style),
        Paragraph(f"Date : {now:%d/%m/%Y}&nbsp;&nbsp;&nbsp;Heure : {now:%H:%M:%S}", meta_style),
        Spacer(1, 7 * mm),
    ]

    # — Analyzed image —
    story.append(Paragraph("Image analysée", h2))
    img_buf = io.BytesIO()
    image.convert("RGB").save(img_buf, format="JPEG", quality=85)
    img_buf.seek(0)
    iw, ih = image.size
    max_w, max_h = 95 * mm, 70 * mm
    w, h = max_w, max_w * ih / iw
    if h > max_h:
        w, h = max_h * iw / ih, max_h
    story.append(RLImage(img_buf, width=w, height=h))
    story.append(Spacer(1, 6 * mm))

    # — Result —
    story.append(Paragraph("Résultat de l'inspection", h2))
    dtype = defect_label_fr(result.defect_type) if not result.is_ok else "—"
    res_rows = [
        ["Type de défaut", dtype],
        ["Sévérité", severity_of(result)],
        ["Score qualité", f"{quality_score_of(result)} / 100"],
        ["Conformité", conformity_of(result)],
    ]
    res_table = Table(res_rows, colWidths=[55 * mm, 105 * mm])
    res_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, _LINE),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef2f7")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (1, 3), (1, 3), _GREEN if result.is_ok else _RED),
        ("FONTNAME", (1, 3), (1, 3), "Helvetica-Bold"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(res_table)
    story.append(Spacer(1, 6 * mm))

    # — Cause / action —
    cause, action = cause_action_of(result)
    story.append(Paragraph("Cause probable", h2))
    story.append(Paragraph(cause, body))
    story.append(Paragraph("Action corrective", h2))
    story.append(Paragraph(action, body))
    story.append(Spacer(1, 6 * mm))

    # — Session history —
    story.append(Paragraph("Historique des analyses (session)", h2))
    header = ["Heure", "Type défaut", "Sévérité", "Score", "Conformité"]
    rows = [header]
    for r in history:
        rows.append([
            r["time"], r["defect_type"], r["severity"],
            f'{r["score"]}/100', r["conformity"],
        ])
    hist_table = Table(rows, colWidths=[26 * mm, 40 * mm, 30 * mm, 24 * mm, 40 * mm],
                       repeatRows=1)
    hist_style = [
        ("GRID", (0, 0), (-1, -1), 0.5, _LINE),
        ("BACKGROUND", (0, 0), (-1, 0), _NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8fb")]),
    ]
    for i, r in enumerate(history, start=1):
        hist_style.append(
            ("TEXTCOLOR", (4, i), (4, i), _GREEN if r["ok"] else _RED)
        )
    hist_table.setStyle(TableStyle(hist_style))
    story.append(hist_table)

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    buf.seek(0)
    return buf.getvalue()
