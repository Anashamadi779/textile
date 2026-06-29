"""Envoi automatique du rapport PDF par e-mail via l'API HTTP de Brevo.

On passe par l'API REST (HTTPS, port 443) plutôt que par le SMTP, car beaucoup
de réseaux bloquent les ports SMTP sortants (587/465/2525). Aucune clé n'est
codée en dur : la clé API vient de settings.py / .env.
"""

import base64
from datetime import datetime

import requests
from loguru import logger

BREVO_API_URL = "https://api.brevo.com/v3/smtp/email"
SUBJECT = "Envoi de rapport d'inspection du tissu"


def _build_body() -> str:
    return (
        "Bonjour,\n\n"
        "Veuillez trouver ci-joint le rapport d'inspection qualité du tissu, "
        f"généré automatiquement par QualityAgent le "
        f"{datetime.now():%d/%m/%Y à %H:%M}.\n\n"
        "Ce rapport présente le résultat de l'inspection (type de défaut, "
        "sévérité, score qualité, conformité), la cause probable, l'action "
        "corrective ainsi que l'historique des analyses de la session.\n\n"
        "Cordialement,\n"
        "QualityAgent — ESITH Casablanca"
    )


def send_report_email(
    pdf_bytes: bytes,
    api_key: str,
    mail_from: str,
    mail_to: str,
    sender_name: str = "QualityAgent",
    filename: str = "rapport_qualite.pdf",
) -> tuple[bool, str]:
    """Envoie le PDF en pièce jointe via l'API Brevo. Retourne (succès, message)."""
    if not api_key:
        return False, "Clé API Brevo non configurée dans .env — e-mail non envoyé."
    if not pdf_bytes:
        return False, "Aucun rapport PDF à envoyer."

    payload = {
        "sender": {"email": mail_from, "name": sender_name},
        "to": [{"email": mail_to}],
        "subject": SUBJECT,
        "textContent": _build_body(),
        "attachment": [{
            "content": base64.b64encode(pdf_bytes).decode("ascii"),
            "name": filename,
        }],
    }
    headers = {
        "api-key": api_key,
        "accept": "application/json",
        "content-type": "application/json",
    }

    try:
        resp = requests.post(BREVO_API_URL, json=payload, headers=headers, timeout=30)
        if resp.status_code in (200, 201, 202):
            logger.info(f"Rapport envoyé via l'API Brevo à {mail_to}")
            return True, f"Rapport envoyé automatiquement à {mail_to}."
        logger.error(f"Échec Brevo (HTTP {resp.status_code}): {resp.text[:300]}")
        return False, f"Échec de l'envoi (HTTP {resp.status_code}) : {resp.text[:200]}"
    except Exception as exc:
        logger.error(f"Échec de l'envoi e-mail : {exc}")
        return False, f"Échec de l'envoi de l'e-mail : {exc}"
