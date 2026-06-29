from dataclasses import dataclass
from typing import Optional

from settings import CONFIDENCE_THRESHOLD, DEFECT_LABELS


@dataclass
class InterpretedResult:
    is_ok:       bool
    label:       str
    confidence:  float
    defect_type: Optional[str] = None  

    def __str__(self) -> str:
        if self.is_ok:
            return "OK"
        if self.defect_type:
            return f"{self.defect_type.replace('_', ' ').title()}  {self.confidence:.0%}"
        return f"Defect Detected  {self.confidence:.0%}"


def interpret(predictions: list) -> InterpretedResult:
    if not predictions:
        return InterpretedResult(is_ok=True, label="ok", confidence=1.0)

    top   = max(predictions, key=lambda p: float(getattr(p, "score", 0.0)))
    label = getattr(top, "label_name", None) or getattr(top, "label", None) or ""
    score = float(getattr(top, "score", 0.0))

    if label in DEFECT_LABELS and score >= CONFIDENCE_THRESHOLD:
        return InterpretedResult(is_ok=False, label=label, confidence=score)

    return InterpretedResult(is_ok=True, label=label, confidence=score)


def decide(
    ll: InterpretedResult,
    image,
    typer,
    ll_valid: bool = True,
    threshold: float = CONFIDENCE_THRESHOLD,
):
    """Orchestration LandingLens + Mistral.

    Règles :
    - Mistral n'est consulté pour le VERDICT (conforme/défaut) que si LandingLens
      est peu sûr (confiance < `threshold`) ou indisponible. Sinon on garde le
      verdict LandingLens sans appeler Mistral.
    - Quand les deux verdicts existent, on retient celui à la meilleure confiance.
    - Le TYPE de défaut vient toujours de Mistral dès que la décision finale est
      « défaut » (même si Mistral n'a pas été consulté pour le verdict).

    Retourne (result, mistral_verdict, mistral_consulted).
    `typer` doit exposer .inspect(image) -> MistralVerdict|None et .classify(image).
    """
    verdict = None
    consulted = False

    # 1) Verdict Mistral seulement si LandingLens est peu sûr / indisponible.
    if (not ll_valid) or ll.confidence < threshold:
        verdict = typer.inspect(image)
        consulted = True

    # 2) Verdict final.
    if not ll_valid:
        if verdict is not None:
            is_ok, confidence = verdict.is_ok, verdict.confidence
        else:
            is_ok, confidence = ll.is_ok, ll.confidence
    elif consulted and verdict is not None:
        if verdict.confidence >= ll.confidence:
            is_ok, confidence = verdict.is_ok, verdict.confidence
        else:
            is_ok, confidence = ll.is_ok, ll.confidence
    else:
        is_ok, confidence = ll.is_ok, ll.confidence

    # 3) Type de défaut via Mistral si la décision finale est « défaut ».
    defect_type = None
    if not is_ok:
        if verdict is not None and verdict.defect_type:
            defect_type = verdict.defect_type
        else:
            defect_type = typer.classify(image)

    result = InterpretedResult(
        is_ok=is_ok,
        label="ok" if is_ok else "Defect",
        confidence=confidence,
        defect_type=defect_type,
    )
    return result, verdict, consulted
