"""
agents/quality_agent.py
=======================
Agent LangGraph — contrôle qualité textile.

Graphe : vision → decision → report

CLI :
    python -m agents.quality_agent <image_path> <id_piece>
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import TypedDict

import requests
from langgraph.graph import END, StateGraph

# ─── Constantes ───────────────────────────────────────────────────────────────

API_URL = "http://127.0.0.1:8000/analyze"
QUALITY_THRESHOLD = 0.7  # score qualité > seuil → conforme


# ─── État partagé ─────────────────────────────────────────────────────────────


class AgentState(TypedDict):
    image_path: str
    id_piece: str
    api_response: dict
    score_qualite: float
    decision: str
    rapport: dict


# ─── Nœuds ────────────────────────────────────────────────────────────────────


def node_vision(state: AgentState) -> dict:
    """Envoie l'image à POST /analyze et stocke la réponse brute."""
    image_path = Path(state["image_path"])
    if not image_path.exists():
        raise FileNotFoundError(f"Image introuvable : {image_path}")

    with image_path.open("rb") as f:
        response = requests.post(
            API_URL,
            files={"file": (image_path.name, f, "image/jpeg")},
            timeout=30,
        )
    response.raise_for_status()
    return {"api_response": response.json()}


def node_decision(state: AgentState) -> dict:
    """
    Calcule le score qualité et tranche la conformité.

    score_qualite = 1.0 si aucun défaut détecté,
                    sinon 1 - max(confiances des défauts).
    confiance > 0.7 → conforme, sinon non-conforme.
    """
    defauts = state["api_response"].get("defauts", [])

    if not defauts:
        score = 1.0
    else:
        max_defect_conf = max(d["confiance"] for d in defauts)
        score = round(max(0.0, 1.0 - max_defect_conf), 4)

    decision = "conforme" if score > QUALITY_THRESHOLD else "non-conforme"
    return {"score_qualite": score, "decision": decision}


def node_report(state: AgentState) -> dict:
    """Génère le rapport qualité final sous forme de dict JSON."""
    api_resp = state["api_response"]
    defauts_raw = api_resp.get("defauts", [])

    rapport = {
        "id_piece": state["id_piece"],
        "date": datetime.now().isoformat(timespec="seconds"),
        "image": api_resp.get("image"),
        "nb_defauts": len(defauts_raw),
        "defauts": [
            {
                "classe": d["classe"],
                "confiance": d["confiance"],
                "bbox": d["bbox"],
            }
            for d in defauts_raw
        ],
        "score_qualite": state["score_qualite"],
        "decision": state["decision"],
    }
    return {"rapport": rapport}


# ─── Graphe LangGraph ─────────────────────────────────────────────────────────


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("vision", node_vision)
    graph.add_node("decision", node_decision)
    graph.add_node("report", node_report)

    graph.set_entry_point("vision")
    graph.add_edge("vision", "decision")
    graph.add_edge("decision", "report")
    graph.add_edge("report", END)

    return graph.compile()


# ─── Interface CLI ────────────────────────────────────────────────────────────


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage : python -m agents.quality_agent <image> <id_piece>")
        sys.exit(1)

    image_path, id_piece = sys.argv[1], sys.argv[2]

    agent = build_graph()
    initial_state: AgentState = {
        "image_path": image_path,
        "id_piece": id_piece,
        "api_response": {},
        "score_qualite": 0.0,
        "decision": "",
        "rapport": {},
    }

    result = agent.invoke(initial_state)
    print(json.dumps(result["rapport"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
