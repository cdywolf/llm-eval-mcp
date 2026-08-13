"""Adaptateur d'entrée HTTP (FastAPI).

Une seconde porte vers la MÊME logique métier, exposée cette fois par le réseau.
Ne contient aucune logique métier : chaque endpoint traduit une requête HTTP en
appel au point de composition partagé (`wiring`). L'endpoint coûteux (`/judge`,
qui appelle Groq) est protégé par une clé d'API.
"""

from __future__ import annotations

import os
import secrets

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from llm_eval_mcp import wiring
from llm_eval_mcp.domain.adversarial import (
    AdversarialTask,
    FailureCategory,
    generate_adversarial_tasks,
)
from llm_eval_mcp.domain.eval_run import EvalStats
from llm_eval_mcp.domain.judging import JudgeVerdict

# --- Modèles de requête -----------------------------------------------------

class TaskRequest(BaseModel):
    category: FailureCategory
    count: int = Field(default=5, ge=1, le=50)


class JudgeRequest(BaseModel):
    prompt: str
    expected_behavior: str
    answer: str


# --- Authentification par clé d'API -----------------------------------------

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(provided: str | None = Depends(_api_key_header)) -> None:
    """Barrière : rejette la requête (401) si l'en-tête X-API-Key est absent ou faux.

    On compare avec `secrets.compare_digest` (temps constant) pour ne pas
    laisser fuiter d'information par le temps de comparaison.
    """
    expected = os.environ.get("API_KEY")
    if not expected or provided is None or not secrets.compare_digest(provided, expected):
        raise HTTPException(
            status_code=401,
            detail="Clé d'API invalide ou manquante (en-tête X-API-Key requis).",
        )


# --- Application ------------------------------------------------------------

app = FastAPI(
    title="LLM Eval MCP — API",
    version="0.1.0",
    description=(
        "Façade HTTP d'une plateforme d'évaluation de fiabilité des LLM : "
        "génération de tâches adversariales, LLM-as-judge et statistiques."
    ),
)


@app.get("/health", tags=["système"])
def health() -> dict[str, str]:
    """Sonde de disponibilité (utilisée par la plateforme de déploiement)."""
    return {"status": "ok"}


@app.post("/tasks", response_model=list[AdversarialTask], tags=["évaluation"])
def create_tasks(req: TaskRequest) -> list[AdversarialTask]:
    """Génère des tâches adversariales (gratuit, sans LLM : accessible sans clé)."""
    return generate_adversarial_tasks(category=req.category, count=req.count)


@app.get("/stats", response_model=EvalStats, tags=["évaluation"])
def stats() -> EvalStats:
    """Statistiques agrégées de toutes les évaluations enregistrées."""
    return wiring.get_stats()


@app.post(
    "/judge",
    response_model=JudgeVerdict,
    dependencies=[Depends(require_api_key)],
    tags=["évaluation"],
)
def judge(req: JudgeRequest) -> JudgeVerdict:
    """Évalue une réponse via le LLM-as-judge et l'enregistre. Nécessite une clé d'API."""
    return wiring.judge_and_record(req.prompt, req.expected_behavior, req.answer)
