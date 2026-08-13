"""Cœur métier de la persistance des évaluations.

Comme pour le jugement, ce module reste PUR : il définit l'entité `EvalRun`,
le contrat `EvalRunRepository` (le port) et le calcul de statistiques. Il ne
connaît ni SQLAlchemy, ni SQLite, ni PostgreSQL ces détails vivent dans
`adapters/persistence.py`. On peut donc tout tester sans base de données.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol

from pydantic import BaseModel, Field


class EvalRun(BaseModel):
    """Une évaluation enregistrée : la consigne, la réponse, et le verdict rendu."""

    id: int | None = Field(default=None, description="Identifiant attribué par la base.")
    prompt: str
    expected_behavior: str
    answer: str
    passed: bool
    score: int = Field(ge=1, le=5)
    justification: str
    model: str = Field(description="Modèle-juge utilisé pour cette évaluation.")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Horodatage de l'évaluation (UTC).",
    )


class EvalStats(BaseModel):
    """Statistiques agrégées sur un ensemble d'évaluations."""

    total: int = Field(description="Nombre total d'évaluations.")
    passed: int = Field(description="Nombre d'évaluations réussies.")
    pass_rate: float = Field(description="Taux de réussite, entre 0 et 1.")
    avg_score: float = Field(description="Score moyen (1 à 5).")


class EvalRunRepository(Protocol):
    """Contrat (port) : tout objet sachant enregistrer et lister des EvalRun."""

    def save(self, run: EvalRun) -> EvalRun:
        ...

    def list_runs(self) -> list[EvalRun]:
        ...


def compute_stats(runs: list[EvalRun]) -> EvalStats:
    """Calcule des statistiques à partir d'une liste d'évaluations (logique pure).

    On garde ce calcul dans le domaine plutôt qu'en SQL : il est ainsi testable
    sans base, et indépendant du fournisseur.
    """
    total = len(runs)
    if total == 0:
        return EvalStats(total=0, passed=0, pass_rate=0.0, avg_score=0.0)

    passed = sum(1 for r in runs if r.passed)
    avg_score = sum(r.score for r in runs) / total
    return EvalStats(
        total=total,
        passed=passed,
        pass_rate=round(passed / total, 3),
        avg_score=round(avg_score, 3),
    )
