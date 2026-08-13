"""Point de composition (composition root) et services applicatifs.

Ce module est le SEUL endroit qui décide QUELLES implémentations concrètes
sont utilisées en production (GroqJudge, SqlEvalRunRepository) et qui les
assemble. Les deux adaptateurs d'entrée  `server.py` (MCP) et `api.py` (HTTP) 
importent ces fonctions et n'en dupliquent pas la logique.

Les dépendances sont construites paresseusement : importer ce module ne
nécessite ni clé API ni base de données. Seuls les appels effectifs en ont besoin.
"""

from __future__ import annotations

import os

from llm_eval_mcp.domain.eval_run import (
    EvalRun,
    EvalRunRepository,
    EvalStats,
    compute_stats,
)
from llm_eval_mcp.domain.judging import Judge, JudgeVerdict, run_llm_as_judge

# Modèle-juge (surchargable par variable d'environnement).
_JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "llama-3.3-70b-versatile")

_judge: Judge | None = None
_repo: EvalRunRepository | None = None


def get_judge() -> Judge:
    """Renvoie le juge de production (GroqJudge), construit à la première demande."""
    global _judge
    if _judge is None:
        from llm_eval_mcp.adapters.groq_judge import GroqJudge

        _judge = GroqJudge.from_env(model=_JUDGE_MODEL)
    return _judge


def get_repository() -> EvalRunRepository:
    """Renvoie le repository de production (SQL), construit à la première demande."""
    global _repo
    if _repo is None:
        from llm_eval_mcp.adapters.persistence import SqlEvalRunRepository

        _repo = SqlEvalRunRepository.from_env()
    return _repo


# --- Services applicatifs (partagés par les adaptateurs MCP et HTTP) ---------

def judge_and_record(prompt: str, expected_behavior: str, answer: str) -> JudgeVerdict:
    """Juge une réponse PUIS enregistre l'évaluation. Logique partagée."""
    verdict = run_llm_as_judge(prompt, expected_behavior, answer, judge=get_judge())
    get_repository().save(
        EvalRun(
            prompt=prompt,
            expected_behavior=expected_behavior,
            answer=answer,
            passed=verdict.passed,
            score=verdict.score,
            justification=verdict.justification,
            model=_JUDGE_MODEL,
        )
    )
    return verdict


def get_stats() -> EvalStats:
    """Renvoie les statistiques agrégées de toutes les évaluations enregistrées."""
    return compute_stats(get_repository().list_runs())
