"""Doublures de test (test doubles).

FakeJudge se conforme au contrat `Judge` sans en hériter : il a juste la bonne
méthode `evaluate`. Il ne fait aucun appel réseau et renvoie un verdict fixé
d'avance, ce qui rend les tests déterministes, rapides et exécutables en CI
sans clé API.
"""

from __future__ import annotations

from llm_eval_mcp.domain.eval_run import EvalRun
from llm_eval_mcp.domain.judging import JudgeVerdict


class FakeJudge:
    """Juge factice : renvoie toujours le verdict qu'on lui a fourni."""

    def __init__(self, verdict: JudgeVerdict) -> None:
        self._verdict = verdict
        self.calls: list[tuple[str, str, str]] = []  # trace des appels reçus

    def evaluate(self, prompt: str, expected_behavior: str, answer: str) -> JudgeVerdict:
        self.calls.append((prompt, expected_behavior, answer))
        return self._verdict


class InMemoryEvalRunRepository:
    """Repository factice : stocke les évaluations dans une simple liste.

    Se conforme au port `EvalRunRepository` sans base de données : les tests
    tournent donc sans SQLite ni PostgreSQL, de façon déterministe.
    """

    def __init__(self) -> None:
        self._runs: list[EvalRun] = []
        self._next_id = 1

    def save(self, run: EvalRun) -> EvalRun:
        stored = run.model_copy(update={"id": self._next_id})
        self._next_id += 1
        self._runs.append(stored)
        return stored

    def list_runs(self) -> list[EvalRun]:
        return list(self._runs)
