"""Tests de la persistance.

Trois niveaux : la logique de stats (pure), l'adaptateur SQL réel (contre un
SQLite en mémoire, rapide), et l'intégration du serveur avec des doublures
injectées (aucune vraie base, aucune clé, aucun réseau).
"""

import anyio
import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from llm_eval_mcp.adapters.persistence import SqlEvalRunRepository
from llm_eval_mcp.domain.eval_run import EvalRun, compute_stats
from llm_eval_mcp.domain.judging import JudgeVerdict
from tests.fakes import FakeJudge, InMemoryEvalRunRepository


def _run(passed=True, score=4) -> EvalRun:
    return EvalRun(
        prompt="p",
        expected_behavior="e",
        answer="a",
        passed=passed,
        score=score,
        justification="j",
        model="test-model",
    )


# --- 1. Statistiques (logique pure) ---------------------------------------

def test_stats_on_empty_list():
    stats = compute_stats([])
    assert stats.total == 0
    assert stats.pass_rate == 0.0
    assert stats.avg_score == 0.0


def test_stats_pass_rate_and_average():
    runs = [_run(passed=True, score=5), _run(passed=False, score=1), _run(passed=True, score=3)]
    stats = compute_stats(runs)
    assert stats.total == 3
    assert stats.passed == 2
    assert stats.pass_rate == round(2 / 3, 3)
    assert stats.avg_score == 3.0


# --- 2. Repository SQL réel (SQLite en mémoire) ---------------------------

@pytest.fixture()
def sql_repo():
    # SQLite en mémoire partagé sur une seule connexion (StaticPool) : rapide,
    # jetable, et surtout un VRAI test de l'adaptateur SQLAlchemy.
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    repo = SqlEvalRunRepository(engine)
    repo.create_schema()
    return repo


def test_sql_save_assigns_id_and_persists(sql_repo):
    saved = sql_repo.save(_run())
    assert saved.id is not None  # la base a attribué un id

    runs = sql_repo.list_runs()
    assert len(runs) == 1
    assert runs[0].prompt == "p"
    assert runs[0].model == "test-model"


def test_sql_roundtrip_preserves_fields(sql_repo):
    sql_repo.save(_run(passed=False, score=2))
    (loaded,) = sql_repo.list_runs()
    assert loaded.passed is False
    assert loaded.score == 2


def test_normalize_db_url_forces_psycopg_driver():
    from llm_eval_mcp.adapters.persistence import _normalize_db_url

    # Les URL fournies par les plateformes doivent viser le driver psycopg3.
    assert _normalize_db_url("postgres://u:p@h:5432/db") == "postgresql+psycopg://u:p@h:5432/db"
    assert _normalize_db_url("postgresql://u:p@h:5432/db") == "postgresql+psycopg://u:p@h:5432/db"
    # SQLite (local) reste inchangé.
    assert _normalize_db_url("sqlite:///./eval_runs.db") == "sqlite:///./eval_runs.db"


# --- 3. Intégration du serveur avec doublures injectées -------------------

def test_judge_tool_persists_and_stats_reflect_it(monkeypatch):
    from llm_eval_mcp import server, wiring

    # On injecte un faux juge et un repo en mémoire dans le point de composition.
    fake_judge = FakeJudge(JudgeVerdict(passed=True, score=5, justification="ok"))
    repo = InMemoryEvalRunRepository()
    monkeypatch.setattr(wiring, "_judge", fake_judge)
    monkeypatch.setattr(wiring, "_repo", repo)

    # Appel du tool de jugement : doit renvoyer un verdict ET l'enregistrer.
    res = anyio.run(
        lambda: server.mcp.call_tool(
            "run_llm_as_judge",
            {"prompt": "p", "expected_behavior": "e", "answer": "a"},
        )
    )
    assert res.is_error is False
    assert len(repo.list_runs()) == 1

    # L'outil de stats doit refléter l'évaluation enregistrée.
    stats_res = anyio.run(lambda: server.mcp.call_tool("get_eval_stats", {}))
    assert stats_res.structured_content["total"] == 1
    assert stats_res.structured_content["passed"] == 1
