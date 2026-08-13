"""Tests de l'adaptateur HTTP (FastAPI), sans réseau.

On utilise le TestClient de FastAPI : il appelle l'app en mémoire, sans ouvrir
de port. Les endpoints coûteux sont testés avec des doublures injectées dans le
point de composition, et l'auth est vérifiée dans ses trois cas.
"""

import pytest
from fastapi.testclient import TestClient

from llm_eval_mcp import wiring
from llm_eval_mcp.api import app
from llm_eval_mcp.domain.judging import JudgeVerdict
from tests.fakes import FakeJudge, InMemoryEvalRunRepository

client = TestClient(app)

API_KEY = "test-secret-key"


@pytest.fixture(autouse=True)
def _wire_fakes(monkeypatch):
    # Clé d'API connue + doublures : aucun appel réseau, aucune vraie base.
    monkeypatch.setenv("API_KEY", API_KEY)
    monkeypatch.setattr(
        wiring, "_judge", FakeJudge(JudgeVerdict(passed=True, score=5, justification="ok"))
    )
    monkeypatch.setattr(wiring, "_repo", InMemoryEvalRunRepository())


def test_health_is_open():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_generate_tasks_without_key():
    # /tasks est gratuit (pas de LLM) : accessible sans clé.
    resp = client.post("/tasks", json={"category": "hallucination", "count": 3})
    assert resp.status_code == 200
    assert len(resp.json()) == 3


def test_judge_without_key_is_rejected():
    resp = client.post(
        "/judge",
        json={"prompt": "p", "expected_behavior": "e", "answer": "a"},
    )
    assert resp.status_code == 401


def test_judge_with_wrong_key_is_rejected():
    resp = client.post(
        "/judge",
        headers={"X-API-Key": "mauvaise-cle"},
        json={"prompt": "p", "expected_behavior": "e", "answer": "a"},
    )
    assert resp.status_code == 401


def test_judge_with_correct_key_succeeds_and_persists():
    resp = client.post(
        "/judge",
        headers={"X-API-Key": API_KEY},
        json={"prompt": "p", "expected_behavior": "e", "answer": "a"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["passed"] is True
    assert body["score"] == 5

    # L'évaluation a bien été enregistrée : /stats la reflète.
    stats = client.get("/stats").json()
    assert stats["total"] == 1
    assert stats["passed"] == 1
