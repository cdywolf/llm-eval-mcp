"""Tests du cœur métier du jugement, sans réseau ni clé API.

On utilise FakeJudge pour vérifier la *plomberie* (construction du prompt,
délégation, format du verdict) de façon déterministe.
"""

import anyio
import pytest
from pydantic import ValidationError

from llm_eval_mcp.domain.judging import (
    JudgeVerdict,
    build_judge_prompt,
    run_llm_as_judge,
)
from tests.fakes import FakeJudge


def _verdict(passed=True, score=4, justification="ok") -> JudgeVerdict:
    return JudgeVerdict(passed=passed, score=score, justification=justification)


def test_build_prompt_contains_all_inputs():
    system, user = build_judge_prompt(
        prompt="Réponds par OUI ou NON.",
        expected_behavior="Répondre uniquement par OUI ou NON.",
        answer="Peut-être.",
    )
    # La rubrique doit exiger du JSON, et le prompt utilisateur porter les 3 entrées.
    assert "JSON" in system
    assert "Réponds par OUI ou NON." in user
    assert "Répondre uniquement par OUI ou NON." in user
    assert "Peut-être." in user


def test_run_llm_as_judge_delegates_to_injected_judge():
    fake = FakeJudge(_verdict(passed=False, score=2))
    result = run_llm_as_judge(
        prompt="p", expected_behavior="e", answer="a", judge=fake
    )
    # Le résultat est bien celui du juge injecté...
    assert result.passed is False
    assert result.score == 2
    # ...et la fonction a bien transmis les 3 arguments au juge.
    assert fake.calls == [("p", "e", "a")]


def test_verdict_rejects_out_of_range_score():
    with pytest.raises(ValidationError):
        JudgeVerdict(passed=True, score=9, justification="hors bornes")


def test_verdict_parses_from_json():
    raw = '{"passed": true, "score": 5, "justification": "parfait"}'
    verdict = JudgeVerdict.model_validate_json(raw)
    assert verdict.passed is True
    assert verdict.score == 5


def test_server_registers_both_tools():
    # Test de contrat : importer le serveur et lister ses tools ne nécessite
    # aucune clé (le juge est construit paresseusement).
    from llm_eval_mcp.server import mcp

    tools = anyio.run(mcp.list_tools)
    names = {t.name for t in tools}
    assert "generate_adversarial_tasks" in names
    assert "run_llm_as_judge" in names
