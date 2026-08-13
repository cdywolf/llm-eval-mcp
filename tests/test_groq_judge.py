"""Tests de l'adaptateur Groq, SANS réseau.

On injecte un faux client Groq qui imite `client.chat.completions.create(...)`.
On teste ainsi la logique de l'adaptateur (envoi des bons paramètres, parsing
et validation du verdict) sans jamais appeler la vraie API.
"""

from types import SimpleNamespace

import pytest

from llm_eval_mcp.adapters.groq_judge import GroqJudge
from llm_eval_mcp.domain.judging import JudgeError, JudgeVerdict


class FakeGroqClient:
    """Imite juste ce que GroqJudge utilise : chat.completions.create(...)."""

    def __init__(self, content: str) -> None:
        self._content = content
        self.last_kwargs: dict | None = None
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        message = SimpleNamespace(content=self._content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_evaluate_parses_valid_verdict():
    client = FakeGroqClient('{"passed": true, "score": 5, "justification": "impeccable"}')
    judge = GroqJudge(client, model="test-model")

    verdict = judge.evaluate(prompt="p", expected_behavior="e", answer="a")

    assert isinstance(verdict, JudgeVerdict)
    assert verdict.passed is True
    assert verdict.score == 5


def test_evaluate_requests_json_and_zero_temperature():
    client = FakeGroqClient('{"passed": false, "score": 1, "justification": "non"}')
    judge = GroqJudge(client, model="test-model")

    judge.evaluate(prompt="p", expected_behavior="e", answer="a")

    # On force bien la sortie JSON et un jugement stable.
    assert client.last_kwargs["response_format"] == {"type": "json_object"}
    assert client.last_kwargs["temperature"] == 0
    assert client.last_kwargs["model"] == "test-model"


def test_evaluate_raises_on_malformed_json():
    client = FakeGroqClient("désolé, voici mon avis en texte libre…")
    judge = GroqJudge(client, model="test-model")

    with pytest.raises(JudgeError):
        judge.evaluate(prompt="p", expected_behavior="e", answer="a")
