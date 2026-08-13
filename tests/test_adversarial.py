"""Tests unitaires de la logique métier pure.

On teste la génération de tâches SANS démarrer de serveur MCP : c'est tout
l'intérêt d'avoir séparé le domaine du câblage.
"""

import pytest

from llm_eval_mcp.domain.adversarial import (
    AdversarialTask,
    FailureCategory,
    generate_adversarial_tasks,
)


def test_generates_requested_count():
    tasks = generate_adversarial_tasks(FailureCategory.HALLUCINATION, count=5)
    assert len(tasks) == 5
    assert all(isinstance(t, AdversarialTask) for t in tasks)


def test_all_tasks_have_target_category():
    tasks = generate_adversarial_tasks(FailureCategory.PROMPT_INJECTION, count=4)
    assert all(t.category is FailureCategory.PROMPT_INJECTION for t in tasks)


def test_ids_are_unique_and_stable():
    tasks = generate_adversarial_tasks(FailureCategory.REASONING_ERROR, count=3)
    ids = [t.id for t in tasks]
    assert len(set(ids)) == len(ids)  # unicité


def test_reproducible_with_same_seed():
    a = generate_adversarial_tasks(FailureCategory.TOOL_MISUSE, count=5, seed=7)
    b = generate_adversarial_tasks(FailureCategory.TOOL_MISUSE, count=5, seed=7)
    assert [t.prompt for t in a] == [t.prompt for t in b]


def test_difficulty_is_within_bounds():
    tasks = generate_adversarial_tasks(FailureCategory.UNSAFE_OUTPUT, count=6)
    assert all(1 <= t.difficulty <= 5 for t in tasks)


@pytest.mark.parametrize("bad_count", [0, -1, 51, 100])
def test_rejects_invalid_count(bad_count):
    with pytest.raises(ValueError):
        generate_adversarial_tasks(FailureCategory.HALLUCINATION, count=bad_count)
