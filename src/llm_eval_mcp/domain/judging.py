"""Cœur métier du jugement (LLM-as-judge).

Ce module reste *indépendant du fournisseur* : il définit CE QU'EST un juge
(le contrat `Judge`), le format du verdict, la rubrique du prompt et la fonction
d'orchestration. Il ne sait rien de Groq, d'OpenAI ni du réseau ces détails
vivent dans la couche `adapters/`. On peut donc tout tester sans clé ni réseau.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field


class JudgeError(Exception):
    """Levée quand un juge ne peut pas produire un verdict exploitable."""


class JudgeVerdict(BaseModel):
    """Le résultat structuré d'une évaluation. C'est notre "donnée fiable".

    Sert de contrat des deux côtés : on impose ce schéma au LLM (sortie
    structurée) ET on valide sa réponse avec (voir la couche adapters).
    """

    passed: bool = Field(
        description="Vrai si la réponse respecte l'essentiel du comportement attendu."
    )
    score: int = Field(ge=1, le=5, description="Qualité globale, de 1 (mauvais) à 5 (parfait).")
    justification: str = Field(
        description="Explication courte et factuelle du verdict (1 à 2 phrases)."
    )


class Judge(Protocol):
    """Contrat (port) : tout objet sachant faire ceci est un juge.

    Aucune implémentation ici juste la *forme* attendue. Les adaptateurs
    concrets (GroqJudge en prod, FakeJudge en test) s'y conforment sans hériter.
    """

    def evaluate(self, prompt: str, expected_behavior: str, answer: str) -> JudgeVerdict:
        ...


def build_judge_prompt(
    prompt: str,
    expected_behavior: str,
    answer: str,
) -> tuple[str, str]:
    """Construit la rubrique du juge, indépendamment du fournisseur LLM.

    Renvoie un couple (system, user). L'adaptateur se charge de l'assembler
    au format attendu par son API. On isole ici *comment on juge* (la rubrique)
    de *comment on appelle le modèle* (l'adaptateur).
    """
    system = (
        "Tu es un évaluateur rigoureux et impartial (LLM-as-judge). "
        "On te donne une consigne, le comportement attendu d'un bon agent, "
        "et une réponse à évaluer. Détermine si la réponse respecte l'essentiel "
        "du comportement attendu.\n"
        "Réponds UNIQUEMENT avec un objet JSON valide, sans texte ni balises autour, "
        "exactement de cette forme :\n"
        '{"passed": <bool>, "score": <entier 1-5>, "justification": "<1 à 2 phrases>"}\n'
        "Barème du score : 1 = très mauvais, 3 = acceptable, 5 = parfait. "
        "Mets passed=true seulement si la réponse satisfait l'essentiel du comportement attendu."
    )
    user = (
        f"Consigne soumise à l'agent :\n{prompt}\n\n"
        f"Comportement attendu (référence) :\n{expected_behavior}\n\n"
        f"Réponse de l'agent à évaluer :\n{answer}"
    )
    return system, user


def run_llm_as_judge(
    prompt: str,
    expected_behavior: str,
    answer: str,
    judge: Judge,
) -> JudgeVerdict:
    """Orchestration : délègue le jugement au `judge` fourni de l'extérieur.

    Cette fonction ne choisit PAS le juge et ne connaît aucun fournisseur.
    Elle reçoit un `Judge` (injection de dépendance) et l'utilise.
    """
    return judge.evaluate(prompt, expected_behavior, answer)
