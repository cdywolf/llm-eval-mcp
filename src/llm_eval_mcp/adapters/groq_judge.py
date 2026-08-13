"""Adaptateur Groq : l'implémentation *réelle* du juge.

C'est le seul fichier qui connaît Groq, le réseau et la clé API. Il se conforme
au contrat `Judge` du domaine sans en hériter (duck typing structurel). Pour en
changer (Gemini, OpenAI…), on écrit un autre adaptateur respectant le même
contrat, sans toucher au domaine.
"""

from __future__ import annotations

import os

from pydantic import ValidationError

from llm_eval_mcp.domain.judging import JudgeError, JudgeVerdict, build_judge_prompt

# Modèle par défaut : un modèle "fort" pour la qualité de jugement.
DEFAULT_JUDGE_MODEL = "llama-3.3-70b-versatile"


class GroqJudge:
    """Juge qui délègue l'évaluation à un modèle hébergé sur Groq."""

    def __init__(self, client, model: str = DEFAULT_JUDGE_MODEL) -> None:
        # `client` est injecté (un groq.Groq). On ne le crée pas ici pour rester
        # testable et ne pas coder le SDK en dur dans la logique.
        self._client = client
        self._model = model

    @classmethod
    def from_env(cls, model: str = DEFAULT_JUDGE_MODEL) -> GroqJudge:
        """Construit un GroqJudge en lisant GROQ_API_KEY (via .env si présent)."""
        from dotenv import load_dotenv

        load_dotenv()  # charge un éventuel fichier .env dans les variables d'env
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise JudgeError(
                "GROQ_API_KEY manquante. Crée un fichier .env à la racine avec "
                "GROQ_API_KEY=... (clé gratuite sur https://console.groq.com)."
            )
        from groq import Groq

        return cls(Groq(api_key=api_key), model=model)

    def evaluate(self, prompt: str, expected_behavior: str, answer: str) -> JudgeVerdict:
        """Appelle Groq, contraint la sortie en JSON, puis valide avec Pydantic."""
        system, user = build_judge_prompt(prompt, expected_behavior, answer)

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},  # force du JSON valide
            temperature=0,  # jugement le plus stable possible
        )
        content = response.choices[0].message.content or ""

        try:
            # Ceinture de sécurité : on ne fait pas confiance au LLM, on valide.
            return JudgeVerdict.model_validate_json(content)
        except ValidationError as exc:
            raise JudgeError(
                f"Le juge a renvoyé un JSON non conforme au schéma attendu : {content!r}"
            ) from exc
