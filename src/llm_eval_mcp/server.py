"""Serveur MCP exposant les outils d'évaluation de fiabilité des LLM.

Adaptateur d'entrée MCP : il publie la logique métier comme des tools MCP, en
s'appuyant sur le point de composition partagé (`wiring`). Fine couche de câblage.
"""

from __future__ import annotations

from mcp.server import MCPServer

from llm_eval_mcp import wiring
from llm_eval_mcp.domain.adversarial import (
    AdversarialTask,
    FailureCategory,
)
from llm_eval_mcp.domain.adversarial import (
    generate_adversarial_tasks as _generate,
)
from llm_eval_mcp.domain.eval_run import EvalStats
from llm_eval_mcp.domain.judging import JudgeVerdict

# L'objet serveur. `name` et `version` sont annoncés au client pendant la
# négociation initiale ; `instructions` aide le host à savoir à quoi sert ce serveur.
mcp = MCPServer(
    name="llm-eval-mcp",
    version="0.1.0",
    instructions=(
        "Boîte à outils d'évaluation de fiabilité des agents LLM : "
        "génération de tâches adversariales, jugement automatisé et statistiques."
    ),
)


@mcp.tool()
def generate_adversarial_tasks(
    category: FailureCategory,
    count: int = 5,
) -> list[AdversarialTask]:
    """Génère des tâches de test adversariales pour éprouver un agent LLM.

    Appelle cet outil quand tu veux construire un jeu de tests ciblant un mode
    de défaillance précis (hallucination, injection de prompt, non-respect des
    consignes, sortie dangereuse, erreur de raisonnement, mauvais usage d'outil).
    Chaque tâche inclut le comportement attendu, qui sert de référence au juge.

    Args:
        category: La famille de défaillance à cibler.
        count: Nombre de tâches à générer (entre 1 et 50). Par défaut 5.

    Returns:
        La liste des tâches adversariales annotées.
    """
    return _generate(category=category, count=count)


@mcp.tool()
def run_llm_as_judge(
    prompt: str,
    expected_behavior: str,
    answer: str,
) -> JudgeVerdict:
    """Évalue la réponse d'un agent à une consigne, via un juge LLM.

    Compare `answer` au `expected_behavior` attendu pour la `prompt` donnée, et
    renvoie un verdict structuré : réussite (booléen), score de 1 à 5, et une
    courte justification. L'évaluation est enregistrée pour l'historique.

    Args:
        prompt: La consigne qui avait été soumise à l'agent.
        expected_behavior: Le comportement attendu d'un bon agent (référence).
        answer: La réponse produite par l'agent, à évaluer.

    Returns:
        Le verdict structuré du juge.
    """
    return wiring.judge_and_record(prompt, expected_behavior, answer)


@mcp.tool()
def get_eval_stats() -> EvalStats:
    """Renvoie des statistiques agrégées sur toutes les évaluations enregistrées.

    Fournit le nombre total d'évaluations, le nombre de réussites, le taux de
    réussite (0 à 1) et le score moyen. Ne nécessite pas de clé API.

    Returns:
        Les statistiques agrégées.
    """
    return wiring.get_stats()


def main() -> None:
    """Point d'entrée : démarre le serveur en transport stdio (local)."""
    from dotenv import load_dotenv

    load_dotenv()  # rend GROQ_API_KEY disponible si un fichier .env existe
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
