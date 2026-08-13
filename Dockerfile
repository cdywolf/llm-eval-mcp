# syntax=docker/dockerfile:1

# ============================================================================
# Étape 1 — "builder" (l'atelier) : on installe le projet et ses dépendances
# dans un environnement virtuel isolé. Cette étape peut être "lourde" : elle
# ne sera PAS livrée, seule la suivante devient l'image finale.
# ============================================================================
FROM python:3.12-slim AS builder

# Un venv dédié, qu'on copiera tel quel dans l'image finale.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# On copie le strict nécessaire au build : le pyproject, le README (lu par la
# config du projet) et le code source. Puis on installe le projet + ses
# dépendances d'exécution dans le venv.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# ============================================================================
# Étape 2 — image finale (la livraison) : base neuve et propre. On ne récupère
# QUE le venv déjà construit — aucun outil de build, aucun cache pip.
# ============================================================================
FROM python:3.12-slim

# On emporte seulement le "produit fini" de l'atelier.
COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Principe de moindre privilège : on ne tourne pas en root.
RUN useradd --create-home appuser
USER appuser
WORKDIR /home/appuser

# Commande de démarrage : le service web (FastAPI) servi par Uvicorn.
# On écoute sur 0.0.0.0 (toutes les interfaces, requis en conteneur) et sur le
# port fourni par la plateforme via $PORT (8000 par défaut en local).
# Forme "shell" pour permettre l'expansion de la variable $PORT.
CMD uvicorn llm_eval_mcp.api:app --host 0.0.0.0 --port ${PORT:-8000}
