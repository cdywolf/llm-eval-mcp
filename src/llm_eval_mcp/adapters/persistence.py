"""Adaptateur de persistance : l'implémentation *réelle* du repository.

Seul fichier qui connaît SQLAlchemy et le dialecte de base. Le modèle ORM
(`EvalRunRow`, la table) est distinct du modèle de domaine (`EvalRun`) : le
repository traduit de l'un à l'autre, ce qui garde le domaine ignorant du SQL.

L'URL de connexion vient de DATABASE_URL. Par défaut, un fichier SQLite local ;
en production on mettra une URL PostgreSQL — le code ne change pas (c'est tout
l'intérêt de l'ORM).
"""

from __future__ import annotations

import os
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from llm_eval_mcp.domain.eval_run import EvalRun

DEFAULT_DATABASE_URL = "sqlite:///./eval_runs.db"


def _normalize_db_url(url: str) -> str:
    """Adapte l'URL au driver psycopg3 attendu par SQLAlchemy.

    Les plateformes (ex. Render) fournissent souvent `postgres://` ou
    `postgresql://`, que SQLAlchemy interpréterait avec psycopg2. On force le
    dialecte psycopg3 (`postgresql+psycopg://`), celui qu'on installe en prod.
    """
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


class Base(DeclarativeBase):
    """Base déclarative SQLAlchemy pour nos tables."""


class EvalRunRow(Base):
    """Modèle ORM : la table `eval_runs`. Distinct du modèle de domaine."""

    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prompt: Mapped[str] = mapped_column(Text)
    expected_behavior: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    passed: Mapped[bool] = mapped_column(Boolean)
    score: Mapped[int] = mapped_column(Integer)
    justification: Mapped[str] = mapped_column(Text)
    model: Mapped[str] = mapped_column(String(120))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


def _to_row(run: EvalRun) -> EvalRunRow:
    """Traduit le modèle de domaine vers le modèle ORM (on laisse la base gérer l'id)."""
    return EvalRunRow(
        prompt=run.prompt,
        expected_behavior=run.expected_behavior,
        answer=run.answer,
        passed=run.passed,
        score=run.score,
        justification=run.justification,
        model=run.model,
        created_at=run.created_at,
    )


def _to_domain(row: EvalRunRow) -> EvalRun:
    """Traduit le modèle ORM vers le modèle de domaine."""
    return EvalRun(
        id=row.id,
        prompt=row.prompt,
        expected_behavior=row.expected_behavior,
        answer=row.answer,
        passed=row.passed,
        score=row.score,
        justification=row.justification,
        model=row.model,
        created_at=row.created_at,
    )


class SqlEvalRunRepository:
    """Repository SQL. Reçoit un `engine` (injection), conforme au port `EvalRunRepository`."""

    def __init__(self, engine) -> None:
        self._engine = engine

    @classmethod
    def from_env(cls) -> SqlEvalRunRepository:
        """Construit le repository depuis DATABASE_URL (SQLite local par défaut)."""
        from dotenv import load_dotenv

        load_dotenv()
        url = _normalize_db_url(os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL))
        engine = create_engine(url)
        repo = cls(engine)
        repo.create_schema()  # crée la table si elle n'existe pas encore
        return repo

    def create_schema(self) -> None:
        """Crée les tables décrites par les modèles ORM (idempotent)."""
        Base.metadata.create_all(self._engine)

    def save(self, run: EvalRun) -> EvalRun:
        """Enregistre une évaluation et renvoie sa version persistée (avec id)."""
        with Session(self._engine) as session:
            row = _to_row(run)
            session.add(row)
            session.commit()
            session.refresh(row)  # récupère l'id attribué par la base
            return _to_domain(row)

    def list_runs(self) -> list[EvalRun]:
        """Renvoie toutes les évaluations, des plus anciennes aux plus récentes."""
        with Session(self._engine) as session:
            rows = session.scalars(
                select(EvalRunRow).order_by(EvalRunRow.created_at)
            ).all()
            return [_to_domain(r) for r in rows]
