# llm-eval-mcp

[![CI](https://github.com/cdywolf/llm-eval-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/cdywolf/llm-eval-mcp/actions/workflows/ci.yml)

A small platform for evaluating the reliability of LLM agents. It generates adversarial test tasks, scores answers with an LLM acting as a judge, and keeps track of the results so you can see how often a model actually behaves the way it should.

The same logic is exposed two ways: as an **MCP server** (so any MCP client can call the tools) and as a **web API** (so anyone can try it from a browser). Both sit on top of one shared core.

## What it does

The idea is simple. Testing whether an LLM answer is "good" with fixed rules is fragile, so this project leans on three moving parts that work together:

1. It builds **adversarial tasks** across six failure categories (hallucination, instruction following, prompt injection, unsafe output, reasoning errors, tool misuse). Each task comes with the behavior a reliable agent should show, which acts as the reference.
2. It runs an **LLM as judge**: a strong model reads the task, the expected behavior, and a candidate answer, then returns a structured verdict (pass or fail, a score from 1 to 5, and a short justification).
3. It **stores every judgment** and reports aggregate statistics, so a single answer becomes data you can count, compare, and later analyze.

## The tools

The MCP server and the web API both expose the same three capabilities:

`generate_adversarial_tasks` produces test tasks for a chosen failure category. It runs without any LLM, so it is free and deterministic.

`run_llm_as_judge` (or `POST /judge`) scores an answer against its expected behavior and records the result. This is the one call that talks to the judge model.

`get_eval_stats` (or `GET /stats`) returns the totals: how many evaluations ran, how many passed, the pass rate, and the average score.

## How it is built

```
src/llm_eval_mcp/
├── domain/                 # Pure business logic (no network, no provider, no SQL)
│   ├── adversarial.py      #   adversarial task generation
│   ├── judging.py          #   verdict model, Judge contract, prompt, orchestration
│   └── eval_run.py         #   EvalRun entity, EvalRunRepository contract, stats
├── adapters/               # Technical details (network, providers, database)
│   ├── groq_judge.py       #   the real judge, backed by Groq
│   └── persistence.py      #   the real repository, backed by SQLAlchemy
├── wiring.py               # Composition root: assembles the real implementations
├── server.py               # MCP input adapter (stdio transport)
└── api.py                  # HTTP input adapter (FastAPI) with API key auth
tests/
├── test_adversarial.py     # generator tests
├── test_judging.py         # judging tests (fake judge)
├── test_groq_judge.py      # Groq adapter tests (fake client)
├── test_persistence.py     # stats, SQL repository, and server integration
├── test_api.py             # HTTP API tests (TestClient, auth), no network
└── fakes.py                # test doubles (FakeJudge, InMemoryEvalRunRepository)
```

## Why it is built this way

The whole codebase follows one rule: the business logic never depends on a specific tool. It depends on **contracts** instead.

The `domain` package knows nothing about Groq, HTTP, or the database. It defines what a judge is (a `Judge` protocol) and what a store is (an `EvalRunRepository` protocol), and it works against those ideas. The real implementations live in `adapters` and plug into those contracts. A small `wiring` module is the only place that decides which real implementations to use and how to assemble them.

This buys three concrete things. The tests run with in memory doubles, so they need no API key, no network, and no database, and they always give the same result. Swapping Groq for another provider means writing one new adapter, not touching the core. And the two front doors, MCP and HTTP, reuse the exact same logic through `wiring`, so nothing is duplicated.

## Getting started

You need Python 3.10 or newer. From the project root:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

The judge needs a Groq API key, which is free and does not require a card at https://console.groq.com:

```bash
cp .env.example .env             # then paste your key into GROQ_API_KEY
```

Listing the tools works without a key. Only the judge call needs one, since the key is read lazily.

## Running the tests

```bash
pytest
```

Everything runs offline thanks to the test doubles, so this is fast and does not touch Groq or any database.

## Running the MCP server

The MCP server speaks the stdio transport, so you do not use it by hand. The easiest way to inspect it is the official inspector:

```bash
mcp dev src/llm_eval_mcp/server.py
```

That opens MCP Inspector in your browser, where you can list the three tools and call them live.

## Running the web API

```bash
uvicorn llm_eval_mcp.api:app --reload
```

Then open http://localhost:8000/docs for the interactive Swagger page. `POST /judge` expects an `X-API-Key` header whose value matches `API_KEY` in your `.env`. The health check, task generation, and stats endpoints are open.

## Docker

The image uses a two stage build, so the final image carries only what it needs to run and none of the build tooling. It runs as a non root user and starts the API with Uvicorn.

```bash
docker build -t llm-eval-mcp .
docker run --rm -p 8000:8000 --env-file .env llm-eval-mcp
```

## Deployment

`render.yaml` describes a free web service on Render. Push the repository to GitHub, create a Blueprint that points at it, and set the `GROQ_API_KEY` secret in the dashboard. Render generates the `API_KEY` for you and gives you a public URL, with the interactive docs available at `/docs`.

By default the app uses SQLite. In a container that storage is temporary, so records reset on each redeploy, which is fine for a demo. For durable storage, point `DATABASE_URL` at a managed PostgreSQL instance. Thanks to SQLAlchemy and a small URL normalizer, the switch needs no code change, only the environment variable.

## License

MIT.
