# 07 — Deployment

## Local

```bash
python run.py                  # http://127.0.0.1:8000
python run.py --port 9000
python run.py --demo           # terminal demo, no browser
python run.py --test           # 96 tests
python run.py --eval           # evaluation harness
python run.py --build-data     # regenerate seed data
```

Requires Python 3.9+, `numpy`, `networkx`. Nothing else.

### Optional: PDF résumé upload

DOCX, ODT, RTF and TXT upload works out of the box. PDF needs one backend:

```bash
pip install pdfplumber              # best layout fidelity
# or  pip install pypdf             # lighter
# or  apt-get install poppler-utils # provides the pdftotext binary
```

Without one, the interface greys out PDF and says so — it does not accept a
file and fail afterwards. Check `GET /api/upload/formats` to see what is live.

## Enabling production providers

Everything is environment-driven; no code changes.

```bash
# real dense embeddings
pip install sentence-transformers
export SKILLPATH_EMBEDDINGS=sentence_transformers
export SKILLPATH_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# local LLM explanations via Ollama
ollama serve && ollama pull llama3.1:8b
export SKILLPATH_EXPLAINER=ollama
export SKILLPATH_LLM_MODEL=llama3.1:8b
export SKILLPATH_LLM_BASE_URL=http://localhost:11434

# or any OpenAI-compatible endpoint
export SKILLPATH_EXPLAINER=openai
export SKILLPATH_LLM_BASE_URL=https://your-endpoint
export SKILLPATH_LLM_API_KEY=sk-...

# real ESCO taxonomy
export SKILLPATH_ESCO_SKILLS_CSV=/data/esco/skills_en.csv
export SKILLPATH_ESCO_RELATIONS_CSV=/data/esco/broaderRelations_en.csv
```

If a provider cannot be constructed the system logs why and falls back to the
offline implementation rather than refusing to start. Check `/api/health` to
see which providers are actually live — the interface shows this in its header.

### All environment variables

| Variable | Default |
|---|---|
| `SKILLPATH_EMBEDDINGS` | `hashing` |
| `SKILLPATH_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` |
| `SKILLPATH_EMBEDDING_DIM` | `256` |
| `SKILLPATH_GRAPH_BACKEND` | `networkx` |
| `SKILLPATH_EXPLAINER` | `grounded_template` |
| `SKILLPATH_LLM_MODEL` | `llama3.1:8b` |
| `SKILLPATH_LLM_BASE_URL` | `http://localhost:11434` |
| `SKILLPATH_LLM_API_KEY` | *(empty)* |
| `SKILLPATH_QUALIFY_THRESHOLD` | `0.62` |
| `SKILLPATH_MIN_GROUNDING` | `0.80` |
| `SKILLPATH_MAX_CF_SKILLS` | `5` |
| `SKILLPATH_TOP_K` | `10` |
| `SKILLPATH_RRF_K` | `60` |
| `SKILLPATH_HOST` / `SKILLPATH_PORT` | `127.0.0.1` / `8000` |

## FastAPI (production)

```bash
pip install fastapi "uvicorn[standard]" pydantic
uvicorn server.app_fastapi:app --host 0.0.0.0 --port 8000 --workers 4
```

Adds OpenAPI docs at `/docs`. Identical routes, so the interface works
unchanged.

> This module could not be executed in the environment where the project was
> authored (FastAPI was not installable there). Its handlers delegate to
> `skillpath.pipeline`, which is fully covered by the test suite — but verify
> it in your own environment before relying on it.

## Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python -m skillpath.data.build_seed_data && python -m unittest discover -s tests
ENV SKILLPATH_HOST=0.0.0.0 SKILLPATH_PORT=8000
EXPOSE 8000
CMD ["python", "run.py"]
```

```bash
docker build -t skillpath-xr . && docker run -p 8000:8000 skillpath-xr
```

The build runs the test suite, so a broken image fails to build.

## Before exposing this publicly

The current server is appropriate for a demo, a viva, and local development.
For anything internet-facing you would need to add, at minimum:

- authentication and per-client rate limiting
- HTTPS termination (a reverse proxy)
- a CORS policy if the interface is served from another origin
- a data-retention policy — résumés are personal data; this build holds them
  only in memory for the duration of a request and writes none to disk
