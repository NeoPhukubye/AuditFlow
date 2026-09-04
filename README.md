# AuditFlow

Automated Support Ticket Triage & Compliance Auditor.

A two-agent pipeline that drafts customer-facing support replies and then
**strictly** verifies them against company policy before anything is sent.

- **Agent 1 — Drafter** (Generator, `temperature=0.2`): classifies the ticket,
  diagnoses the probable cause, and drafts a reply. Retrieval-Augmented
  Generation (RAG) surfaces relevant knowledge-base articles so the draft can
  cite real product docs.
- **Agent 2 — Auditor** (Discriminator, `temperature=0.0`): adversarially
  checks the draft against the support rules. It either `APPROVED`s the draft or
  returns a precise, corrective fix instruction.

A **critique-and-refine loop** feeds the auditor's fix instruction back into the
drafter for up to `max_retries` attempts. If every attempt is rejected, the
ticket is **automatically escalated to a human review queue** (hard failover).

## Architecture

```
ticket ──► Drafter ──► draft ──► Auditor ──► APPROVED ──► final reply
               ▲                    │
               │         REJECTED + fix instruction
               └── (retry loop, max_retries)
                                    │
                         all attempts rejected ──► EscalationHandler ──► escalations.jsonl
```

The RAG layer (`knowledge_base.py`) indexes the `kb/` directory and is injected
into the drafter prompt as citable context. It uses Gemini dense embeddings in
production and falls back to a local TF-IDF provider when no API key is set, so
the retrieval path runs fully offline.

## Prerequisites

- Python 3.10+
- A Gemini API key (set as `GEMINI_API_KEY`)

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY="your-api-key"
```

## Usage

### Run the demo pipeline

```bash
python main.py
```

This runs the end-to-end Drafter → Auditor loop against a sample ticket that
demands an immediate refund, printing the RAG retrieval, each attempt's
decision, and the final result.

### Ingest tickets over HTTP (webhook)

```bash
uvicorn webhook:app --host 0.0.0.0 --port 8000
```

| Method   | Endpoint            | Description                              |
|----------|---------------------|------------------------------------------|
| `GET`    | `/health`           | Liveness + component status.             |
| `POST`   | `/webhooks/ticket`  | Accept a normalised ticket payload.      |
| `POST`   | `/webhooks/zendesk` | Accept a Zendesk webhook payload.        |
| `POST`   | `/webhooks/jira`    | Accept a Jira webhook payload.           |

The Zendesk and Jira endpoints normalise their native payloads (including Jira's
Atlassian Document Format descriptions) into the common `TicketPayload`.

Optional security — set `WEBHOOK_SECRET` to require an `X-Webhook-Secret` header
on every POST.

### Run as a library

```python
from main import process_ticket

result = process_ticket(
    "Your service has been down for 4 hours! I lost client data and demand "
    "an immediate full refund of my R1,500 subscription."
)
print(result["success"], result["attempts"])
```

## Knowledge Base

Articles live in `kb/` as `.json` or `.md`/`.txt` files. Add a new article to
extend the RAG coverage; it is indexed automatically on startup.

## Configuration

| Variable                    | Default     | Description                                  |
|-----------------------------|-------------|----------------------------------------------|
| `GEMINI_API_KEY`            | —           | Google Gemini API key (required for the LLM).|
| `GOOGLE_API_KEY`            | —           | Alias accepted by the SDK.                   |
| `GEMINI_EMBEDDING_MODEL`    | `text-embedding-004` | Embedding model for RAG.            |
| `AUDITFLOW_FORCE_LOCAL_KB`  | unset       | Force the offline TF-IDF embedding provider. |
| `AUDITFLOW_PIPELINE_TIMEOUT`| `300`       | Webhook pipeline timeout (seconds).          |
| `WEBHOOK_SECRET`            | unset       | Shared secret for webhook endpoints.         |
| `HOST` / `PORT`             | `0.0.0.0` / `8000` | Webhook server bind address.         |

## Web Frontend & GitHub Pages

A dependency-free static frontend lives in `web/` (`index.html`,
`style.css`, `app.js`). It exposes a live demo form that submits a ticket to
your deployed FastAPI backend and renders the triage result (category, draft
reply, actions, audit decision and cited KB sources).

On every push to `main`, the `.github/workflows/pages.yml` workflow:

1. Injects the `API_URL` GitHub secret into `web/config.json` (the backend URL).
2. Validates the frontend assets.
3. Publishes `web/` to GitHub Pages (served from the `gh-pages` branch).

### Configure the backend URL

Set a repository secret named **`API_URL`** pointing at your deployed backend,
for example `https://your-app.onrender.com`. When unset, the site shows a
"backend not configured" notice.

### Deploy the backend

GitHub Pages is static-hosting only; the FastAPI backend runs separately. The
deploy command targets the ASGI app exposed on `main` (so host with
`uvicorn main:app`) — the same entry the earlier Render deploy used:

```bash
export GEMINI_API_KEY="your-api-key"
export WEBHOOK_SECRET="your-secret"      # optional
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Platforms that build from the `requirements.txt` and run `uvicorn main:app`
(e.g. Render, Fly.io) work out of the box.

### Run the frontend locally

```bash
python3 -m http.server 8000 --directory web
```

## License

MIT
