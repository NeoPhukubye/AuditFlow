"""
AuditFlow FastAPI webhook receiver.

Ingests tickets from external ticketing systems (Zendesk, Jira, or a generic
shape) and runs them through the two-agent triage -> audit pipeline.

Run locally::

    export GEMINI_API_KEY="your-api-key"
    uvicorn webhook:app --host 0.0.0.0 --port 8000

Endpoints
---------
* ``GET  /health``               -- liveness + configured components check.
* ``POST /webhooks/ticket``      -- accept a normalised ticket payload.
* ``POST /webhooks/zendesk``     -- accept a Zendesk webhook payload.
* ``POST /webhooks/jira``        -- accept a Jira webhook payload.

Security
---------
If the ``WEBHOOK_SECRET`` environment variable is set, every POST is gated by
an ``X-Webhook-Secret`` header comparison. Omit it for open local testing.
"""
from __future__ import annotations

import asyncio
import hmac
import logging
import os
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.status import HTTP_503_SERVICE_UNAVAILABLE

import main
from main import EscalationHandler, run_pipeline

logger = logging.getLogger("uvicorn.error")

WEBHOOK_SECRET: str = os.getenv("WEBHOOK_SECRET", "")
PIPELINE_TIMEOUT_SECONDS: int = int(os.getenv("AUDITFLOW_PIPELINE_TIMEOUT", "300"))


# -------------------------------------------------------------
# Request schemas
# -------------------------------------------------------------
class TicketPayload(BaseModel):
    """Normalised ticket shape accepted by the generic endpoint."""

    ticket_id: str = Field(..., description="Upstream ticket identifier.")
    source: str = Field(default="generic", description="Originating system.")
    subject: str = Field(default="", description="Short subject/summary line.")
    description: str = Field(..., description="Full customer description of the issue.")
    customer_email: Optional[str] = None
    priority: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class WebhookResponse(BaseModel):
    ticket_id: str
    source: str
    result: dict[str, Any] = Field(description="Raw triage pipeline output.")


# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------
def _constant_time_eq(a: str, b: str) -> bool:
    """Constant-time string comparison to avoid timing oracles."""
    return hmac.compare_digest(a.encode(), b.encode())


def _compose_ticket_text(payload: TicketPayload) -> str:
    parts = [p for p in (payload.subject, payload.description) if p]
    return "\n\n".join(parts) if parts else payload.description


def _extract_jira_text(value: Any) -> str:
    """Render a Jira (Atlassian Document Format) description to plain text."""
    if not value:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if "content" in value:
            return " ".join(_extract_jira_text(c) for c in value["content"])
        if "text" in value:
            return str(value["text"])
    if isinstance(value, list):
        return " ".join(_extract_jira_text(v) for v in value)
    return str(value)


def _normalize_zendesk(body: dict[str, Any]) -> TicketPayload:
    ticket = body.get("ticket", body)
    requester = ticket.get("requester", {}) or {}
    return TicketPayload(
        ticket_id=str(ticket.get("id", ticket.get("url", ""))),
        source="zendesk",
        subject=str(ticket.get("subject", "")),
        description=str(ticket.get("description", "")),
        customer_email=requester.get("email") or ticket.get("requester_email"),
        priority=ticket.get("priority"),
        metadata={"status": ticket.get("status"), "tags": ticket.get("tags", [])},
    )


def _normalize_jira(body: dict[str, Any]) -> TicketPayload:
    issue = body.get("issue", body)
    fields = issue.get("fields", {}) or {}
    webhook = body.get("webhook", {}) or {}
    return TicketPayload(
        ticket_id=str(issue.get("id", issue.get("key", webhook.get("issueId", "")))),
        source="jira",
        subject=str(fields.get("summary", "")),
        description=_extract_jira_text(fields.get("description", "")),
        customer_email=fields.get("reporter", {}).get("emailAddress"),
        priority=fields.get("priority", {}).get("name"),
        metadata={
            "project": fields.get("project", {}).get("key"),
            "issuetype": fields.get("issuetype", {}).get("name"),
        },
    )


def _run_pipeline_sync(ticket_text: str) -> dict[str, Any]:
    """Blocking pipeline call intended to run in a worker thread."""
    kb = main.get_knowledge_base()
    escalator = EscalationHandler()
    return run_pipeline(
        ticket_text,
        max_retries=2,
        knowledge_base=kb,
        escalation_handler=escalator,
        verbose=False,
    )


# -------------------------------------------------------------
# Routes
# -------------------------------------------------------------
def register_routes(app: FastAPI) -> None:
    @app.get("/health")
    async def health() -> dict[str, Any]:
        kb = main.get_knowledge_base()
        return {
            "status": "ok",
            "knowledge_base_loaded": kb is not None,
            "kb_chunks": kb.chunk_count if kb is not None else 0,
            "gemini_configured": bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
            "webhook_secret_configured": bool(WEBHOOK_SECRET),
        }

    @app.post("/webhooks/ticket", response_model=WebhookResponse)
    async def ingest_generic(payload: TicketPayload) -> WebhookResponse:
        return await _process(payload)

    @app.post("/webhooks/zendesk", response_model=WebhookResponse)
    async def ingest_zendesk(request: Request) -> WebhookResponse:
        body = await request.json()
        try:
            payload = _normalize_zendesk(body)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=422,
                detail=f"Unparseable Zendesk payload: {exc}",
            ) from exc
        return await _process(payload)

    @app.post("/webhooks/jira", response_model=WebhookResponse)
    async def ingest_jira(request: Request) -> WebhookResponse:
        body = await request.json()
        try:
            payload = _normalize_jira(body)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=422,
                detail=f"Unparseable Jira payload: {exc}",
            ) from exc
        return await _process(payload)

    async def _process(payload: TicketPayload) -> WebhookResponse:
        ticket_text = _compose_ticket_text(payload)
        logger.info("Received ticket %s (%s); running triage pipeline.", payload.ticket_id, payload.source)
        try:
            result = await asyncio.wait_for(
                run_in_threadpool(_run_pipeline_sync, ticket_text),
                timeout=PIPELINE_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ticket processing exceeded the configured timeout.",
            )
        except RuntimeError as exc:
            raise HTTPException(
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            )
        return WebhookResponse(ticket_id=payload.ticket_id, source=payload.source, result=result)


# -------------------------------------------------------------
# Application factory
# -------------------------------------------------------------
def create_app() -> FastAPI:
    app = FastAPI(
        title="AuditFlow Webhook Receiver",
        description="Two-agent support triage + compliance auditor.",
        version="1.0.0",
    )

    @app.middleware("http")
    async def _enforce_webhook_secret(request: Request, call_next):
        if WEBHOOK_SECRET:
            provided = request.headers.get("x-webhook-secret", "")
            if not _constant_time_eq(provided, WEBHOOK_SECRET):
                logger.warning("Rejected webhook: invalid or missing X-Webhook-Secret.")
                return JSONResponse(
                    status_code=401, content={"detail": "Invalid webhook secret."}
                )
        return await call_next(request)

    register_routes(app)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")))
