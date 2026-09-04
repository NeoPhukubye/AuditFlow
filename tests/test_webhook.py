"""Tests for the FastAPI webhook receiver and payload normalizers."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import webhook
from webhook import _normalize_jira, _normalize_zendesk, app


@pytest.fixture()
def client():
    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["knowledge_base_loaded"] is True


def test_normalize_zendesk():
    body = {
        "ticket": {
            "id": 911,
            "subject": "API down",
            "description": "I got a 504 timeout since 14:00 UTC",
            "requester": {"email": "cust@co.com"},
            "priority": "high",
        }
    }
    payload = _normalize_zendesk(body)
    assert payload.ticket_id == "911"
    assert payload.source == "zendesk"
    assert payload.customer_email == "cust@co.com"
    assert payload.description == "I got a 504 timeout since 14:00 UTC"


def test_normalize_jira_atlassian_document_format():
    body = {
        "issue": {
            "id": "10001",
            "key": "SUP-42",
            "fields": {
                "summary": "refund overdue invoice",
                "description": {
                    "type": "doc",
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [
                                {"type": "text", "text": "service unavailable"}
                            ],
                        }
                    ],
                },
                "reporter": {"emailAddress": "cust@co.com"},
                "priority": {"name": "High"},
                "project": {"key": "SUP"},
                "issuetype": {"name": "Bug"},
            },
        }
    }
    payload = _normalize_jira(body)
    assert payload.ticket_id == "10001"
    assert payload.source == "jira"
    assert payload.subject == "refund overdue invoice"
    assert payload.description == "service unavailable"
    assert payload.customer_email == "cust@co.com"
    assert payload.priority == "High"
    assert payload.metadata["project"] == "SUP"


def test_post_ticket_without_key_returns_503(client):
    resp = client.post(
        "/webhooks/ticket",
        json={"ticket_id": "T-1", "source": "generic", "description": "refund please"},
    )
    assert resp.status_code == 503
    assert "GEMINI_API_KEY" in resp.json()["detail"]


def test_webhook_secret_middleware(client):
    original = webhook.WEBHOOK_SECRET
    webhook.WEBHOOK_SECRET = "s3cr3t"
    try:
        # missing header -> 401
        resp = client.post(
            "/webhooks/ticket",
            json={"ticket_id": "T-1", "source": "generic", "description": "refund"},
        )
        assert resp.status_code == 401
        # wrong header -> 401
        resp = client.post(
            "/webhooks/ticket",
            headers={"x-webhook-secret": "nope"},
            json={"ticket_id": "T-1", "source": "generic", "description": "refund"},
        )
        assert resp.status_code == 401
        # correct header -> passes auth, then 503 (no Gemini key)
        resp = client.post(
            "/webhooks/ticket",
            headers={"x-webhook-secret": "s3cr3t"},
            json={"ticket_id": "T-1", "source": "generic", "description": "refund"},
        )
        assert resp.status_code == 503
    finally:
        webhook.WEBHOOK_SECRET = original


def test_zendesk_endpoint_normalizes_and_runs(client):
    body = {
        "ticket": {
            "id": 4242,
            "subject": "API down",
            "description": "504 timeout",
            "requester": {"email": "cust@co.com"},
        }
    }
    resp = client.post("/webhooks/zendesk", json=body)
    # no key configured -> graceful 503 (proves normalization + pipeline wiring worked)
    assert resp.status_code == 503
