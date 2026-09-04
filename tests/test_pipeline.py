"""Tests for the two-agent critique-and-refine loop and human failover.

These run fully offline: a fake Gemini client supplies scripted JSON
responses, and the RAG layer uses the local TF-IDF provider (see conftest.py).
"""

from __future__ import annotations

import json
import os

import pytest

from knowledge_base import create_knowledge_base
from main import (
    EscalationHandler,
    run_pipeline,
)

NON_COMPLIANT = json.dumps(
    {
        "category": "Billing",
        "priority": "High",
        "root_cause_summary": "outage",
        "proposed_reply": "I have processed your full refund of R1,500 — credited back immediately.",
        "actions_taken": ["Refunded R1,500"],
    }
)
COMPLIANT = json.dumps(
    {
        "category": "Billing",
        "priority": "High",
        "root_cause_summary": "outage",
        "proposed_reply": "We have submitted your request to the billing department for review within 3-5 business days.",
        "actions_taken": ["Opened incident", "Routed to billing dept"],
    }
)
REJECT = json.dumps(
    {
        "status": "REJECTED",
        "policy_compliant": False,
        "tone_acceptable": True,
        "critique_points": ["Draft promises an immediate refund, violating Rule 1."],
        "instruction_for_fix": "Rewrite the reply to route the refund to billing for a 3-5 day review.",
    }
)
APPROVE = json.dumps(
    {
        "status": "APPROVED",
        "policy_compliant": True,
        "tone_acceptable": True,
        "critique_points": [],
        "instruction_for_fix": None,
    }
)


class _FakeResp:
    def __init__(self, text: str):
        self.text = text


class _FakeModels:
    def __init__(self, drafter_seq, audit_seq):
        self._drafter = list(drafter_seq)
        self._audit = list(audit_seq)
        self._nd = 0
        self._nu = 0
        self.seen_prompts: list[str] = []

    def generate_content(self, model, contents, config=None):
        schema = getattr(config, "response_schema", None) if config else None
        name = getattr(schema, "__name__", "") or ""
        self.seen_prompts.append(contents)
        if name == "DraftResolution":
            text = self._drafter[self._nd % len(self._drafter)]
            self._nd += 1
        else:
            text = self._audit[self._nu % len(self._audit)]
            self._nu += 1
        return _FakeResp(text)


class _FakeClient:
    def __init__(self, models):
        self.models = models


@pytest.fixture(scope="module")
def kb():
    return create_knowledge_base("kb")


def _patch_client(drafter_seq, audit_seq):
    import main

    client = _FakeClient(_FakeModels(drafter_seq, audit_seq))
    main._client = client
    return client


def test_approve_on_first_attempt(kb):
    _patch_client([COMPLIANT], [APPROVE])
    result = run_pipeline(
        "refund request",
        max_retries=2,
        knowledge_base=kb,
        escalation_handler=None,
        verbose=False,
    )
    assert result["success"] is True
    assert result["attempts"] == 1


def test_retry_then_approve(kb, tmp_path):
    client = _patch_client([NON_COMPLIANT, COMPLIANT], [REJECT, APPROVE])
    dest = str(tmp_path / "escalations.jsonl")
    escalator = EscalationHandler(dest)
    result = run_pipeline(
        "I demand an immediate refund and lost data on db-prod-01",
        max_retries=2,
        knowledge_base=kb,
        escalation_handler=escalator,
        verbose=False,
    )
    assert result["success"] is True
    assert result["attempts"] == 2
    # KB context is injected into the drafter prompt
    assert any("Knowledge-Base Context" in p for p in client.models.seen_prompts)
    # no escalation record written on approval
    assert not os.path.exists(dest)


def test_double_reject_triggers_escalation(kb, tmp_path):
    _patch_client([NON_COMPLIANT], [REJECT])
    dest = str(tmp_path / "escalations.jsonl")
    escalator = EscalationHandler(dest)
    result = run_pipeline(
        "refund me now",
        max_retries=2,
        knowledge_base=kb,
        escalation_handler=escalator,
        verbose=False,
    )
    assert result["success"] is False
    assert result["escalated"] is True
    assert result["attempts"] == 2
    with open(dest) as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    assert len(records) == 1
    assert records[0]["attempts"] == 2


def test_no_kb_still_runs_pipeline():
    """Pipeline degrades gracefully when no knowledge base is supplied."""
    _patch_client([COMPLIANT], [APPROVE])
    result = run_pipeline(
        "generic question",
        max_retries=2,
        knowledge_base=None,
        escalation_handler=None,
        verbose=False,
    )
    assert result["success"] is True
