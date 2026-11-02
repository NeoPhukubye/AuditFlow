"""Tests for the AuditFlow CLI entry point (main.cli).

Runs fully offline: a fake Gemini client supplies scripted JSON responses and
conftest.py forces the local TF-IDF KB provider.
"""

from __future__ import annotations

import io
import json
import os
from contextlib import redirect_stdout

import pytest

import main
from main import EscalationHandler, cli

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

    def generate_content(self, model, contents, config=None):
        schema = getattr(config, "response_schema", None) if config else None
        name = getattr(schema, "__name__", "") or ""
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


def _patch_client(drafter_seq, audit_seq):
    client = _FakeClient(_FakeModels(drafter_seq, audit_seq))
    main._client = client
    return client


def _run_cli(argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = cli(argv)
    return code, json.loads(buf.getvalue())


@pytest.fixture(scope="module")
def kb():
    from knowledge_base import create_knowledge_base

    return create_knowledge_base("kb")


def test_cli_approve_first_attempt(kb):
    _patch_client([COMPLIANT], [APPROVE])
    code, result = _run_cli(["--ticket", "refund request", "--quiet"])
    assert code == 0
    assert result["success"] is True
    assert result["attempts"] == 1


def test_cli_retry_then_approve(kb, tmp_path, monkeypatch):
    _patch_client([NON_COMPLIANT, COMPLIANT], [REJECT, APPROVE])
    dest = str(tmp_path / "escalations.jsonl")
    monkeypatch.setattr(main, "get_escalation_handler", lambda: EscalationHandler(dest))
    code, result = _run_cli(
        ["--ticket", "I demand a refund", "--max-retries", "2", "--quiet"]
    )
    assert code == 0
    assert result["success"] is True
    assert result["attempts"] == 2
    assert not os.path.exists(dest)


def test_cli_double_reject_escalates(kb, tmp_path, monkeypatch):
    _patch_client([NON_COMPLIANT], [REJECT])
    dest = str(tmp_path / "escalations.jsonl")
    monkeypatch.setattr(main, "get_escalation_handler", lambda: EscalationHandler(dest))
    code, result = _run_cli(
        ["--ticket", "refund me now", "--max-retries", "2", "--quiet"]
    )
    assert code == 1
    assert result["success"] is False
    assert result["escalated"] is True
    assert result["attempts"] == 2
    with open(dest) as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    assert len(records) == 1
    assert records[0]["attempts"] == 2


def test_cli_output_writes_file(kb, tmp_path):
    _patch_client([COMPLIANT], [APPROVE])
    out = str(tmp_path / "result.json")
    code, result = _run_cli(
        ["--ticket", "generic question", "--quiet", "--output", out]
    )
    assert code == 0
    assert result["success"] is True
    with open(out) as fh:
        written = json.load(fh)
    assert written["success"] is True


def test_cli_demo_runs(kb):
    _patch_client([COMPLIANT], [APPROVE])
    code, result = _run_cli(["--demo", "--quiet"])
    assert code == 0
    assert result["success"] is True


def test_cli_no_key_returns_error(monkeypatch):
    def boom():
        raise RuntimeError("Set the GEMINI_API_KEY environment variable.")

    monkeypatch.setattr(main, "get_client", boom)
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = cli(["--ticket", "x", "--quiet"])
    result = json.loads(buf.getvalue())
    assert code == 1
    assert result["success"] is False
    assert "GEMINI_API_KEY" in result["note"]
