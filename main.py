"""
AuditFlow — Automated Support Ticket Triage & Compliance Auditor.

Two-agent architecture:
  * DraftAgent  (Generator) — classifies the ticket, diagnoses the probable
    cause and drafts a customer-facing reply, optionally grounded in a
    knowledge base (RAG) so recommendations cite real product docs.
  * AuditAgent  (Discriminator) — strictly verifies the draft against company
    policy at temperature 0.0 and either APPROVES it or issues a fix command.

A critique-and-refine loop re-injects the auditor's fix instruction back into
the drafter for up to ``max_retries`` attempts. If every attempt is rejected
(two consecutive REJECTED decisions), the ticket is escalated to a human
review queue as a hard failover.

The Gemini client is initialised lazily so the module can be imported and the
offline RAG / retrieval path exercised without ``GEMINI_API_KEY`` set.
"""

import json
import os
from datetime import datetime, timezone
from typing import Literal

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from knowledge_base import (
    KnowledgeBase,
    RetrievalResult,
    create_knowledge_base,
    format_context_for_prompt,
)


# -------------------------------------------------------------
# 1. Pydantic Schemas for Structured Agent Output
# -------------------------------------------------------------
class DraftResolution(BaseModel):
    category: Literal[
        "Billing", "Technical Support", "Account Access", "General Inquiry"
    ]
    priority: Literal["Low", "Medium", "High", "Critical"]
    root_cause_summary: str = Field(
        description="One-sentence technical summary of the user issue."
    )
    proposed_reply: str = Field(description="Customer-facing response draft.")
    actions_taken: list[str] = Field(
        description="Specific troubleshooting or backend remediation steps proposed."
    )


class AuditEvaluation(BaseModel):
    status: Literal["APPROVED", "REJECTED"]
    policy_compliant: bool = Field(
        description="False if promises unauthorized refunds, shares internal secrets, or gives inaccurate guarantees."
    )
    tone_acceptable: bool = Field(
        description="True if empathetic, professional, and directly actionable."
    )
    critique_points: list[str] = Field(
        default_factory=list,
        description="Specific failure reasons if rejected, empty if approved.",
    )
    instruction_for_fix: str | None = Field(
        None,
        description="Clear, corrective command telling the Drafter how to rewrite.",
    )


# -------------------------------------------------------------
# 2. Hard-Failover Escalation Handler
# -------------------------------------------------------------
class EscalationHandler:
    """Hard failover: persists rejected tickets to a human-review queue (JSONL).

    Invoked when the critique-and-refine loop exhausts ``max_retries`` without
    an APPROVED draft, i.e. two consecutive REJECTED decisions by the auditor.
    """

    def __init__(self, dest: str = "escalations.jsonl") -> None:
        self.dest = dest
        directory = os.path.dirname(dest)
        if directory:
            os.makedirs(directory, exist_ok=True)

    def escalate(
        self,
        ticket_text: str,
        attempts: int,
        last_draft: DraftResolution | None,
        last_audit: AuditEvaluation | None,
    ) -> dict:
        record = {
            "escalated_at": datetime.now(timezone.utc).isoformat(),
            "ticket_text": ticket_text,
            "attempts": attempts,
            "final_draft": last_draft.model_dump() if last_draft else None,
            "audit_notes": last_audit.model_dump() if last_audit else None,
        }
        with open(self.dest, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record


# -------------------------------------------------------------
# 3. Company Knowledge & Policy Guardrails
# -------------------------------------------------------------
COMPANY_POLICY = """
Support Rules & Constraints:
1. NEVER promise a refund directly. Only state: "We have submitted your request to the billing department for review within 3-5 business days."
2. NEVER mention internal IP addresses, database schemas, or internal server paths.
3. For API or technical timeouts, always request request IDs, timestamps, and error codes.
4. Keep the tone calm, direct, and empathetic. Do not use corporate dismissive fluff like "we apologize for any inconvenience caused."
"""

KB_USAGE_INSTRUCTION = """
Knowledge Base Usage:
- You will receive retrieved knowledge-base (KB) excerpts for the ticket.
- When applying KB guidance, cite it explicitly as [KB: <Title>] inside proposed_reply and/or actions_taken.
- NEVER fabricate a KB citation. If no relevant KB context is provided, proceed from first principles but still obey every support rule above.
"""

DRAFTER_SYSTEM_INSTRUCTION = f"""
You are an expert Enterprise Technical Support Drafter.
Your goal is to inspect incoming user tickets, classify them accurately, diagnose the probable cause, and draft a high-clarity resolution reply.

Adhere strictly to this internal company policy:
{COMPANY_POLICY}

{KB_USAGE_INSTRUCTION}
"""

AUDITOR_SYSTEM_INSTRUCTION = f"""
You are a strict, adversarial Support Quality & Compliance Auditor.
Your job is to challenge and verify drafts produced by the Support Drafter.
Evaluate the draft ruthlessly against these exact business rules:
{COMPANY_POLICY}

If the draft violates ANY policy (e.g. promising a refund, exposing internal internals, using banned generic platitudes, or missing technical detail), set status to REJECTED and provide an unambiguous fix instruction.
Only set status to APPROVED if all criteria are met without exception.
"""


# -------------------------------------------------------------
# 4. Lazy Gemini Client (deferred so the module imports without a key)
# -------------------------------------------------------------
_client: genai.Client | None = None


def get_client() -> genai.Client:
    """Initialise and cache the Gemini client on first use."""
    global _client
    if _client is None:
        try:
            _client = genai.Client()
        except Exception as exc:
            raise RuntimeError(
                "Gemini client could not be initialised. "
                "Set the GEMINI_API_KEY environment variable."
            ) from exc
    return _client


# -------------------------------------------------------------
# 5. Agent Functions
# -------------------------------------------------------------
def draft_agent(
    ticket_text: str,
    previous_critique: str | None = None,
    kb_context: str = "",
) -> DraftResolution:
    """Generate a structured draft, optionally grounded in retrieved KB context."""
    context_block = (
        f"\n\nRelevant Knowledge-Base Context:\n{kb_context}" if kb_context else ""
    )
    if previous_critique:
        context_block += (
            "\n\nCRITICAL FIX REQUIRED (Your previous draft was rejected for the "
            f"following reason):\n{previous_critique}\n"
            "Rewrite the draft to fix this issue completely."
        )
    prompt = f"Ticket Details:\n{ticket_text}{context_block}"

    response = get_client().models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=DRAFTER_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=DraftResolution,
            temperature=0.2,
        ),
    )
    return DraftResolution.model_validate_json(response.text)


def audit_agent(ticket_text: str, draft: DraftResolution) -> AuditEvaluation:
    """Verify a draft against company policy (temperature 0.0, deterministic)."""
    prompt = f"""
Original Customer Ticket:
{ticket_text}

Proposed Draft to Audit:
{draft.model_dump_json(indent=2)}
"""
    response = get_client().models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=AUDITOR_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=AuditEvaluation,
            temperature=0.0,
        ),
    )
    return AuditEvaluation.model_validate_json(response.text)


# -------------------------------------------------------------
# 6. Shared Pipeline Components (lazy singletons)
# -------------------------------------------------------------
_default_kb: KnowledgeBase | None = None
_default_escalator: EscalationHandler | None = None


def get_knowledge_base() -> KnowledgeBase | None:
    """Build (once) and return the shared knowledge base, if available."""
    global _default_kb
    if _default_kb is None:
        try:
            _default_kb = create_knowledge_base("kb")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] Knowledge base unavailable ({exc}); continuing without RAG.")
    return _default_kb


def get_escalation_handler() -> EscalationHandler:
    """Build (once) and return the shared escalation handler."""
    global _default_escalator
    if _default_escalator is None:
        _default_escalator = EscalationHandler()
    return _default_escalator


# -------------------------------------------------------------
# 7. Orchestrator: Critique-and-Refine Loop with Human Failover
# -------------------------------------------------------------
def run_pipeline(
    ticket_text: str,
    max_retries: int = 2,
    knowledge_base: KnowledgeBase | None = None,
    escalation_handler: EscalationHandler | None = None,
    verbose: bool = True,
) -> dict:
    """Run the drafter -> auditor -> (re)draft loop with hard-failover escalation.

    Returns a dict describing the outcome. ``success=True`` when the auditor
    approves within ``max_retries`` attempts; otherwise the ticket is escalated
    to human review and ``escalated=True`` is returned.
    """
    kb = knowledge_base if knowledge_base is not None else get_knowledge_base()
    kb_context = ""
    kb_results: list[RetrievalResult] = []
    if kb is not None:
        kb_results = kb.search(ticket_text, k=3)
        kb_context = format_context_for_prompt(kb_results)
        if verbose and kb_results:
            print(
                f"RAG: retrieved {len(kb_results)} KB chunks "
                f"(top: '{kb_results[0].chunk.title}')."
            )

    feedback: str | None = None
    attempt = 1
    draft: DraftResolution | None = None
    audit: AuditEvaluation | None = None

    while attempt <= max_retries:
        if verbose:
            print(f"\n--- [Attempt {attempt}] Running Drafter Agent ---")
        draft = draft_agent(
            ticket_text, previous_critique=feedback, kb_context=kb_context
        )
        if verbose:
            print(f"Draft Category: {draft.category} | Priority: {draft.priority}")

        if verbose:
            print("--- Running Compliance Auditor Agent ---")
        audit = audit_agent(ticket_text, draft)
        if verbose:
            print(f"Auditor Decision: {audit.status}")

        if audit.status == "APPROVED":
            return {
                "success": True,
                "attempts": attempt,
                "final_draft": draft.model_dump(),
                "audit_notes": audit.model_dump(),
                "kb_sources": [r.chunk.title for r in kb_results],
            }

        if verbose:
            print(f"Critique: {audit.critique_points}")
        feedback = audit.instruction_for_fix
        attempt += 1

    # Hard failover: two consecutive REJECTED decisions -> escalate to human.
    escalated = False
    if escalation_handler is not None:
        escalation_handler.escalate(ticket_text, attempt - 1, draft, audit)
        escalated = True
        if verbose:
            print(
                "\n>>> HARD FAILOVER: ticket escalated to human review queue "
                "(escalations.jsonl)."
            )

    return {
        "success": False,
        "error": "Failed verification loop within max attempts",
        "attempts": attempt - 1,
        "last_draft": draft.model_dump() if draft else None,
        "audit_notes": audit.model_dump() if audit else None,
        "escalated": escalated,
        "kb_sources": [r.chunk.title for r in kb_results],
    }


def process_ticket(ticket_text: str, max_retries: int = 2) -> dict:
    """Convenience entry point wired to the shared KB + escalation singletons."""
    return run_pipeline(
        ticket_text,
        max_retries=max_retries,
        knowledge_base=get_knowledge_base(),
        escalation_handler=get_escalation_handler(),
        verbose=True,
    )


# -------------------------------------------------------------
# 8. Hosted deploy entry point
# -------------------------------------------------------------
# Expose the FastAPI application at ``main:app`` so the service can be served
# with ``uvicorn main:app`` in hosted/staging environments. The web layer lives
# in ``webhook.py``; the import is deferred to the bottom of this module so that
# the pure pipeline (and the CLI demo below) stays importable without FastAPI
# configured, and to avoid a circular import with ``webhook``.
try:
    from webhook import app
except ImportError:  # pragma: no cover - web layer optional for CLI usage
    app = None


# -------------------------------------------------------------
# 9. Demonstration & Verification Traces
# -------------------------------------------------------------
if __name__ == "__main__":
    # Test Case: A frustrated customer demanding an immediate refund for service downtime
    test_ticket = (
        "Your service has been down for 4 hours! I lost client data on production database db-prod-01. "
        "I demand an immediate full refund of my R1,500 subscription and an explanation!"
    )

    # Always demonstrate the offline-capable RAG retrieval layer first.
    kb = get_knowledge_base()
    if kb is not None:
        results = kb.search(test_ticket, k=3)
        print("\n=== Retrieval-Augmented Context (RAG) ===")
        print(format_context_for_prompt(results) or "(no KB hits)")

    try:
        result = process_ticket(test_ticket)
    except RuntimeError as exc:
        result = {
            "success": False,
            "error": str(exc),
            "note": "Set GEMINI_API_KEY to run the live two-agent verification loop.",
        }

    print("\n================ FINAL SYSTEM RESULT ================")
    print(json.dumps(result, indent=2))
