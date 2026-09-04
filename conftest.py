"""Shared pytest configuration for AuditFlow.

Forces the offline TF-IDF embedding provider so the entire suite runs
credential-free and deterministically in CI (no ``GEMINI_API_KEY`` needed).
"""

import os

os.environ.setdefault("AUDITFLOW_FORCE_LOCAL_KB", "1")

import pytest


@pytest.fixture(autouse=True)
def _reset_gemini_client():
    """Ensure each test starts with no cached mock/real Gemini client."""
    import main

    main._client = None
    yield
    main._client = None
