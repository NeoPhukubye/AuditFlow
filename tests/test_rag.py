"""Tests for the Retrieval-Augmented Generation (RAG) knowledge base."""

from __future__ import annotations

import numpy as np

from knowledge_base import (
    KnowledgeBase,
    LocalEmbeddingProvider,
    create_knowledge_base,
    format_context_for_prompt,
)


def test_knowledge_base_loads_kb_documents():
    kb = create_knowledge_base("kb")
    assert kb.chunk_count > 0


def test_search_returns_relevant_results():
    kb = create_knowledge_base("kb")
    results = kb.search("I demand an immediate refund", k=3)
    assert results
    assert results[0].score > 0
    assert results[0].chunk.category == "Billing"


def test_search_empty_query_returns_nothing():
    kb = create_knowledge_base("kb")
    assert kb.search("", k=3) == []


def test_search_respects_category_filter():
    kb = create_knowledge_base("kb")
    billing = kb.search("refund", k=5, category_filter="Billing")
    assert billing
    assert all(r.chunk.category == "Billing" for r in billing)


def test_format_context_is_citable():
    kb = create_knowledge_base("kb")
    results = kb.search("refund policy", k=2)
    ctx = format_context_for_prompt(results)
    assert "KB Title" in ctx
    assert results[0].chunk.title in ctx
    assert "relevance" in ctx


def test_local_provider_is_deterministic_and_normalised():
    provider = LocalEmbeddingProvider()
    provider.fit(["hello world foo", "bar baz qux", "unrelated terms"])
    vecs = provider.embed(["hello world foo", "bar baz qux"])
    assert vecs.shape == (2, provider.dimensionality)
    # embeddings are L2-normalised -> cosine sim == dot product
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0)


def test_local_provider_is_idempotent_across_repeats():
    kb = KnowledgeBase(LocalEmbeddingProvider())
    kb.load_directory("kb")
    first = kb.search("service timeout", k=3)
    second = kb.search("service timeout", k=3)
    assert [r.chunk.chunk_id for r in first] == [r.chunk.chunk_id for r in second]
