"""
embeddings.py
─────────────
Batched embedding generation via the OpenAI Embeddings API.

Design notes:
  • Texts are split into batches of BATCH_SIZE and sent in parallel using
    concurrent.futures to maximise throughput without overwhelming the API.
  • Each text is truncated to 8000 tokens before sending (model hard limit is
    8191; we leave a small safety margin).
  • Retries with exponential back-off are handled by the openai client library
    (httpx underneath) — no extra retry logic needed here.
  • The function preserves input order: embeddings[i] corresponds to texts[i].
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Sequence

logger = logging.getLogger(__name__)

# Rough character limit to avoid exceeding the token limit.
# text-embedding-3-small has an 8191-token context; 4 chars ≈ 1 token → 32 000 chars.
_MAX_CHARS = 30_000


def generate_embeddings(
    texts: list[str],
    api_key: str,
    model: str = "text-embedding-3-small",
    batch_size: int = 100,
    max_workers: int = 4,
) -> list[list[float]]:
    """
    Generate embeddings for a list of texts.

    Args:
        texts:       Input strings (order preserved in output).
        api_key:     OpenAI API key.
        model:       Embedding model name.
        batch_size:  Number of texts per API call.
        max_workers: Number of parallel API calls.

    Returns:
        List of embedding vectors, one per input text.
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    # Truncate extremely long texts
    truncated = [t[:_MAX_CHARS] for t in texts]

    # Split into batches, preserving original index
    batches: list[tuple[int, list[str]]] = []   # (start_index, texts)
    for start in range(0, len(truncated), batch_size):
        batches.append((start, truncated[start : start + batch_size]))

    results: dict[int, list[list[float]]] = {}  # start_index → embeddings

    def _call_api(start: int, batch: list[str]) -> tuple[int, list[list[float]]]:
        response = client.embeddings.create(input=batch, model=model)
        # Sort by index to guarantee order (API doesn't guarantee it)
        items = sorted(response.data, key=lambda x: x.index)
        return start, [item.embedding for item in items]

    logger.info(
        f"Generating embeddings for {len(texts)} texts in "
        f"{len(batches)} batches (workers={max_workers}) …"
    )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_call_api, s, b): s for s, b in batches}
        for future in as_completed(futures):
            start, vecs = future.result()
            results[start] = vecs
            logger.debug(f"  Batch starting at {start}: {len(vecs)} embeddings received.")

    # Reconstruct in original order
    embeddings: list[list[float]] = []
    for start in sorted(results):
        embeddings.extend(results[start])

    assert len(embeddings) == len(texts), (
        f"Embedding count mismatch: got {len(embeddings)}, expected {len(texts)}"
    )
    return embeddings
