"""
blob_store.py
─────────────
Thin wrapper around the Vercel Blob REST API for uploading and deleting
images extracted from PDFs.

Vercel Blob Python SDK (vercel-blob) wraps the same REST API that the
JS SDK uses.  We fall back to raw httpx calls if the SDK isn't available
so the module has no hard runtime dependency.

Pathname convention:
    {folder}/{doc_slug}/page-{page_num:04d}-img-{image_index:04d}.png

This lets us do prefix-based deletes when a document is re-ingested:
    delete_by_prefix(f"{folder}/{doc_slug}/")
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ─── Upload ───────────────────────────────────────────────────────────────────

def upload_image(
    image_bytes: bytes,
    pathname: str,
    token: str,
    content_type: str = "image/png",
) -> dict:
    """
    Upload image bytes to Vercel Blob and return the blob metadata dict.

    Returns:
        {
            "url": "https://…",
            "pathname": "pdf-images/…",
            "contentType": "image/png",
            "size": 12345,
            …
        }
    """
    try:
        import vercel_blob  # type: ignore
        result = vercel_blob.put(
            pathname,
            image_bytes,
            {"access": "public", "token": token, "contentType": content_type},
        )
        return result
    except ImportError:
        pass  # fall back to raw HTTP

    # ── Raw HTTP fallback (httpx) ─────────────────────────────────────────────
    import httpx

    resp = httpx.put(
        f"https://blob.vercel-storage.com/{pathname}",
        content=image_bytes,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": content_type,
            "x-api-version": "7",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def build_pathname(
    folder: str,
    doc_slug: str,
    page_num: int,
    image_index: int,
    ext: str = "png",
) -> str:
    """Return the canonical Vercel Blob pathname for a page image."""
    return f"{folder}/{doc_slug}/page-{page_num:04d}-img-{image_index:04d}.{ext}"


def slugify(filename: str) -> str:
    """Convert a PDF filename to a URL-safe slug used as the blob sub-folder."""
    stem = Path(filename).stem
    return (
        stem.lower()
        .replace(" ", "-")
        .replace("_", "-")
        .replace(",", "")
        .replace("(", "")
        .replace(")", "")
        .replace(".", "-")
    )


# ─── Delete ───────────────────────────────────────────────────────────────────

def delete_blobs(blob_urls: list[str], token: str) -> None:
    """
    Delete a list of Vercel Blob URLs in one API call (batch delete).
    Silently skips URLs that are not found (404).
    """
    if not blob_urls:
        return

    try:
        import vercel_blob  # type: ignore
        vercel_blob.delete(blob_urls, {"token": token})
        logger.info(f"Deleted {len(blob_urls)} blobs via SDK.")
        return
    except ImportError:
        pass

    # ── Raw HTTP fallback ─────────────────────────────────────────────────────
    import httpx, json

    resp = httpx.delete(
        "https://blob.vercel-storage.com",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "x-api-version": "7",
        },
        content=json.dumps({"urls": blob_urls}).encode(),
        timeout=30,
    )
    if resp.status_code not in (200, 404):
        resp.raise_for_status()
    logger.info(f"Deleted {len(blob_urls)} blobs via HTTP.")


def list_blobs(prefix: str, token: str) -> list[dict]:
    """
    List all blobs under a given pathname prefix.
    Returns a list of blob metadata dicts (each has at least 'url' and 'pathname').
    """
    try:
        import vercel_blob  # type: ignore
        result = vercel_blob.list({"prefix": prefix, "token": token})
        return result.get("blobs", [])
    except ImportError:
        pass

    import httpx
    from urllib.parse import urlencode

    blobs = []
    cursor = None

    while True:
        params = {"prefix": prefix, "limit": "1000"}
        if cursor:
            params["cursor"] = cursor
        url = "https://blob.vercel-storage.com?" + urlencode(params)
        resp = httpx.get(
            url,
            headers={"Authorization": f"Bearer {token}", "x-api-version": "7"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        blobs.extend(data.get("blobs", []))
        if not data.get("hasMore"):
            break
        cursor = data.get("cursor")

    return blobs
