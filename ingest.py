#!/usr/bin/env python3
"""
ingest.py
─────────
One-shot document ingestion pipeline for the PDF Q&A knowledge base.

Usage
─────
  # First-time or complete refresh (clears ALL data + blobs then re-ingests):
  python ingest.py --full-reload

  # Default incremental mode (skips PDFs whose SHA-256 hash hasn't changed):
  python ingest.py

  # Apply/verify schema then run incremental ingest:
  python ingest.py --setup-schema

  # Dry-run: show what would be processed without writing anything:
  python ingest.py --dry-run

  # Point at a different PDF folder:
  python ingest.py --source-dir /path/to/pdfs

Environment
───────────
  All credentials are read from .env (auto-loaded) or from actual env vars.
  See .env for the full list of config keys.

Pipeline per document
─────────────────────
  1. Compute SHA-256 of the PDF file.
  2. In incremental mode: skip if a documents row with the same filename
     AND the same hash already exists.
  3. Parse PDF → TextChunk list + PageImage list  (pdf_parser.py)
  4. Upload images to Vercel Blob                 (blob_store.py)
  5. Generate embeddings for all text chunks      (embeddings.py)
  6. Write everything to Neon in a single transaction:
       • Upsert documents row
       • Bulk-insert text_chunks  (COPY via psycopg2 execute_values)
       • Bulk-insert page_images
     Any previous rows for this doc_id are deleted first (CASCADE handles it).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

# ── Local modules ──────────────────────────────────────────────────────────────
from pdf_parser import ParsedDocument, PageImage, TextChunk, parse_pdf, compute_file_hash
from blob_store import build_pathname, delete_blobs, list_blobs, slugify, upload_image
from embeddings import generate_embeddings

# ─────────────────────────────────────────────────────────────────────────────

load_dotenv(Path(__file__).parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Config ──────────────────────────────────────────────────────────────────

def _cfg(key: str, default: str | None = None) -> str:
    val = os.environ.get(key, default)
    if val is None:
        raise RuntimeError(f"Missing required config: {key} (set in .env or environment)")
    return val


def get_config() -> dict:
    return {
        "database_url":        _cfg("DATABASE_URL"),
        "blob_token":          _cfg("BLOB_READ_WRITE_TOKEN"),
        "openai_api_key":      _cfg("OPENAI_API_KEY"),
        "pdf_source_dir":      _cfg("PDF_SOURCE_DIR", "../customer_data"),
        "chunk_size":          int(_cfg("CHUNK_SIZE", "2000")),
        "chunk_overlap":       int(_cfg("CHUNK_OVERLAP", "200")),
        "embedding_model":     _cfg("EMBEDDING_MODEL", "text-embedding-3-small"),
        "embedding_batch":     int(_cfg("EMBEDDING_BATCH_SIZE", "100")),
        "min_image_width":     int(_cfg("MIN_IMAGE_WIDTH", "80")),
        "min_image_height":    int(_cfg("MIN_IMAGE_HEIGHT", "80")),
        "blob_folder":         _cfg("BLOB_FOLDER", "pdf-images"),
    }


# ─── Schema management ───────────────────────────────────────────────────────

SCHEMA_SQL = Path(__file__).parent / "schema.sql"


def apply_schema(conn: psycopg2.extensions.connection, drop_first: bool = False) -> None:
    """Create or migrate tables / indexes from schema.sql."""
    sql = SCHEMA_SQL.read_text()
    with conn.cursor() as cur:
        if drop_first:
            cur.execute("""
                DROP TABLE IF EXISTS page_images  CASCADE;
                DROP TABLE IF EXISTS text_chunks  CASCADE;
                DROP TABLE IF EXISTS documents    CASCADE;
            """)
            logger.info("Dropped existing tables for full reload.")
        cur.execute(sql)
    conn.commit()
    logger.info("Schema applied / verified.")


def inspect_existing_schema(conn: psycopg2.extensions.connection) -> None:
    """Log a summary of what is currently in the database."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public' ORDER BY table_name
        """)
        tables = [r[0] for r in cur.fetchall()]
        logger.info(f"Existing tables: {tables}")
        for t in tables:
            cur.execute(f"SELECT COUNT(*) FROM {t}")
            count = cur.fetchone()[0]
            logger.info(f"  {t}: {count} rows")


# ─── Full-reload helpers ──────────────────────────────────────────────────────

def collect_blob_urls(conn: psycopg2.extensions.connection) -> list[str]:
    """Return all Vercel blob URLs currently stored in page_images."""
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT blob_url FROM page_images")
            return [r[0] for r in cur.fetchall()]
    except Exception:
        conn.rollback()
        return []


def purge_all(
    conn: psycopg2.extensions.connection,
    cfg: dict,
    dry_run: bool,
    blob_urls: list[str] | None = None,
) -> None:
    """
    Delete ALL data from the three tables and ALL blobs under BLOB_FOLDER.
    This is destructive and irreversible.

    Pass pre-collected blob_urls when the tables have already been dropped
    (e.g. after apply_schema drop_first=True).
    """
    if dry_run:
        logger.info("[DRY RUN] Would purge all DB rows and Vercel blobs.")
        return

    logger.info("─── Full-reload: purging existing data ───")

    # 1. Collect blob URLs from DB if not pre-collected
    if blob_urls is None:
        blob_urls = collect_blob_urls(conn)

    # 2. Delete DB rows (CASCADE handles child tables)
    with conn.cursor() as cur:
        cur.execute("DELETE FROM documents")
        deleted = cur.rowcount
    conn.commit()
    logger.info(f"  Deleted {deleted} document(s) from DB (cascaded to chunks + images).")

    # 3. Delete blobs
    if blob_urls:
        delete_blobs(blob_urls, cfg["blob_token"])
        logger.info(f"  Deleted {len(blob_urls)} blobs from Vercel.")
    else:
        logger.info("  No blobs to delete.")


# ─── Incremental helpers ──────────────────────────────────────────────────────

def get_processed_hashes(conn: psycopg2.extensions.connection) -> dict[str, str]:
    """Return {filename: file_hash} for every document already in the DB."""
    with conn.cursor() as cur:
        cur.execute("SELECT filename, file_hash FROM documents")
        return {r[0]: r[1] for r in cur.fetchall()}


def delete_document(conn: psycopg2.extensions.connection, filename: str, cfg: dict) -> None:
    """Remove a document (and all its chunks + images) from DB and Blob."""
    with conn.cursor() as cur:
        cur.execute("SELECT doc_id FROM documents WHERE filename = %s", (filename,))
        row = cur.fetchone()
        if not row:
            return
        doc_id = row[0]

        # Collect blob URLs first
        cur.execute("SELECT blob_url FROM page_images WHERE doc_id = %s", (doc_id,))
        blob_urls = [r[0] for r in cur.fetchall()]

        # Delete document (CASCADE removes chunks + images)
        cur.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))

    conn.commit()

    if blob_urls:
        delete_blobs(blob_urls, cfg["blob_token"])
        logger.info(f"  Deleted {len(blob_urls)} old blobs for '{filename}'.")


# ─── Core ingestion ───────────────────────────────────────────────────────────

def ingest_document(
    pdf_path: Path,
    conn: psycopg2.extensions.connection,
    cfg: dict,
    dry_run: bool,
) -> None:
    filename = pdf_path.name
    logger.info(f"┌── Processing: {filename}")
    t0 = time.time()

    # ── 1. Parse PDF ──────────────────────────────────────────────────────────
    logger.info(f"│   Parsing PDF …")
    parsed: ParsedDocument = parse_pdf(
        pdf_path,
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
        min_image_width=cfg["min_image_width"],
        min_image_height=cfg["min_image_height"],
    )
    logger.info(
        f"│   Pages: {parsed.total_pages} | "
        f"Chunks: {len(parsed.chunks)} | "
        f"Images: {len(parsed.images)}"
    )

    if dry_run:
        logger.info(f"│   [DRY RUN] Skipping DB write and blob upload.")
        logger.info(f"└── Done (dry run) in {time.time()-t0:.1f}s")
        return

    # ── 2. Upload images to Vercel Blob ───────────────────────────────────────
    doc_slug = slugify(filename)
    image_upload_results: list[dict] = []  # [{"blob_url": …, "blob_pathname": …}]

    if parsed.images:
        logger.info(f"│   Uploading {len(parsed.images)} images to Vercel Blob …")
        for img in parsed.images:
            pathname = build_pathname(
                cfg["blob_folder"], doc_slug, img.page_num, img.image_index, img.image_format
            )
            try:
                result = upload_image(img.data, pathname, cfg["blob_token"])
                image_upload_results.append({
                    "blob_url":      result["url"],
                    "blob_pathname": result.get("pathname", pathname),
                })
            except Exception as e:
                logger.warning(f"│   ⚠ Failed to upload image {pathname}: {e}")
                image_upload_results.append({
                    "blob_url":      "",
                    "blob_pathname": pathname,
                })
    else:
        logger.info(f"│   No images to upload.")

    # ── 3. Generate embeddings ────────────────────────────────────────────────
    logger.info(f"│   Generating {len(parsed.chunks)} embeddings …")
    chunk_texts = [c.text for c in parsed.chunks]
    embeddings = generate_embeddings(
        chunk_texts,
        api_key=cfg["openai_api_key"],
        model=cfg["embedding_model"],
        batch_size=cfg["embedding_batch"],
    )

    # ── 4. Write to database (single transaction) ─────────────────────────────
    logger.info(f"│   Writing to database …")

    with conn.cursor() as cur:
        # Upsert document record
        cur.execute("""
            INSERT INTO documents (filename, file_path, total_pages, file_hash, metadata)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (filename) DO UPDATE SET
                file_path    = EXCLUDED.file_path,
                total_pages  = EXCLUDED.total_pages,
                file_hash    = EXCLUDED.file_hash,
                processed_at = NOW(),
                metadata     = EXCLUDED.metadata
            RETURNING doc_id
        """, (
            parsed.filename,
            parsed.file_path,
            parsed.total_pages,
            parsed.file_hash,
            json.dumps({}),
        ))
        doc_id = cur.fetchone()[0]

        # Delete old chunks + images for this document (re-ingest case)
        cur.execute("DELETE FROM text_chunks WHERE doc_id = %s", (doc_id,))
        cur.execute("DELETE FROM page_images WHERE doc_id = %s", (doc_id,))

        # Bulk-insert text chunks
        if parsed.chunks:
            chunk_rows = [
                (
                    str(doc_id),
                    chunk.page_num,
                    chunk.chunk_index,
                    chunk.text,
                    embeddings[i],      # list[float] → psycopg2 converts to vector
                    chunk.section_heading,
                    json.dumps(chunk.bbox),
                    parsed.filename,
                    parsed.total_pages,
                    json.dumps(chunk.metadata),
                )
                for i, chunk in enumerate(parsed.chunks)
            ]
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO text_chunks
                    (doc_id, page_num, chunk_index, text, embedding,
                     section_heading, bbox, doc_filename, doc_total_pages, metadata)
                VALUES %s
                """,
                chunk_rows,
                template="(%s, %s, %s, %s, %s::vector, %s, %s::jsonb, %s, %s, %s::jsonb)",
                page_size=200,
            )

        # Bulk-insert page images
        if parsed.images:
            image_rows = []
            for i, img in enumerate(parsed.images):
                upload = image_upload_results[i] if i < len(image_upload_results) else {}
                if not upload.get("blob_url"):
                    continue   # skip images that failed to upload
                image_rows.append((
                    str(doc_id),
                    img.page_num,
                    img.image_index,
                    upload["blob_url"],
                    upload["blob_pathname"],
                    img.width,
                    img.height,
                    img.image_format,
                    json.dumps(img.bbox),
                    img.caption_text,
                    parsed.filename,
                    json.dumps(img.metadata),
                ))

            if image_rows:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO page_images
                        (doc_id, page_num, image_index, blob_url, blob_pathname,
                         width, height, image_format, bbox, caption_text,
                         doc_filename, metadata)
                    VALUES %s
                    """,
                    image_rows,
                    template=(
                        "(%s, %s, %s, %s, %s, %s, %s, %s, "
                        "%s::jsonb, %s, %s, %s::jsonb)"
                    ),
                    page_size=200,
                )

    conn.commit()
    elapsed = time.time() - t0
    logger.info(
        f"└── ✓ {filename} done in {elapsed:.1f}s "
        f"({len(parsed.chunks)} chunks, {len(image_upload_results)} images)"
    )


# ─── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="PDF → pgvector + Vercel Blob ingestion pipeline")
    parser.add_argument(
        "--full-reload",
        action="store_true",
        help="Delete ALL existing data and re-ingest every PDF from scratch.",
    )
    parser.add_argument(
        "--setup-schema",
        action="store_true",
        help="Apply schema.sql to the database before ingesting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse PDFs and log what would be done, without writing to DB or Blob.",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default=None,
        help="Override PDF_SOURCE_DIR from .env.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg = get_config()
    if args.source_dir:
        cfg["pdf_source_dir"] = args.source_dir

    source_dir = Path(cfg["pdf_source_dir"]).expanduser().resolve()
    if not source_dir.exists():
        logger.error(f"PDF source directory not found: {source_dir}")
        sys.exit(1)

    pdf_files = sorted(source_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {source_dir}")
        sys.exit(0)

    logger.info(f"Source directory : {source_dir}")
    logger.info(f"PDF files found  : {len(pdf_files)}")
    for p in pdf_files:
        logger.info(f"  {p.name}  ({p.stat().st_size / 1024:.0f} KB)")

    # ── Connect ───────────────────────────────────────────────────────────────
    logger.info("Connecting to database …")
    conn = psycopg2.connect(cfg["database_url"])
    conn.autocommit = False
    psycopg2.extras.register_uuid()

    # ── Schema ────────────────────────────────────────────────────────────────
    if args.setup_schema or args.full_reload:
        # Collect blob URLs before dropping tables so we can clean up Vercel
        existing_blobs = collect_blob_urls(conn) if args.full_reload else []
        apply_schema(conn, drop_first=args.full_reload)
    else:
        existing_blobs = []
        inspect_existing_schema(conn)

    # ── Full reload: purge everything ─────────────────────────────────────────
    if args.full_reload:
        purge_all(conn, cfg, args.dry_run, blob_urls=existing_blobs)
        files_to_process = pdf_files
    else:
        # ── Incremental: skip unchanged files ─────────────────────────────────
        processed = get_processed_hashes(conn)
        files_to_process = []
        skipped = []

        for pdf in pdf_files:
            current_hash = compute_file_hash(pdf)
            if pdf.name in processed and processed[pdf.name] == current_hash:
                skipped.append(pdf.name)
            else:
                if pdf.name in processed:
                    logger.info(f"  Changed: {pdf.name} (hash differs) → will re-ingest")
                    delete_document(conn, pdf.name, cfg)
                files_to_process.append(pdf)

        if skipped:
            logger.info(f"Skipping {len(skipped)} unchanged file(s): {skipped}")

    if not files_to_process:
        logger.info("Nothing to do — all files are up to date.")
        conn.close()
        return

    logger.info(f"\nIngesting {len(files_to_process)} file(s) …\n")

    # ── Process each PDF ──────────────────────────────────────────────────────
    success, failed = 0, 0
    for pdf_path in files_to_process:
        try:
            ingest_document(pdf_path, conn, cfg, args.dry_run)
            success += 1
        except Exception as e:
            logger.error(f"✗ Failed to process {pdf_path.name}: {e}", exc_info=True)
            conn.rollback()   # don't leave a partial transaction
            failed += 1

    conn.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n" + "─" * 60)
    logger.info(f"Ingestion complete.  Success: {success}  Failed: {failed}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
