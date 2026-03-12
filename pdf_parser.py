"""
pdf_parser.py
─────────────
Extracts structured text chunks and images from a PDF file using PyMuPDF.

Design decisions:
  • Text is chunked by paragraph boundaries first, then hard-capped at
    CHUNK_SIZE characters with CHUNK_OVERLAP to keep semantic units intact.
  • Section headings are detected heuristically: large/bold spans near the top
    of a text block are promoted to headings and carried forward into subsequent
    chunks on the same page.
  • Images below MIN_IMAGE_WIDTH × MIN_IMAGE_HEIGHT are silently dropped
    (icons, bullets, hairlines).
  • Caption text is the nearest text block within CAPTION_PROXIMITY pts of the
    image bounding box — usually a label printed immediately above/below.
  • All coordinates are in PDF user-space points (1 pt = 1/72 inch).
"""

from __future__ import annotations

import hashlib
import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

import fitz  # PyMuPDF


# ─── Dataclasses ─────────────────────────────────────────────────────────────

@dataclass
class TextChunk:
    page_num: int          # 1-based
    chunk_index: int       # order within page
    text: str
    section_heading: str | None
    bbox: dict             # {"x0": f, "y0": f, "x1": f, "y1": f}
    metadata: dict = field(default_factory=dict)


@dataclass
class PageImage:
    page_num: int          # 1-based
    image_index: int       # order within page
    data: bytes            # raw image bytes
    image_format: str      # "png" | "jpeg" | "webp" …
    width: int
    height: int
    bbox: dict             # position of the image on the page
    caption_text: str | None
    metadata: dict = field(default_factory=dict)


@dataclass
class ParsedDocument:
    filename: str
    file_path: str
    total_pages: int
    file_hash: str
    chunks: list[TextChunk]
    images: list[PageImage]


# ─── Constants ───────────────────────────────────────────────────────────────

CAPTION_PROXIMITY = 60   # pts — search for caption within this vertical range
HEADING_SIZE_RATIO = 1.2 # font size factor above page median to call it a heading


# ─── Public API ──────────────────────────────────────────────────────────────

def parse_pdf(
    pdf_path: str | Path,
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
    min_image_width: int = 80,
    min_image_height: int = 80,
) -> ParsedDocument:
    """Parse a PDF file and return all text chunks + images."""
    pdf_path = Path(pdf_path)
    raw_bytes = pdf_path.read_bytes()
    file_hash = hashlib.sha256(raw_bytes).hexdigest()

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)

    all_chunks: list[TextChunk] = []
    all_images: list[PageImage] = []

    for page_idx in range(total_pages):
        page = doc[page_idx]
        page_num = page_idx + 1

        page_chunks = _extract_text_chunks(page, page_num, chunk_size, chunk_overlap)
        all_chunks.extend(page_chunks)

        page_images = _extract_images(
            doc, page, page_num, min_image_width, min_image_height
        )
        all_images.extend(page_images)

    doc.close()

    return ParsedDocument(
        filename=pdf_path.name,
        file_path=str(pdf_path.resolve()),
        total_pages=total_pages,
        file_hash=file_hash,
        chunks=all_chunks,
        images=all_images,
    )


def compute_file_hash(pdf_path: str | Path) -> str:
    """Return SHA-256 hex digest of a file (used for incremental checks)."""
    return hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest()


# ─── Text extraction ─────────────────────────────────────────────────────────

def _extract_text_chunks(
    page: fitz.Page,
    page_num: int,
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextChunk]:
    """
    Extract text from a page as a list of chunks.

    Strategy:
      1. Use get_text("dict") to get individual text spans with font metadata.
      2. Reconstruct paragraph-level blocks preserving reading order.
      3. Detect section headings from font size / bold flags.
      4. Slide a window of `chunk_size` chars with `chunk_overlap` over the
         paragraph stream to produce final chunks.
    """
    blocks = page.get_text("dict", sort=True).get("blocks", [])

    # Collect all text spans with their font info
    paragraphs: list[dict] = []          # {"text": str, "bbox": tuple, "is_heading": bool}

    # Compute median font size on this page for heading detection
    all_sizes = []
    for blk in blocks:
        if blk.get("type") != 0:         # 0 = text block
            continue
        for line in blk.get("lines", []):
            for span in line.get("spans", []):
                if span.get("size"):
                    all_sizes.append(span["size"])

    median_size = _median(all_sizes) if all_sizes else 12.0

    for blk in blocks:
        if blk.get("type") != 0:
            continue
        block_text_parts = []
        block_bbox = blk["bbox"]         # (x0, y0, x1, y1)
        is_heading = False

        for line in blk.get("lines", []):
            line_parts = []
            for span in line.get("spans", []):
                txt = span.get("text", "").strip()
                if not txt:
                    continue
                line_parts.append(txt)
                # Heading heuristic: size significantly larger OR bold flag
                size = span.get("size", 0)
                flags = span.get("flags", 0)
                is_bold = bool(flags & 2**4)  # bit 4 = bold in PyMuPDF
                if size >= median_size * HEADING_SIZE_RATIO or (is_bold and size >= median_size):
                    is_heading = True
            if line_parts:
                block_text_parts.append(" ".join(line_parts))

        block_text = "\n".join(block_text_parts).strip()
        if block_text:
            paragraphs.append({
                "text": block_text,
                "bbox": block_bbox,
                "is_heading": is_heading,
            })

    if not paragraphs:
        return []

    # ── Slide a window over the paragraph stream ──────────────────────────────
    chunks: list[TextChunk] = []
    current_heading: str | None = None
    buffer = ""
    buffer_bbox: list[float] = []
    chunk_index = 0

    def flush(buf: str, bbox: list[float], heading: str | None) -> None:
        nonlocal chunk_index
        buf = buf.strip()
        if not buf:
            return
        b = _bbox_from_list(bbox) if bbox else {"x0": 0, "y0": 0, "x1": 0, "y1": 0}
        chunks.append(TextChunk(
            page_num=page_num,
            chunk_index=chunk_index,
            text=buf,
            section_heading=heading,
            bbox=b,
        ))
        chunk_index += 1

    for para in paragraphs:
        if para["is_heading"] and len(para["text"]) < 120:
            # Flush current buffer before starting a new section
            flush(buffer, buffer_bbox, current_heading)
            buffer = ""
            buffer_bbox = []
            current_heading = para["text"]

        candidate = (buffer + "\n\n" + para["text"]).strip() if buffer else para["text"]

        if len(candidate) <= chunk_size:
            buffer = candidate
            buffer_bbox = _merge_bbox(buffer_bbox, list(para["bbox"]))
        else:
            # Flush and start new chunk with overlap
            flush(buffer, buffer_bbox, current_heading)
            # Carry overlap from end of previous buffer
            overlap_text = buffer[-chunk_overlap:] if len(buffer) > chunk_overlap else buffer
            buffer = (overlap_text + "\n\n" + para["text"]).strip()
            buffer_bbox = list(para["bbox"])

    flush(buffer, buffer_bbox, current_heading)
    return chunks


# ─── Image extraction ─────────────────────────────────────────────────────────

def _extract_images(
    doc: fitz.Document,
    page: fitz.Page,
    page_num: int,
    min_width: int,
    min_height: int,
) -> list[PageImage]:
    """
    Extract all qualifying images from a page.

    For each image we:
      • Convert to PNG for uniform storage (handles JPEG, JBIG2, CCITT, etc.)
      • Record its bounding box on the page (via get_image_rects)
      • Find the nearest text block to use as a caption
    """
    images: list[PageImage] = []
    seen_xrefs: set[int] = set()
    image_index = 0

    # get_image_rects gives us (rect, transform) per (xref, refname)
    for img_info in page.get_images(full=True):
        xref = img_info[0]
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)

        # Get pixel dimensions before extracting (cheap check)
        width = img_info[2]
        height = img_info[3]
        if width < min_width or height < min_height:
            continue

        # Extract raw image and convert to PNG
        try:
            pix = fitz.Pixmap(doc, xref)
            if pix.n > 4:               # CMYK or other — convert to RGB
                pix = fitz.Pixmap(fitz.csRGB, pix)
            img_bytes = pix.tobytes("png")
            pix = None                  # release memory
        except Exception:
            continue

        # Bounding box of the image on the page
        rects = page.get_image_rects(xref)
        if rects:
            r = rects[0]
            bbox = {"x0": r.x0, "y0": r.y0, "x1": r.x1, "y1": r.y1}
        else:
            bbox = {"x0": 0, "y0": 0, "x1": float(width), "y1": float(height)}

        caption = _find_caption(page, bbox)

        images.append(PageImage(
            page_num=page_num,
            image_index=image_index,
            data=img_bytes,
            image_format="png",
            width=width,
            height=height,
            bbox=bbox,
            caption_text=caption,
        ))
        image_index += 1

    return images


def _find_caption(page: fitz.Page, img_bbox: dict) -> str | None:
    """
    Return the nearest text block within CAPTION_PROXIMITY pts of the image,
    prioritising text immediately below the image (most common caption placement).
    """
    blocks = page.get_text("blocks", sort=True)  # (x0,y0,x1,y1,text,block_no,block_type)
    img_y1 = img_bbox["y1"]
    img_y0 = img_bbox["y0"]
    img_cx = (img_bbox["x0"] + img_bbox["x1"]) / 2

    best_text: str | None = None
    best_dist = float("inf")

    for blk in blocks:
        if blk[6] != 0:                 # skip non-text blocks
            continue
        bx0, by0, bx1, by1, btext = blk[0], blk[1], blk[2], blk[3], blk[4]
        btext = btext.strip()
        if not btext or len(btext) > 300:
            continue

        # Vertical proximity: prefer blocks just below the image
        below_dist = by0 - img_y1     # positive if block is below image
        above_dist = img_y0 - by1     # positive if block is above image

        if 0 <= below_dist <= CAPTION_PROXIMITY:
            dist = below_dist
        elif 0 <= above_dist <= CAPTION_PROXIMITY:
            dist = above_dist + CAPTION_PROXIMITY  # slightly penalise above
        else:
            continue

        # Horizontal overlap check (caption should be roughly aligned)
        block_cx = (bx0 + bx1) / 2
        horiz_dist = abs(block_cx - img_cx)
        if horiz_dist > (img_bbox["x1"] - img_bbox["x0"]) * 1.5:
            continue

        total_dist = dist + horiz_dist * 0.1
        if total_dist < best_dist:
            best_dist = total_dist
            best_text = btext

    return best_text


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _median(values: list[float]) -> float:
    if not values:
        return 12.0
    s = sorted(values)
    n = len(s)
    return (s[n // 2] + s[(n - 1) // 2]) / 2


def _bbox_from_list(coords: list[float]) -> dict:
    if len(coords) == 4:
        return {"x0": coords[0], "y0": coords[1], "x1": coords[2], "y1": coords[3]}
    return {"x0": 0, "y0": 0, "x1": 0, "y1": 0}


def _merge_bbox(existing: list[float], new: list[float]) -> list[float]:
    """Expand a bounding box to include another."""
    if not existing:
        return new
    return [
        min(existing[0], new[0]),
        min(existing[1], new[1]),
        max(existing[2], new[2]),
        max(existing[3], new[3]),
    ]
