"""
core/security.py — Input Validation, Sanitization & Security Utilities.

Centralizes all defensive input handling so that no raw user data reaches
the processing pipeline without validation.
"""

import re
from pathlib import Path

from core.config import ALLOWED_EXTENSIONS, MAX_PDF_SIZE_MB, MAX_QUERY_LENGTH
from core.logger import get_logger

log = get_logger("security")


# ─── Exceptions ───────────────────────────────────────────────
class ValidationError(Exception):
    """Raised when user-supplied input fails validation."""


# ─── PDF Validation ───────────────────────────────────────────
def validate_pdf_upload(file_obj) -> None:
    """Validates an uploaded PDF for extension, size, and header integrity.

    Args:
        file_obj: A file-like object (e.g. Streamlit ``UploadedFile``).

    Raises:
        ValidationError: If any check fails.
    """
    filename = getattr(file_obj, "name", "")
    ext = Path(filename).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise ValidationError(
            f"Invalid file type '{ext}'. Only PDF files are accepted."
        )

    # Size check
    file_obj.seek(0, 2)
    size_mb = file_obj.tell() / (1024 * 1024)
    file_obj.seek(0)

    if size_mb == 0:
        raise ValidationError("Uploaded file is empty.")
    if size_mb > MAX_PDF_SIZE_MB:
        raise ValidationError(
            f"File too large ({size_mb:.1f} MB). Maximum allowed: {MAX_PDF_SIZE_MB} MB."
        )

    # PDF magic-byte header check
    header = file_obj.read(5)
    file_obj.seek(0)
    if header != b"%PDF-":
        raise ValidationError(
            "File does not appear to be a valid PDF (invalid header)."
        )

    log.info("PDF validated: %s (%.1f MB)", filename, size_mb)


# ─── Filename Sanitization ────────────────────────────────────
def sanitize_filename(filename: str) -> str:
    """Strips dangerous characters from a filename while keeping it readable."""
    sanitized = re.sub(r"[^\w\s\-.]", "_", filename).strip()
    return sanitized or "unnamed_document.pdf"


# ─── User Input Sanitization ─────────────────────────────────
def sanitize_user_input(text: str, max_length: int = MAX_QUERY_LENGTH) -> str:
    """Cleans user-supplied text by trimming length and removing control chars."""
    if not text or not isinstance(text, str):
        return ""
    text = text[:max_length].strip()
    # Remove null bytes and non-printable control characters (keep \\n, \\t)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text


# ─── HTML Sanitization ────────────────────────────────────────
def sanitize_html(text: str) -> str:
    """Escapes HTML special characters to prevent XSS in rendered content."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )

