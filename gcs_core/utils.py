# gcs-core/gcs_core/utils.py
# =============================================================================
# GenCyberSynth Core – Small, reliable file/time utilities
#
# What this module provides
# -------------------------
# 1) now_iso()                     → UTC timestamp as RFC 3339 / ISO-8601 (Z)
# 2) ensure_dir(path)              → Create a directory (parents=True), return Path
# 3) write_text(path, text, ...)   → Write or append UTF-8 text (atomic by default)
# 4) write_json(path, obj, ...)    → Pretty/compact JSON to file (atomic write)
# 5) read_json(path)               → Load JSON from file
# 6) write_json_line(jsonl, obj)   → Append one compact JSON object per line
# 7) iter_jsonl(jsonl)             → Iterate objects from a JSONL file lazily
#
# Design notes
# ------------
# - All writes are atomic (write to a temp file, fsync, then replace) except for
#   JSONL appends, which intentionally use append mode to support concurrent logs.
# - JSON encoding is robust to Path objects and (optionally) NumPy scalars/arrays
#   if NumPy is present; we never hard-require NumPy here.
# - Everything is type-hinted and documented; callers can rely on stable behavior.
# =============================================================================

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Generator, Iterable, Optional, Union
import datetime as _dt

__all__ = [
    "now_iso",
    "ensure_dir",
    "write_text",
    "write_json",
    "read_json",
    "write_json_line",
    "iter_jsonl",
]

# Compact JSON separators used for single-line and log-style writes
_JSON_COMPACT_SEPARATORS = (",", ":")


# =============================================================================
# Time
# =============================================================================
def now_iso() -> str:
    """
    Return a UTC timestamp in RFC 3339 / ISO-8601 extended format with a 'Z' suffix.

    Example
    -------
    >>> now_iso()  # '2025-03-04T12:34:56Z'
    """
    return _dt.datetime.now(tz=_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# =============================================================================
# Filesystem helpers
# =============================================================================
def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists (parents=True) and return it as a Path.

    Parameters
    ----------
    path : str | Path
        Directory path.

    Returns
    -------
    Path
        The created/existing directory.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _atomic_write_bytes(dst: Path, data: bytes) -> None:
    """
    Atomically write bytes to `dst` by writing into a temp file in the same directory
    and replacing it. Ensures data is flushed and fsync'd before replace().

    Notes
    -----
    - This is used by write_text() and write_json() for durability.
    - For JSONL appends, we intentionally use append mode instead (non-atomic)
      to allow incremental logging across processes.
    """
    dst = Path(dst)
    ensure_dir(dst.parent)
    # Create a NamedTemporaryFile in the same directory for atomic replace on all OSes.
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(dst.parent)) as tmp:
        tmp_path = Path(tmp.name)
        try:
            tmp.write(data)
            tmp.flush()
            os.fsync(tmp.fileno())
        except Exception:
            # Clean up on error
            try:
                tmp_path.unlink(missing_ok=True)  # type: ignore[arg-type]
            finally:
                raise
    # Atomic replace
    os.replace(str(tmp_path), str(dst))


def write_text(
    path: Union[str, Path],
    text: str,
    *,
    append: bool = False,
    ensure_trailing_newline: bool = False,
    atomic: bool = True,
    encoding: str = "utf-8",
) -> Path:
    """
    Write UTF-8 text to a file.

    Parameters
    ----------
    path : str | Path
        File path to write.
    text : str
        Content to write.
    append : bool, default False
        If True, append to the file (non-atomic). If False, overwrite (atomic by default).
    ensure_trailing_newline : bool, default False
        If True, ensures the file ends with exactly one newline.
    atomic : bool, default True
        If True (and append=False), write atomically via a temp file + replace.
    encoding : str, default 'utf-8'
        Text encoding.

    Returns
    -------
    Path
        The file path written.

    Notes
    -----
    - When append=True, the operation is not atomic by design (log-like behavior).
    """
    p = Path(path)
    if ensure_trailing_newline:
        if not text.endswith("\n"):
            text = text + "\n"

    if append:
        ensure_dir(p.parent)
        with p.open("a", encoding=encoding) as f:
            f.write(text)
        return p

    if atomic:
        _atomic_write_bytes(p, text.encode(encoding))
        return p

    # Non-atomic overwrite
    ensure_dir(p.parent)
    with p.open("w", encoding=encoding) as f:
        f.write(text)
    return p


# =============================================================================
# JSON helpers
# =============================================================================
def _json_default(obj: Any) -> Any:
    """
    Default converter for json.dumps to handle common non-JSON types.

    - pathlib.Path → str
    - NumPy scalars → native Python scalars
    - NumPy arrays  → lists

    We avoid importing NumPy globally; if unavailable, we simply skip those branches.
    """
    if isinstance(obj, Path):
        return str(obj)

    # Optional NumPy support without a hard dependency
    try:
        import numpy as _np  # type: ignore

        if isinstance(obj, _np.generic):  # scalar
            return obj.item()
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    except Exception:
        pass

    # Fallback: let json raise a TypeError (better than silently mangling)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json(
    path: Union[str, Path],
    obj: Any,
    *,
    indent: Optional[int] = 2,
    sort_keys: bool = True,
    ensure_ascii: bool = False,
    atomic: bool = True,
) -> Path:
    """
    Serialize `obj` to JSON and write to `path`.

    Parameters
    ----------
    path : str | Path
        Destination file path.
    obj : Any
        JSON-serializable object (supports Path and NumPy types via default handler).
    indent : int | None, default 2
        Indentation for pretty output. Use None for compact.
    sort_keys : bool, default True
        Sort object keys for stable diffs.
    ensure_ascii : bool, default False
        If True, escape non-ASCII chars; otherwise write UTF-8.
    atomic : bool, default True
        Write atomically via temp file + replace.

    Returns
    -------
    Path
        The file path written.
    """
    # Choose separators: pretty vs compact
    separators = None if indent is not None else _JSON_COMPACT_SEPARATORS
    data = json.dumps(
        obj,
        indent=indent,
        sort_keys=sort_keys,
        ensure_ascii=ensure_ascii,
        separators=separators,
        default=_json_default,
    )
    if atomic:
        _atomic_write_bytes(Path(path), data.encode("utf-8" if not ensure_ascii else "ascii"))
    else:
        write_text(path, data, append=False, atomic=False)
    return Path(path)


def read_json(path: Union[str, Path]) -> Any:
    """
    Read JSON from `path` and return the decoded Python object.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json_line(
    obj: Any,
    jsonl_path: Union[str, Path],
    *,
    sort_keys: bool = False,
    ensure_ascii: bool = False,
) -> None:
    """
    Append a single compact JSON object to a JSONL file.

    Parameters
    ----------
    obj : Any
        JSON-serializable object (supports Path and NumPy types via default handler).
    jsonl_path : str | Path
        Destination .jsonl file (created if missing).
    sort_keys : bool, default False
        Sort keys for readability; off by default for speed.
    ensure_ascii : bool, default False
        Escape non-ASCII if True; otherwise UTF-8.

    Notes
    -----
    - Appends in text mode with '\n' terminator.
    - Not atomic (by design) to support concurrent logging across processes.
    """
    line = json.dumps(
        obj,
        ensure_ascii=ensure_ascii,
        separators=_JSON_COMPACT_SEPARATORS,
        sort_keys=sort_keys,
        default=_json_default,
    )
    write_text(jsonl_path, line, append=True, ensure_trailing_newline=True, atomic=False)


def iter_jsonl(jsonl_path: Union[str, Path]) -> Generator[Any, None, None]:
    """
    Lazily iterate objects from a JSONL file. Skips blank lines.

    Parameters
    ----------
    jsonl_path : str | Path
        Path to a .jsonl file.

    Yields
    ------
    Any
        Decoded JSON object per non-blank line.

    Raises
    ------
    json.JSONDecodeError
        If a non-blank line contains malformed JSON.
    """
    p = Path(jsonl_path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield json.loads(s)
