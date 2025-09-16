# gcs-core/gcs_core/synth_loader.py
# =============================================================================
# Canonical synthetic data discovery & loading utilities (frozen core)
#
# What this module solves
# -----------------------
# Repos produce synthetic samples in slightly different layouts. This module
# provides a **single, stable** way to:
#   1) Resolve where the synthetic folder *should* be on disk.
#   2) Load synthetic arrays regardless of whether they were saved as:
#        • Combined arrays:   x_synth.npy + y_synth.npy
#        • Per-class arrays:  gen_class_<k>.npy + labels_class_<k>.npy
#        • Legacy layout:     class_<k>/*.npy (images only)
#
# Guarantees & conventions
# ------------------------
# • Returns `x_s, y_s` NumPy arrays.
# • `x_s` is returned exactly as saved (dtype promoted to float32). We do NOT
#   normalize or reshape here; downstream evaluators (e.g., val_common) handle
#   range/shape mapping uniformly.
# • `y_s` is returned as one-hot (float32, shape [N, K]) when possible. If the
#   source labels are already one-hot with the correct width, they are passed
#   through. If labels are missing (legacy layout), we synthesize them based on
#   the class folder/index.
#
# Typical usage
# -------------
#     from pathlib import Path
#     from gcs_core.synth_loader import resolve_synth_dir, load_synth_any
#
#     repo_root = Path(__file__).resolve().parents[1]
#     synth_dir = resolve_synth_dir(cfg, repo_root)
#     x_s, y_s  = load_synth_any(synth_dir, num_classes=9)
#
# Design notes
# ------------
# • No TensorFlow dependency here. Label one-hot is done with NumPy.
# • We raise FileNotFoundError when nothing usable is found; model CLIs should
#   catch and continue with REAL-only evaluation.
# • Paths from config are treated as-is when absolute; relative paths are
#   resolved against `repo_root`.
# =============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

import logging
import numpy as np

__all__ = ["resolve_synth_dir", "load_synth_any", "save_synth_monolithic"]

_logger = logging.getLogger(__name__)
_logger.addHandler(logging.NullHandler())

# -----------------------------------------------------------------------------#
# Internal helpers
# -----------------------------------------------------------------------------#


def _to_float32(x: np.ndarray) -> np.ndarray:
    """Ensure float32 dtype for arrays that represent images."""
    if x is None:
        return x
    return np.asarray(x, dtype=np.float32)


def _to_one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert labels to one-hot (float32).
    Accepts:
      • shape (N,) integer class ids
      • shape (N, K) one-hot (passes through if K==num_classes)
    """
    labels = np.asarray(labels)
    if labels.ndim == 2 and labels.shape[1] == num_classes:
        # already one-hot
        y = labels.astype(np.float32, copy=False)
        # optional safety: clip to {0,1}
        y[y < 0] = 0.0
        y[y > 1] = 1.0
        return y

    if labels.ndim != 1:
        raise ValueError(f"Labels must be 1-D ints or 2-D one-hot; got shape {labels.shape}.")

    if labels.size == 0:
        return np.zeros((0, num_classes), dtype=np.float32)

    if labels.min() < 0 or labels.max() >= num_classes:
        raise ValueError(
            f"Label ids out of range [0,{num_classes-1}]: min={labels.min()}, max={labels.max()}"
        )

    eye = np.eye(num_classes, dtype=np.float32)
    return eye[labels.astype(int)]


def _concat_or_none(chunks: Iterable[np.ndarray]) -> Optional[np.ndarray]:
    """Concatenate non-empty chunk list or return None if list is empty."""
    chunks = [np.asarray(c) for c in chunks if c is not None and np.asarray(c).size > 0]
    if not chunks:
        return None
    return np.concatenate(chunks, axis=0)


def _find_first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


# -----------------------------------------------------------------------------#
# Public API
# -----------------------------------------------------------------------------#


def resolve_synth_dir(cfg: dict, repo_root: Path) -> Path:
    """
    Resolve the directory where synthetic arrays are (or will be) stored.

    Resolution order:
      1) If cfg["ARTIFACTS"]["synthetic"] exists:
           - If absolute, return it.
           - If relative, return repo_root / that path.
      2) Otherwise, pick the first existing directory among common defaults:
           artifacts/synthetic
           artifacts/gan/synthetic
           artifacts/diffusion/synthetic
           artifacts/vae/synthetic
           artifacts/autoregressive/synthetic
           artifacts/gaussianmixture/synthetic
           artifacts/restrictedboltzmann/synthetic
           artifacts/maskedautoflow/synthetic
      3) Fallback to repo_root / "artifacts/synthetic" (may not exist yet).

    Parameters
    ----------
    cfg : dict
        The run configuration (parsed YAML). We look for cfg["ARTIFACTS"]["synthetic"].
    repo_root : Path
        Path to the repository root (e.g., Path(__file__).resolve().parents[1]).

    Returns
    -------
    Path
        Resolved path to a synthetic directory. The path may not exist yet.
    """
    try:
        art = cfg.get("ARTIFACTS", {}) if isinstance(cfg, dict) else {}
    except Exception:
        art = {}

    raw = art.get("synthetic")
    if isinstance(raw, str) and raw.strip():
        p = Path(raw)
        return p if p.is_absolute() else (repo_root / p)

    # Probe common defaults
    candidates = [
        repo_root / "artifacts/synthetic",
        repo_root / "artifacts/gan/synthetic",
        repo_root / "artifacts/diffusion/synthetic",
        repo_root / "artifacts/vae/synthetic",
        repo_root / "artifacts/autoregressive/synthetic",
        repo_root / "artifacts/gaussianmixture/synthetic",
        repo_root / "artifacts/restrictedboltzmann/synthetic",
        repo_root / "artifacts/maskedautoflow/synthetic",
    ]
    found = _find_first_existing(candidates)
    if found is not None:
        return found

    # Last resort: a sensible default under artifacts/
    return repo_root / "artifacts/synthetic"


def load_synth_any(
    synth_dir: Path | str,
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load synthetic arrays from a directory, supporting both "combined" and
    "per-class" layouts (with a legacy fallback).

    Layouts supported
    -----------------
    1) Combined (preferred):
         synth_dir/x_synth.npy   -> images (any numeric dtype)
         synth_dir/y_synth.npy   -> labels (int ids or one-hot)
    2) Per-class:
         synth_dir/gen_class_<k>.npy     -> images for class k
         synth_dir/labels_class_<k>.npy  -> labels for class k (optional; if
                                            missing, we synthesize [k, k, ...])
    3) Legacy:
         synth_dir/class_<k>/*.npy       -> images for class k (labels inferred)

    Returns
    -------
    (x_s, y_s) : Tuple[np.ndarray, np.ndarray]
        x_s : float32 array of images, concatenated across all classes.
        y_s : float32 one-hot array of shape (N, num_classes).

    Raises
    ------
    FileNotFoundError
        If no supported synthetic files are found under `synth_dir`.
    ValueError
        If label shapes are inconsistent or out of range.
    """
    synth_dir = Path(synth_dir)

    # ---------- 1) Combined ----------
    x_path = synth_dir / "x_synth.npy"
    y_path = synth_dir / "y_synth.npy"
    if x_path.exists() and y_path.exists():
        x = _to_float32(np.load(x_path, allow_pickle=False))
        y_raw = np.load(y_path, allow_pickle=False)
        try:
            y = _to_one_hot(y_raw, num_classes)
        except Exception as e:
            raise ValueError(
                f"Invalid labels in combined layout: {y_path.name} ({e})"
            ) from e
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"Sample mismatch between x_synth ({x.shape[0]}) and y_synth ({y.shape[0]})."
            )
        _logger.debug("Loaded combined synthetic arrays from %s", synth_dir)
        return x, y

    # ---------- 2) Per-class ----------
    xs, ys = [], []
    any_per_class = False
    for k in range(num_classes):
        gx = synth_dir / f"gen_class_{k}.npy"
        gy = synth_dir / f"labels_class_{k}.npy"
        if gx.exists():
            any_per_class = True
            xk = np.load(gx, allow_pickle=False)
            if xk.size > 0:
                xs.append(xk)
                if gy.exists():
                    yk_raw = np.load(gy, allow_pickle=False)
                    # Support a variety of shapes for labels:
                    #  - one number per sample (ints)
                    #  - one-hot rows
                    #  - a single scalar (apply to all rows)
                    yk_raw = np.asarray(yk_raw)
                    if yk_raw.ndim == 0:  # scalar class id
                        yk = np.full((len(xk),), int(yk_raw), dtype=int)
                    elif yk_raw.ndim == 1:
                        if len(yk_raw) != len(xk):
                            raise ValueError(
                                f"labels_class_{k}.npy length {len(yk_raw)} "
                                f"does not match gen_class_{k}.npy length {len(xk)}"
                            )
                        yk = yk_raw.astype(int, copy=False)
                    elif yk_raw.ndim == 2 and yk_raw.shape[0] == len(xk):
                        # already one-hot for this class; convert later with _to_one_hot pass-through
                        yk = yk_raw
                    else:
                        raise ValueError(
                            f"Unsupported label shape for labels_class_{k}.npy: {yk_raw.shape}"
                        )
                else:
                    # No labels file → synthesize integer ids for this class
                    yk = np.full((len(xk),), k, dtype=int)
                ys.append(yk)

    if any_per_class:
        X = _concat_or_none(xs)
        if X is None:
            raise FileNotFoundError(
                f"Per-class layout detected in {synth_dir}, but all gen_class_* files were empty."
            )
        # Concatenate & normalize labels
        Y_raw = _concat_or_none(ys)
        if Y_raw is None:
            raise FileNotFoundError(
                f"Per-class layout detected in {synth_dir}, but no labels were loadable."
            )
        Y = _to_one_hot(Y_raw, num_classes)
        X = _to_float32(X)
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"Sample mismatch after per-class concat: X={X.shape[0]} vs Y={Y.shape[0]}"
            )
        _logger.debug("Loaded per-class synthetic arrays from %s", synth_dir)
        return X, Y

    # ---------- 3) Legacy (class_<k>/*.npy) ----------
    xs_legacy, ys_legacy = [], []
    any_legacy = False
    for k in range(num_classes):
        d = synth_dir / f"class_{k}"
        if d.exists() and d.is_dir():
            files = sorted([p for p in d.glob("*.npy") if p.is_file()])
            if files:
                any_legacy = True
                imgs = [np.load(p, allow_pickle=False) for p in files]
                arr = np.stack(imgs, axis=0)
                xs_legacy.append(arr)
                ys_legacy.append(np.full((arr.shape[0],), k, dtype=int))

    if any_legacy:
        X = _concat_or_none(xs_legacy)
        Y = _concat_or_none(ys_legacy)
        if X is None or Y is None:
            raise FileNotFoundError(
                f"Legacy layout detected under {synth_dir}, but could not assemble non-empty arrays."
            )
        X = _to_float32(X)
        Y = _to_one_hot(Y, num_classes)
        _logger.debug("Loaded legacy-layout synthetic arrays from %s", synth_dir)
        return X, Y

    # Nothing found
    raise FileNotFoundError(
        f"No synthetic arrays found under {synth_dir}. "
        "Expected one of: x_synth.npy + y_synth.npy, "
        "gen_class_<k>.npy (+ optional labels_class_<k>.npy), or class_<k>/*.npy."
    )


def save_synth_monolithic(
    synth_dir: Path | str,
    x_s: np.ndarray,
    y_s: np.ndarray,
    *,
    overwrite: bool = True,
) -> Tuple[Path, Path]:
    """
    Save synthetic arrays in the canonical "combined" format:

        synth_dir/x_synth.npy   (float32)
        synth_dir/y_synth.npy   (int ids or one-hot; saved as-is)

    Parameters
    ----------
    synth_dir : Path | str
        Target directory to write into (created if needed).
    x_s : np.ndarray
        Image array (any numeric dtype). Will be saved as float32.
    y_s : np.ndarray
        Labels (int ids or one-hot). Saved as provided.
    overwrite : bool
        If False and files already exist, raises FileExistsError.

    Returns
    -------
    (x_path, y_path) : Tuple[Path, Path]
        Paths to the saved .npy files.
    """
    synth_dir = Path(synth_dir)
    synth_dir.mkdir(parents=True, exist_ok=True)

    x_path = synth_dir / "x_synth.npy"
    y_path = synth_dir / "y_synth.npy"

    if not overwrite:
        for p in (x_path, y_path):
            if p.exists():
                raise FileExistsError(f"Refusing to overwrite existing file: {p}")

    np.save(x_path, _to_float32(np.asarray(x_s)), allow_pickle=False)
    np.save(y_path, np.asarray(y_s), allow_pickle=False)

    _logger.info("Saved monolithic synthetic arrays: %s, %s", x_path, y_path)
    return x_path, y_path
