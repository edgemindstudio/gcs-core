# gcs-core/gcs_core/data.py
# =============================================================================
# Frozen-core data utilities for GenCyberSynth repos
#
# What this module provides
# -------------------------
# 1) Robust, dependency-light loaders for the standard USTC-TFC2016-style splits
#    saved as .npy files:
#       - train_data.npy / train_labels.npy
#       - test_data.npy  / test_labels.npy
#    The loader returns images in [0,1] as float32, shaped (N, H, W, C), and
#    labels as one-hot (N, K) float32. The provided test split is further
#    divided into validation and final test using a configurable fraction.
#
# 2) Small helpers:
#       - one_hot(...) without TensorFlow
#       - to_01_hwc(...) to normalize shapes/ranges
#       - dataset_counts(...) to build the "images" block expected by Phase-2
#       - load_splits(...) to expose path mapping when only paths are needed
#
# Design goals
# ------------
# - **No TensorFlow dependency:** keeps the core package lightweight.
# - **Safe defaults:** never mutates inputs; outputs are clipped to [0,1].
# - **Helpful errors:** missing files or bad shapes raise clear exceptions.
#
# Typical usage
# -------------
#     from pathlib import Path
#     from gcs_core.data import load_npy_splits, dataset_counts
#
#     xtr, ytr, xva, yva, xte, yte = load_npy_splits(
#         data_dir=Path("data/ustc"),
#         img_shape=(40, 40, 1),
#         num_classes=9,
#         val_fraction=0.5,
#     )
#     counts = dataset_counts(xtr, xva, xte)
#
# =============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import logging
import numpy as np

__all__ = [
    "load_npy_splits",
    "one_hot",
    "to_01_hwc",
    "dataset_counts",
    "load_splits",
]

_logger = logging.getLogger(__name__)
_logger.addHandler(logging.NullHandler())


# -----------------------------------------------------------------------------
# Core helpers
# -----------------------------------------------------------------------------
def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert class labels to one-hot (float32) without framework deps.

    Accepts:
      * shape (N,) integer class ids in [0, num_classes-1]
      * shape (N, K) that already looks one-hot (K must equal num_classes)

    Returns
    -------
    np.ndarray
        Shape (N, num_classes), dtype float32.
    """
    lab = np.asarray(labels)
    if lab.ndim == 2 and lab.shape[1] == num_classes:
        y = lab.astype(np.float32, copy=False)
        # Safety: clamp potential numeric noise to [0,1]
        np.clip(y, 0.0, 1.0, out=y)
        return y

    if lab.ndim != 1:
        raise ValueError(f"Labels must be 1-D ints or 2-D one-hot; got shape {lab.shape}.")

    if lab.size == 0:
        return np.zeros((0, num_classes), dtype=np.float32)

    if lab.min() < 0 or lab.max() >= num_classes:
        raise ValueError(
            f"Label ids out of range [0, {num_classes-1}] (min={lab.min()}, max={lab.max()})."
        )

    eye = np.eye(num_classes, dtype=np.float32)
    return eye[lab.astype(int)]


def to_01_hwc(x: np.ndarray, img_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Normalize images to float32 in [0,1] and reshape to (N, H, W, C).

    Accepted input ranges:
      * [0,255] -> divides by 255
      * [-1,1]  -> maps to [0,1] via (x + 1)/2
      * [0,1]   -> pass-through

    Parameters
    ----------
    x : np.ndarray
        Input images, any dtype. Can be (N,H,W) or already (N,H,W,C).
    img_shape : (H,W,C)
        Target shape for each image.

    Returns
    -------
    np.ndarray
        Float32 array, shape (N,H,W,C), clipped to [0,1].
    """
    H, W, C = img_shape
    arr = np.asarray(x, dtype=np.float32)

    # If input is (N, H, W), add channel
    if arr.ndim == 3:
        arr = arr[..., None]

    # Heuristic range mapping
    x_min, x_max = float(np.min(arr)), float(np.max(arr))
    if x_max > 1.5:
        arr = arr / 255.0
    elif x_min < 0.0:
        arr = (arr + 1.0) / 2.0

    # Final reshape + clamp
    arr = arr.reshape((-1, H, W, C))
    np.clip(arr, 0.0, 1.0, out=arr)
    return arr


# -----------------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------------
def load_npy_splits(
    *,
    data_dir: Path | str,
    img_shape: Tuple[int, int, int],
    num_classes: int,
    val_fraction: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load classification splits from .npy quartet and return:
        (x_train01, y_train1h, x_val01, y_val1h, x_test01, y_test1h)

    Expected files in `data_dir`:
        train_data.npy, train_labels.npy, test_data.npy, test_labels.npy

    Shapes / ranges:
      * Images are returned as float32 in [0,1], NHWC per `img_shape`.
      * Labels are returned as one-hot float32 with width `num_classes`.
      * The provided test set is split into (val, test) by `val_fraction`.

    Raises
    ------
    FileNotFoundError
        If any of the four required files are missing.
    ValueError
        If labels are out of range or shapes mismatch.

    Notes
    -----
    - No dequantization noise is added here; callers can add it if desired.
    - This function is intentionally small and framework-agnostic.
    """
    data_dir = Path(data_dir)
    H, W, C = img_shape

    req = {
        "train_data": data_dir / "train_data.npy",
        "train_labels": data_dir / "train_labels.npy",
        "test_data": data_dir / "test_data.npy",
        "test_labels": data_dir / "test_labels.npy",
    }
    missing = [k for k, p in req.items() if not p.exists()]
    if missing:
        pretty = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing required files in {data_dir}: {pretty}. "
            "Expected train/test data and labels as .npy files."
        )

    # Load
    x_train_raw = np.load(req["train_data"], allow_pickle=False)
    y_train_raw = np.load(req["train_labels"], allow_pickle=False)
    x_test_raw  = np.load(req["test_data"], allow_pickle=False)
    y_test_raw  = np.load(req["test_labels"], allow_pickle=False)

    # Normalize images & one-hot labels
    x_train01 = to_01_hwc(x_train_raw, img_shape)
    x_test01  = to_01_hwc(x_test_raw,  img_shape)
    y_train1h = one_hot(y_train_raw, num_classes)
    y_test1h  = one_hot(y_test_raw,  num_classes)

    # Split provided test -> (val, test)
    n_val = int(len(x_test01) * float(val_fraction))
    n_val = max(0, min(n_val, len(x_test01)))
    x_val01, y_val1h = x_test01[:n_val], y_test1h[:n_val]
    x_test01, y_test1h = x_test01[n_val:], y_test1h[n_val:]

    # Sanity checks
    if x_train01.shape[0] != y_train1h.shape[0]:
        raise ValueError(
            f"Train count mismatch: images={x_train01.shape[0]} vs labels={y_train1h.shape[0]}"
        )
    if x_val01.shape[0] != y_val1h.shape[0]:
        raise ValueError(
            f"Val count mismatch: images={x_val01.shape[0]} vs labels={y_val1h.shape[0]}"
        )
    if x_test01.shape[0] != y_test1h.shape[0]:
        raise ValueError(
            f"Test count mismatch: images={x_test01.shape[0]} vs labels={y_test1h.shape[0]}"
        )

    _logger.debug(
        "Loaded splits from %s | train=%d, val=%d, test=%d | HWC=%s",
        data_dir, len(x_train01), len(x_val01), len(x_test01), (H, W, C)
    )
    return x_train01, y_train1h, x_val01, y_val1h, x_test01, y_test1h


# -----------------------------------------------------------------------------
# Small conveniences
# -----------------------------------------------------------------------------
def dataset_counts(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    x_synth: np.ndarray | None = None,
) -> Dict[str, int | None]:
    """
    Build the "images" counts block used by Phase-2 summaries.

    Returns
    -------
    dict
        {
          "train_real": int,
          "val_real":   int,
          "test_real":  int,
          "synthetic":  Optional[int],
        }
    """
    return {
        "train_real": int(len(x_train)),
        "val_real":   int(len(x_val)),
        "test_real":  int(len(x_test)),
        "synthetic":  (int(len(x_synth)) if isinstance(x_synth, np.ndarray) else None),
    }


def load_splits(real_train_dir: str, val_dir: str, test_dir: str) -> Dict[str, str]:
    """
    Lightweight path mapper (kept for compatibility with earlier scaffolds).

    Parameters
    ----------
    real_train_dir : str
        Directory (or file path) for training data.
    val_dir : str
        Directory (or file path) for validation data.
    test_dir : str
        Directory (or file path) for test data.

    Returns
    -------
    dict
        {"train": "<path>", "val": "<path>", "test": "<path>"}
    """
    return {
        "train": str(Path(real_train_dir)),
        "val":   str(Path(val_dir)),
        "test":  str(Path(test_dir)),
    }
