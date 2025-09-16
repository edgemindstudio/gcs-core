# gcs-core/gcs_core/metrics.py
# =============================================================================
# GenCyberSynth Core – Generative Metrics
#
# This module provides a *standalone* implementation of the generative-quality
# metrics used across repos:
#
#   • FID (macro, pooled across classes)
#   • cFID (per-class FID and its macro average)
#   • JS / KL divergences (pixel histograms on the pooled subset)
#   • Diversity (mean feature-wise variance over synthetic images)
#
# Design goals
# ------------
# - **No hard dependency on GPU frameworks at import time.** TensorFlow/Keras
#   is imported lazily only when FID is computed. If InceptionV3 cannot be
#   loaded (e.g., offline env), FID/cFID return `None` gracefully.
# - **Robust file discovery.** The loader accepts several common file patterns
#   for validation and synthetic arrays (combined and per-class dumps).
# - **Stable preprocessing.** Inputs are mapped to float32 in [0,1] with NHWC
#   layout; grayscale is converted to RGB internally for Inception.
#
# Expected on-disk layouts
# ------------------------
# Validation directory (`val_dir`) — any of the following pairs:
#   • x_val.npy + y_val.npy
#   • val_data.npy + val_labels.npy
#   • test_data.npy + test_labels.npy
#   • x.npy + y.npy
#
# Synthetic directory (`synth_dir`) — one of:
#   • x_synth.npy + y_synth.npy                        (monolithic)
#   • gen_class_<k>.npy + labels_class_<k>.npy         (per-class, k=0..K-1)
#     (if labels_class_<k>.npy is missing, we infer label=k for all samples)
#
# Returned dictionary
# -------------------
# {
#   "fid_macro": Optional[float],
#   "cfid_macro": Optional[float],
#   "cfid_per_class": List[Optional[float]],
#   "js": Optional[float],
#   "kl": Optional[float],
#   "diversity": Optional[float]
# }
#
# =============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np

# =============================================================================
# Public API
# =============================================================================


def compute_generative_metrics(
    val_dir: str,
    synth_dir: str,
    fid_cap_per_class: int = 200,
    seed: int = 42,
) -> Dict[str, Optional[float] | List[Optional[float]]]:
    """
    Compute FID/cFID/JS/KL/Diversity between REAL validation and SYNTH images.

    Parameters
    ----------
    val_dir : str
        Directory containing validation arrays (see "Expected on-disk layouts").
    synth_dir : str
        Directory containing synthetic arrays (monolithic or per-class).
    fid_cap_per_class : int, default=200
        Maximum number of samples per class used for cFID/FID pooling.
    seed : int, default=42
        RNG seed for per-class subsampling.

    Returns
    -------
    dict
        See "Returned dictionary" section in the module docstring.

    Raises
    ------
    FileNotFoundError
        If no valid arrays can be found in either directory.
    ValueError
        If arrays cannot be coerced to 4D NHWC tensors with compatible sizes.
    """
    # ---- Load arrays (robust discovery) -------------------------------------
    x_val, y_val = _load_xy_any(Path(val_dir), is_val=True)
    x_syn, y_syn = _load_xy_any(Path(synth_dir), is_val=False)

    if x_val is None or y_val is None:
        raise FileNotFoundError(
            f"Could not locate validation arrays under {val_dir}. "
            "Expected x_val.npy/y_val.npy or val_data.npy/val_labels.npy (etc.)."
        )
    if x_syn is None or y_syn is None:
        raise FileNotFoundError(
            f"Could not locate synthetic arrays under {synth_dir}. "
            "Expected x_synth.npy/y_synth.npy or per-class gen_class_k.npy/labels_class_k.npy."
        )

    # ---- Standardize arrays -------------------------------------------------
    x_val = _to_01_nhwc(x_val)
    x_syn = _to_01_nhwc(x_syn)
    y_val_int = _onehot_to_int(y_val)
    y_syn_int = _onehot_to_int(y_syn)

    # ---- Build per-class pools (balanced + capped) --------------------------
    cfid_per_class: List[Optional[float]] = []
    pooled_val_idx: List[np.ndarray] = []
    pooled_syn_idx: List[np.ndarray] = []

    K = int(max(y_val_int.max(initial=0), y_syn_int.max(initial=0)) + 1)
    rng = np.random.default_rng(seed)

    for k in range(K):
        v_idx = np.where(y_val_int == k)[0]
        s_idx = np.where(y_syn_int == k)[0]
        if len(v_idx) == 0 or len(s_idx) == 0:
            cfid_per_class.append(None)
            continue

        n = min(len(v_idx), len(s_idx), int(fid_cap_per_class))
        if n <= 0:
            cfid_per_class.append(None)
            continue

        v_sel = rng.choice(v_idx, size=n, replace=False)
        s_sel = rng.choice(s_idx, size=n, replace=False)
        pooled_val_idx.append(v_sel)
        pooled_syn_idx.append(s_sel)

        # Per-class FID
        cfid_per_class.append(_fid_keras_safe(x_val[v_sel], x_syn[s_sel]))

    valid_cfid = [v for v in cfid_per_class if v is not None]
    cfid_macro = float(np.mean(valid_cfid)) if valid_cfid else None

    # ---- Pooled subset for macro FID + JS/KL + Diversity --------------------
    fid_macro = None
    js = kl = diversity = None
    if pooled_val_idx and pooled_syn_idx:
        pv = np.concatenate(pooled_val_idx, axis=0) if len(pooled_val_idx) > 1 else pooled_val_idx[0]
        ps = np.concatenate(pooled_syn_idx, axis=0) if len(pooled_syn_idx) > 1 else pooled_syn_idx[0]
        rv = x_val[pv]
        sv = x_syn[ps]

        fid_macro = _fid_keras_safe(rv, sv)
        js, kl = _js_kl_pixels(rv, sv)
        diversity = _diversity_score(sv)

    return {
        "fid_macro": fid_macro,
        "cfid_macro": cfid_macro,
        "cfid_per_class": cfid_per_class,
        "js": js,
        "kl": kl,
        "diversity": diversity,
    }


# =============================================================================
# Helpers – data loading / coercion
# =============================================================================


def _load_xy_any(root: Path, *, is_val: bool) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Discover and load (x, y) arrays from a directory.

    For validation:
      Tries (in order): (x_val, y_val), (val_data, val_labels),
                        (test_data, test_labels), (x, y)

    For synthetic:
      Tries monolithic: (x_synth, y_synth)
      Else per-class:   gen_class_<k> (+ optional labels_class_<k>)
                        Concatenates and constructs labels if missing.

    Returns
    -------
    (x, y) or (None, None) if nothing found.
    """
    root = Path(root)

    if is_val:
        cand = [
            ("x_val.npy", "y_val.npy"),
            ("val_data.npy", "val_labels.npy"),
            ("test_data.npy", "test_labels.npy"),
            ("x.npy", "y.npy"),
        ]
        for xn, yn in cand:
            x_p, y_p = root / xn, root / yn
            if x_p.exists() and y_p.exists():
                return np.load(x_p, allow_pickle=False), np.load(y_p, allow_pickle=False)

        return None, None

    # Synthetic: monolithic first
    xs, ys = root / "x_synth.npy", root / "y_synth.npy"
    if xs.exists() and ys.exists():
        return np.load(xs, allow_pickle=False), np.load(ys, allow_pickle=False)

    # Synthetic: per-class fallback
    # We accept either labels_class_<k>.npy or infer labels=k if missing.
    x_chunks, y_chunks = [], []
    # Discover all gen_class_<k>.npy
    for gen_file in sorted(root.glob("gen_class_*.npy")):
        try:
            k = int(gen_file.stem.split("_")[-1])
        except Exception:
            continue
        lbl_file = root / f"labels_class_{k}.npy"

        xk = np.load(gen_file, allow_pickle=False)
        if lbl_file.exists():
            yk = np.load(lbl_file, allow_pickle=False)
        else:
            # If labels missing, assume all samples are class k
            yk = np.full((len(xk),), k, dtype=np.int64)

        x_chunks.append(xk)
        y_chunks.append(yk)

    if x_chunks:
        x = np.concatenate(x_chunks, axis=0)
        y = np.concatenate(y_chunks, axis=0)
        return x, y

    return None, None


def _to_01_nhwc(x: np.ndarray) -> np.ndarray:
    """
    Coerce an image tensor to float32 in [0,1] with 4D NHWC layout.
    Accepts (N,H,W) or (N,H,W,C). If values look like [0,255], rescales.
    """
    x = np.asarray(x)
    if x.ndim == 3:  # (N,H,W) -> add channel
        x = x[..., None]
    if x.ndim != 4:
        raise ValueError(f"Expected 3D/4D array for images, got shape {x.shape}")

    x = x.astype(np.float32)
    if float(x.max(initial=0.0)) > 1.5:  # likely bytes
        x = x / 255.0
    return np.clip(x, 0.0, 1.0)


def _onehot_to_int(y: np.ndarray) -> np.ndarray:
    """
    Convert labels to integer class IDs.
    Accepts one-hot (N,K) or integer vector (N,).
    """
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] > 1:
        return y.argmax(axis=1).astype(np.int64)
    return y.reshape(-1).astype(np.int64)


# =============================================================================
# Helpers – metrics implementations
# =============================================================================


def _fid_keras_safe(real_01: np.ndarray, fake_01: np.ndarray) -> Optional[float]:
    """
    Compute standard FID in InceptionV3 (ImageNet) feature space.

    Returns None if TensorFlow/Keras or weights are unavailable.

    Inputs
    ------
    real_01, fake_01 : float32 arrays in [0,1], shape (N,H,W,1 or 3)
    """
    try:
        import tensorflow as tf  # local import to keep package import light
        from tensorflow.keras.applications import InceptionV3
        from tensorflow.keras.applications.inception_v3 import preprocess_input
    except Exception:
        return None

    # Ensure 3 channels and 299×299
    def _prep(x: np.ndarray) -> np.ndarray:
        x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
        if x_tf.shape[-1] == 1:
            x_tf = tf.image.grayscale_to_rgb(x_tf)
        x_tf = tf.image.resize(x_tf, (299, 299))
        # Inception expects 0..255 before preprocess_input
        x_tf = preprocess_input(x_tf * 255.0)
        return x_tf

    try:
        inc = InceptionV3(include_top=False, pooling="avg", input_shape=(299, 299, 3))
    except Exception:
        return None

    a = inc.predict(_prep(real_01), verbose=0)
    b = inc.predict(_prep(fake_01), verbose=0)
    return _fid_from_features(a, b)


def _fid_from_features(a: np.ndarray, b: np.ndarray) -> float:
    """
    Fréchet distance between two Gaussians N(m1,C1) and N(m2,C2) fit to features.
    """
    m1, C1 = a.mean(axis=0), np.cov(a, rowvar=False)
    m2, C2 = b.mean(axis=0), np.cov(b, rowvar=False)
    diff = m1 - m2

    # Stable sqrtm via eigendecomposition (avoids SciPy dependency)
    def _sqrtm_psd(M: np.ndarray) -> np.ndarray:
        vals, vecs = np.linalg.eigh(M)
        vals = np.clip(vals, 0.0, None)
        return (vecs * np.sqrt(vals)) @ vecs.T

    C12 = _sqrtm_psd(C1 @ C2)
    fid = float(diff @ diff + np.trace(C1 + C2 - 2.0 * C12))
    return fid


def _js_kl_pixels(real01: np.ndarray, fake01: np.ndarray, bins: int = 256, eps: float = 1e-8) -> Tuple[Optional[float], Optional[float]]:
    """
    JS and KL divergences between pixel-intensity histograms in [0,1].
    """
    try:
        pr, _ = np.histogram(real01.ravel(), bins=bins, range=(0.0, 1.0), density=True)
        pf, _ = np.histogram(fake01.ravel(), bins=bins, range=(0.0, 1.0), density=True)
        pr = pr.astype(np.float64) + eps
        pf = pf.astype(np.float64) + eps
        pr /= pr.sum()
        pf /= pf.sum()
        m = 0.5 * (pr + pf)

        def _kl(p: np.ndarray, q: np.ndarray) -> float:
            return float(np.sum(p * (np.log(p) - np.log(q))))

        js = 0.5 * (_kl(pr, m) + _kl(pf, m))
        kl = _kl(pr, pf)
        return js, kl
    except Exception:
        return None, None


def _diversity_score(x01: np.ndarray) -> Optional[float]:
    """
    Diversity: mean variance across flattened synthetic images (higher = more varied).
    """
    try:
        flat = x01.reshape((x01.shape[0], -1)).astype(np.float32)
        return float(np.mean(np.var(flat, axis=0)))
    except Exception:
        return None
