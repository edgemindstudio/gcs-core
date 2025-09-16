# ===== eval/val_common.py =====
"""
Unified evaluation utilities for comparing generative models on cybersecurity image data.

What this module provides
-------------------------
1) compute_all_metrics(...): returns FID/cFID/JS/KL/Diversity + REAL-only and REAL+SYNTH
   utility metrics using a small, consistent CNN evaluator.

2) evaluate_model_suite(...): a wrapper used by repo CLIs that:
   - calls compute_all_metrics(...)
   - emits a standardized summary dict with keys your aggregator expects:
       {
         "model": "...",
         "seed": 42,
         "images": {train_real, val_real, test_real, synthetic},
         "generative": {"fid", "fid_macro", "cfid_macro", "js", "kl", "diversity"},
         "utility_real_only": {...},
         "utility_real_plus_synth": {...},
         "deltas_RS_minus_R": {...}
       }

Notes
-----
- Images are handled in [0,1] float32 NHWC. We auto-map from [0,255] and [-1,1].
- FID uses Keras InceptionV3 (ImageNet). If the weights aren’t available and can’t be
  downloaded, generative metrics are set to None (no crash).
"""

from __future__ import annotations
from typing import Callable, Dict, Optional, Tuple, List

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    average_precision_score, roc_curve, precision_recall_fscore_support
)
from sklearn.utils.class_weight import compute_class_weight
from scipy.linalg import sqrtm
from scipy.stats import entropy
from tensorflow.keras import layers, models, optimizers


# ======================================================================================
# 0) Repro / basic utilities
# ======================================================================================

def set_global_seed(seed: int = 42) -> None:
    """Set seeds for reproducibility (NumPy + TensorFlow)."""
    np.random.seed(seed)
    try:
        tf.random.set_seed(seed)
    except Exception:
        pass


def to_01_hwc(x: np.ndarray, img_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Ensure images are float32 in [0,1] and shaped (N, H, W, C).

    Accepts input ranges:
      * [0,255] -> divides by 255
      * [-1,1]  -> maps to [0,1]
      * [0,1]   -> pass-through
    Accepts (N,H,W) and adds a channel dimension.
    """
    x = np.asarray(x)
    if x.ndim == 3:
        x = x[..., None]
    x = x.astype(np.float32)

    x_min, x_max = float(x.min()), float(x.max())
    if x_max > 1.5:
        x = x / 255.0
    elif x_min < 0.0:
        x = (x + 1.0) / 2.0

    H, W, C = img_shape
    x = x.reshape((-1, H, W, C))
    return np.clip(x, 0.0, 1.0)


def onehot_to_int(y: np.ndarray) -> np.ndarray:
    """Convert one-hot labels to integer class IDs; pass-through if already integers."""
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] > 1:
        return y.argmax(axis=1)
    return y.astype(int)


# ======================================================================================
# 1) Generative metrics: FID / cFID / JS / KL / Diversity (+ optional domain-FID)
# ======================================================================================

_inception_model: Optional[tf.keras.Model] = None
_inception_ok = True  # flip to False if we fail to init (prevents repeated errors)

def _get_inception() -> Optional[tf.keras.Model]:
    """Singleton Keras InceptionV3 (avg pooled) used for standard FID."""
    global _inception_model, _inception_ok
    if not _inception_ok:
        return None
    if _inception_model is None:
        try:
            _inception_model = tf.keras.applications.InceptionV3(
                include_top=False, pooling="avg", input_shape=(299, 299, 3)
            )
        except Exception:
            # If weights cannot be loaded/downloaded, disable generative metrics gracefully
            _inception_ok = False
            _inception_model = None
    return _inception_model


def _fid_from_activations(a: np.ndarray, b: np.ndarray) -> float:
    """Fréchet distance between two Gaussian approximations (a,b) of features."""
    mu1, sig1 = a.mean(axis=0), np.cov(a, rowvar=False)
    mu2, sig2 = b.mean(axis=0), np.cov(b, rowvar=False)
    diff = mu1 - mu2
    covmean = sqrtm(sig1 @ sig2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sig1 + sig2 - 2.0 * covmean))


def fid_keras(real_01: np.ndarray, fake_01: np.ndarray) -> Optional[float]:
    """
    Standard FID in ImageNet InceptionV3 feature space.

    Inputs:
        real_01, fake_01: float32 in [0,1], shape (N,H,W,1)
    """
    inc = _get_inception()
    if inc is None:
        return None
    real = tf.image.resize(real_01, (299, 299))
    fake = tf.image.resize(fake_01, (299, 299))
    real = tf.image.grayscale_to_rgb(real)
    fake = tf.image.grayscale_to_rgb(fake)
    # real = tf.keras.applications.inception_v3.preprocess_input(real)
    # fake = tf.keras.applications.inception_v3.preprocess_input(fake)

    # scale 0..1 -> 0..255 before preprocess_input
    real = tf.keras.applications.inception_v3.preprocess_input(real * 255.0)
    fake = tf.keras.applications.inception_v3.preprocess_input(fake * 255.0)

    a = inc.predict(real, verbose=0)
    b = inc.predict(fake, verbose=0)
    return _fid_from_activations(a, b)


def fid_in_domain_embedding(
    real_01: np.ndarray, fake_01: np.ndarray, embed_fn: Callable[[np.ndarray], np.ndarray]
) -> Optional[float]:
    """
    Optional domain-FID using a custom embedding.
    `embed_fn` must accept (N,H,W,1) images in [0,1] and return features (N,D).
    """
    try:
        fr = embed_fn(real_01)
        ff = embed_fn(fake_01)
        return _fid_from_activations(fr, ff)
    except Exception:
        return None


def js_kl_on_pixels(real01: np.ndarray, fake01: np.ndarray, bins: int = 256, eps: float = 1e-8) -> Tuple[Optional[float], Optional[float]]:
    """Compute JS and KL divergences on pixel histograms (same subset used for FID)."""
    try:
        pr, _ = np.histogram(real01.ravel(), bins=bins, range=(0, 1), density=True)
        pf, _ = np.histogram(fake01.ravel(), bins=bins, range=(0, 1), density=True)
        pr = pr + eps
        pf = pf + eps
        m = 0.5 * (pr + pf)
        js = 0.5 * (entropy(pr, m) + entropy(pf, m))
        kl = entropy(pr, pf)
        return float(js), float(kl)
    except Exception:
        return None, None


def diversity_score(x01: np.ndarray) -> Optional[float]:
    """Mean feature-wise variance across synthetic images (higher = more varied)."""
    try:
        flat = x01.reshape((x01.shape[0], -1))
        return float(np.mean(np.var(flat, axis=0)))
    except Exception:
        return None


# ======================================================================================
# 2) Evaluator CNN + utility metrics
# ======================================================================================

def _build_eval_cnn(img_shape: tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    """
    Tiny evaluator CNN used for downstream utility metrics.
    This version is robust to very small inputs (>=1x1):
      • Convs use 'same' padding (no spatial shrinkage).
      • MaxPool is applied only if both H and W are >= 2.
      • GlobalAveragePooling2D avoids any fixed spatial assumptions.
    """
    inp = layers.Input(shape=img_shape, name="x")

    def maybe_pool(x):
        # Static shapes preferred; if unknown, be conservative and pool.
        h, w = x.shape[1], x.shape[2]
        if (h is not None and h < 2) or (w is not None and w < 2):
            return x  # too small to pool
        return layers.MaxPooling2D(pool_size=2, strides=2, padding="valid")(x)

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = maybe_pool(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = maybe_pool(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = maybe_pool(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.1)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inp, out, name="EvalCNN")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _macro_auprc(y_true_int: np.ndarray, proba: np.ndarray, K: int) -> float:
    """Macro AUPRC computed one-vs-rest."""
    y_oh = np.eye(K)[y_true_int]
    aps = []
    for k in range(K):
        aps.append(average_precision_score(y_oh[:, k], proba[:, k]))
    return float(np.mean(aps))


def _recall_at_fpr(y_true_int: np.ndarray, proba: np.ndarray, K: int, target_fpr: float = 0.01) -> float:
    """Macro avg recall at a given false positive rate (1% by default), one-vs-rest."""
    y_oh = np.eye(K)[y_true_int]
    recalls = []
    for k in range(K):
        try:
            fpr, tpr, _ = roc_curve(y_oh[:, k], proba[:, k])
            idx = np.where(fpr <= target_fpr)[0]
            recalls.append(0.0 if len(idx) == 0 else float(np.max(tpr[idx])))
        except Exception:
            recalls.append(0.0)
    return float(np.mean(recalls))


def _ece(proba: np.ndarray, y_true_int: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error using max-probability binning."""
    conf = proba.max(axis=1)
    preds = proba.argmax(axis=1)
    correct = (preds == y_true_int).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(conf)
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() == 0:
            continue
        ece += abs(correct[m].mean() - conf[m].mean()) * (m.sum() / N)
    return float(ece)


def _brier_multiclass(y_true_oh: np.ndarray, proba: np.ndarray) -> float:
    """Multiclass Brier score: mean over samples of sum_k (p_k - y_k)^2."""
    y_true_oh = y_true_oh.astype(np.float32)
    proba = proba.astype(np.float32)
    return float(np.mean(np.sum((proba - y_true_oh) ** 2, axis=1)))


def _fit_eval_cnn(
    x_train: np.ndarray, y_train_oh: np.ndarray,
    x_val: np.ndarray,   y_val_oh: np.ndarray,
    input_shape: Tuple[int, int, int],
    num_classes: int,
    class_weight: Optional[Dict[int, float]] = None,
    seed: int = 42,
    epochs: int = 20,
) -> tf.keras.Model:
    """Train the evaluator with EarlyStopping; class_weight optional."""
    set_global_seed(seed)
    m = _build_eval_cnn(input_shape, num_classes)
    cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
    m.fit(
        x_train, y_train_oh,
        validation_data=(x_val, y_val_oh),
        epochs=epochs,
        batch_size=128,
        verbose=0,
        callbacks=cb,
        class_weight=class_weight
    )
    return m


def _eval_on_test(model: tf.keras.Model, x_test: np.ndarray, y_test_oh: np.ndarray) -> Dict:
    """Compute global & per-class metrics on REAL test data."""
    y_true = onehot_to_int(y_test_oh)
    proba = model.predict(x_test, verbose=0)
    y_pred = proba.argmax(axis=1)
    K = y_test_oh.shape[1]

    out = {
        "accuracy":            float(accuracy_score(y_true, y_pred)),
        "macro_f1":            float(f1_score(y_true, y_pred, average="macro")),
        "bal_acc":             float(balanced_accuracy_score(y_true, y_pred)),
        "macro_auprc":         _macro_auprc(y_true, proba, K),
        "recall_at_1pct_fpr":  _recall_at_fpr(y_true, proba, K, target_fpr=0.01),
        "ece":                 _ece(proba, y_true, n_bins=15),
        "brier":               _brier_multiclass(np.eye(K)[y_true], proba),
    }

    prec, rec, f1c, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(K)), average=None, zero_division=0
    )
    out["per_class"] = {
        "precision": prec.astype(float).tolist(),
        "recall":    rec.astype(float).tolist(),
        "f1":        f1c.astype(float).tolist(),
        "support":   sup.astype(int).tolist(),
    }
    return out


# ======================================================================================
# 3) High-level evaluator: compute ALL metrics (core)
# ======================================================================================

def compute_all_metrics(
    *,
    img_shape: Tuple[int, int, int],
    x_train_real: np.ndarray, y_train_real: np.ndarray,
    x_val_real:   np.ndarray, y_val_real:   np.ndarray,
    x_test_real:  np.ndarray, y_test_real:  np.ndarray,
    x_synth: Optional[np.ndarray] = None,
    y_synth: Optional[np.ndarray] = None,
    fid_cap_per_class: int = 200,
    seed: int = 42,
    domain_embed_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    epochs: int = 20,
) -> Dict:
    """
    Compute the complete suite of metrics required by the study.

    Returns a dict with keys:
      fid_macro, cfid_macro, js, kl, diversity,
      real_only{...}, real_plus_synth{...}
    """
    set_global_seed(seed)
    H, W, C = img_shape

    # --- Standardize arrays ---
    xr = to_01_hwc(x_train_real, img_shape)
    xv = to_01_hwc(x_val_real,   img_shape)
    xt = to_01_hwc(x_test_real,  img_shape)

    if y_train_real.ndim == 1:
        K = int(np.max(y_train_real) + 1)
        yr = np.eye(K)[y_train_real.astype(int)]
        yv = np.eye(K)[y_val_real.astype(int)]
        yt = np.eye(K)[y_test_real.astype(int)]
    else:
        K = y_train_real.shape[1]
        yr, yv, yt = y_train_real, y_val_real, y_test_real

    has_synth = (x_synth is not None) and (y_synth is not None)
    if has_synth:
        xs = to_01_hwc(x_synth, img_shape)
        ys = y_synth if y_synth.ndim == 2 else np.eye(K)[y_synth.astype(int)]
    else:
        xs, ys = None, None

    # --- Generative quality (balanced per-class VAL vs SYNTH) ---
    fid_macro: Optional[float] = None
    cfid_per_class: Optional[List[Optional[float]]] = None
    cfid_macro: Optional[float] = None
    js: Optional[float] = None
    kl: Optional[float] = None
    diversity: Optional[float] = None
    fid_domain: Optional[float] = None

    if has_synth:
        yv_int = onehot_to_int(yv)
        ys_int = onehot_to_int(ys)
        per_class: List[Optional[float]] = []
        idx_val_cat: List[np.ndarray] = []
        idx_syn_cat: List[np.ndarray] = []

        for k in range(K):
            v_idx = np.where(yv_int == k)[0]
            s_idx = np.where(ys_int == k)[0]
            if len(v_idx) == 0 or len(s_idx) == 0:
                per_class.append(None)
                continue
            n = min(len(v_idx), len(s_idx), int(fid_cap_per_class))
            if n <= 0:
                per_class.append(None)
                continue
            rng = np.random.default_rng(seed + k)
            v_sel = rng.choice(v_idx, size=n, replace=False)
            s_sel = rng.choice(s_idx, size=n, replace=False)
            idx_val_cat.append(v_sel)
            idx_syn_cat.append(s_sel)
            per_class.append(fid_keras(xv[v_sel], xs[s_sel]))

        valid = [v for v in per_class if v is not None]
        cfid_per_class = per_class
        cfid_macro = float(np.mean(valid)) if len(valid) > 0 else None

        if len(idx_val_cat) > 0:
            idx_val_all = np.concatenate(idx_val_cat) if len(idx_val_cat) > 1 else idx_val_cat[0]
            idx_syn_all = np.concatenate(idx_syn_cat) if len(idx_syn_cat) > 1 else idx_syn_cat[0]
            rv = xv[idx_val_all]
            sv = xs[idx_syn_all]

            fid_macro = fid_keras(rv, sv)
            js, kl = js_kl_on_pixels(rv, sv)
            diversity = diversity_score(sv)

            if domain_embed_fn is not None:
                fid_domain = fid_in_domain_embedding(rv, sv, domain_embed_fn)

    # --- Downstream utility on REAL test (small CNN) ---
    ytr_int = onehot_to_int(yr)
    cls_w_arr = compute_class_weight(class_weight="balanced", classes=np.arange(K), y=ytr_int)
    class_weight = {i: float(w) for i, w in enumerate(cls_w_arr)}

    # A) Train on REAL only (with class weights)
    clf_R = _fit_eval_cnn(
        xr, yr, xv, yv, input_shape=(H, W, C), num_classes=K,
        class_weight=class_weight, seed=seed, epochs=epochs
    )
    util_R = _eval_on_test(clf_R, xt, yt)

    # B) Train on REAL + SYNTH (if provided), no class weights
    if has_synth:
        x_rs = np.concatenate([xr, xs], axis=0)
        y_rs = np.concatenate([yr, ys], axis=0)
        clf_RS = _fit_eval_cnn(
            x_rs, y_rs, xv, yv, input_shape=(H, W, C), num_classes=K,
            class_weight=None, seed=seed, epochs=epochs
        )
        util_RS = _eval_on_test(clf_RS, xt, yt)
    else:
        util_RS = {
            "accuracy": None, "macro_f1": None, "bal_acc": None,
            "macro_auprc": None, "recall_at_1pct_fpr": None,
            "ece": None, "brier": None, "per_class": None
        }

    return {
        "fid_macro": fid_macro,
        "cfid_macro": cfid_macro,
        "cfid_per_class": cfid_per_class,   # optional extra detail
        "js": js,
        "kl": kl,
        "diversity": diversity,
        "fid_domain": fid_domain,           # optional extra detail
        "real_only": util_R,
        "real_plus_synth": util_RS,
    }


# ======================================================================================
# 4) Wrapper used by repo CLIs: emits standardized summary for the aggregator
# ======================================================================================

def _map_util_names(util_block: Dict) -> Dict:
    """Normalize utility metric key names to aggregator-expected labels."""
    if util_block is None:
        return {}
    return {
        "accuracy":               util_block.get("accuracy"),
        "macro_f1":               util_block.get("macro_f1"),
        "balanced_accuracy":      util_block.get("bal_acc"),            # rename
        "macro_auprc":            util_block.get("macro_auprc"),
        "recall_at_1pct_fpr":     util_block.get("recall_at_1pct_fpr"),
        "ece":                    util_block.get("ece"),
        "brier":                  util_block.get("brier"),
        "per_class":              util_block.get("per_class"),
    }


def evaluate_model_suite(
    *,
    model_name: str,
    img_shape: Tuple[int, int, int],
    x_train_real: np.ndarray, y_train_real: np.ndarray,
    x_val_real:   np.ndarray, y_val_real:   np.ndarray,
    x_test_real:  np.ndarray, y_test_real:  np.ndarray,
    x_synth: Optional[np.ndarray] = None,
    y_synth: Optional[np.ndarray] = None,
    per_class_cap_for_fid: int = 200,
    seed: int = 42,
    epochs: int = 20,
) -> Dict:
    """
    Compute metrics and assemble a standardized summary dict that the repo CLIs
    can json.dump(...) directly. This matches what your aggregator expects.
    """
    metrics = compute_all_metrics(
        img_shape=img_shape,
        x_train_real=x_train_real, y_train_real=y_train_real,
        x_val_real=x_val_real,     y_val_real=y_val_real,
        x_test_real=x_test_real,   y_test_real=y_test_real,
        x_synth=x_synth,           y_synth=y_synth,
        fid_cap_per_class=per_class_cap_for_fid,
        seed=seed,
        epochs=epochs,
    )

    # Images accounting
    counts = {
        "train_real": int(len(x_train_real)),
        "val_real":   int(len(x_val_real)),
        "test_real":  int(len(x_test_real)),
        "synthetic":  (int(len(x_synth)) if x_synth is not None else None),
    }

    # Generative block (provide both 'fid' and 'fid_macro' for aggregator compatibility)
    generative = {
        "fid":         metrics.get("fid_macro"),
        "fid_macro":   metrics.get("fid_macro"),
        "cfid_macro":  metrics.get("cfid_macro"),
        "js":          metrics.get("js"),
        "kl":          metrics.get("kl"),
        "diversity":   metrics.get("diversity"),
        # extras are kept but the aggregator will ignore them if not used
        "cfid_per_class": metrics.get("cfid_per_class"),
        "fid_domain":     metrics.get("fid_domain"),
    }

    util_R_raw  = metrics.get("real_only")
    util_RS_raw = metrics.get("real_plus_synth")
    util_R  = _map_util_names(util_R_raw)
    util_RS = _map_util_names(util_RS_raw)

    # Deltas (RS - R) if both sets are numeric
    def _delta(k: str) -> Optional[float]:
        a = util_RS.get(k)
        b = util_R.get(k)
        return (None if (a is None or b is None) else float(a - b))

    deltas = {
        "accuracy":           _delta("accuracy"),
        "macro_f1":           _delta("macro_f1"),
        "balanced_accuracy":  _delta("balanced_accuracy"),
        "macro_auprc":        _delta("macro_auprc"),
        "recall_at_1pct_fpr": _delta("recall_at_1pct_fpr"),
        "ece":                _delta("ece"),
        "brier":              _delta("brier"),
    }

    summary = {
        "model": model_name,
        "seed":  int(seed),
        "images": counts,
        "generative": generative,
        "utility_real_only": util_R,
        "utility_real_plus_synth": util_RS,
        "deltas_RS_minus_R": deltas,
    }
    return summary


# ======================================================================================
# 5) Optional: convenience wrapper that also writes console+JSON via gcs-core
# ======================================================================================

def write_summary_with_gcs_core(
    *,
    model_name: str,
    seed: int,
    real_dirs: Dict[str, str],
    synth_dir: str,
    fid_cap_per_class: int,
    output_json: str,
    output_console: str,
    metrics: Dict,
    notes: str = "phase2-real",
) -> Dict:
    """
    Phase-2 standalone writer (no gcs_core dependency).

    - Accepts raw `metrics` from compute_all_metrics(...)
    - Produces flattened record with:
        generative, utility_real_only, utility_real_plus_synth,
        deltas_RS_minus_R, images
    - Writes a human-readable console block and appends one JSON line.
    """
    import json
    from pathlib import Path
    from datetime import datetime, timezone
    import numpy as np

    # ---------- map raw metrics -> Phase-2 blocks ----------
    def _map_util_names(util_block: Dict) -> Dict:
        if not isinstance(util_block, dict):
            return {}
        return {
            "accuracy":               util_block.get("accuracy"),
            "macro_f1":               util_block.get("macro_f1"),
            "balanced_accuracy":      util_block.get("bal_acc"),
            "macro_auprc":            util_block.get("macro_auprc"),
            "recall_at_1pct_fpr":     util_block.get("recall_at_1pct_fpr"),
            "ece":                    util_block.get("ece"),
            "brier":                  util_block.get("brier"),
            "per_class":              util_block.get("per_class"),
        }

    gen = {
        "fid":           metrics.get("fid_macro"),
        "fid_macro":     metrics.get("fid_macro"),
        "cfid_macro":    metrics.get("cfid_macro"),
        "js":            metrics.get("js"),
        "kl":            metrics.get("kl"),
        "diversity":     metrics.get("diversity"),
        # optional extras
        "cfid_per_class": metrics.get("cfid_per_class"),
        "fid_domain":     metrics.get("fid_domain"),
    }

    util_R  = _map_util_names(metrics.get("real_only") or {})
    util_RS = _map_util_names(metrics.get("real_plus_synth") or {})

    def _delta(k: str):
        a = util_RS.get(k)
        b = util_R.get(k)
        return None if (a is None or b is None) else float(a - b)

    deltas = {
        "accuracy":           _delta("accuracy"),
        "macro_f1":           _delta("macro_f1"),
        "balanced_accuracy":  _delta("balanced_accuracy"),
        "macro_auprc":        _delta("macro_auprc"),
        "recall_at_1pct_fpr": _delta("recall_at_1pct_fpr"),
        "ece":                _delta("ece"),
        "brier":              _delta("brier"),
    }

    # ---------- best-effort image counts ----------
    def _safe_count(npy_path: str) -> int | None:
        try:
            p = Path(npy_path)
            if p.suffix == ".npy" and p.exists():
                arr = np.load(p, mmap_mode="r")
                return int(arr.shape[0])
        except Exception:
            pass
        return None

    def _count_synth(dir_str: str) -> int | None:
        try:
            d = Path(dir_str)
            # preferred: monolithic x_synth.npy
            x_all = d / "x_synth.npy"
            if x_all.exists():
                arr = np.load(x_all, mmap_mode="r")
                return int(arr.shape[0])
            # fallback: per-class gen_class_*.npy
            total = 0
            found = False
            for gx in sorted(d.glob("gen_class_*.npy")):
                arr = np.load(gx, mmap_mode="r")
                total += int(arr.shape[0])
                found = True
            return total if found else None
        except Exception:
            return None

    images = {
        "train_real": _safe_count(real_dirs.get("train", "")),
        # val/test are often described textually in CLI (split of test), so we may not be
        # able to resolve actual .npy paths here; leave None in that case.
        "val_real":   _safe_count(real_dirs.get("val", "")),
        "test_real":  _safe_count(real_dirs.get("test", "")),
        "synthetic":  _count_synth(synth_dir) if synth_dir else None,
    }

    # ---------- final record ----------
    record = {
        "model": model_name,
        "seed": int(seed),
        "generative": gen,
        "utility_real_only": util_R,
        "utility_real_plus_synth": util_RS,
        "deltas_RS_minus_R": deltas,
        "images": images,
    }

    # ---------- I/O ----------
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_console).parent.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [f"[{ts}] {model_name} | seed={seed} | notes={notes}"]
    if gen.get("fid") is not None:
        lines.append(f"  FID: {gen['fid']:.4f}")
    if gen.get("cfid_macro") is not None:
        lines.append(f"  cFID (macro): {gen['cfid_macro']:.4f}")
    if gen.get("diversity") is not None:
        lines.append(f"  Diversity: {gen['diversity']:.4f}")
    if gen.get("js") is not None or gen.get("kl") is not None:
        lines.append(f"  JS: {gen.get('js')} | KL: {gen.get('kl')}")
    if any(v is not None for v in deltas.values()):
        lines.append("  Utility Δ(RS−R): " + ", ".join(
            f"{k}={v:.4f}" for k, v in deltas.items() if v is not None
        ))
    if any(v is not None for v in images.values()):
        lines.append(f"  Images: {images}")
    lines.append("")
    Path(output_console).write_text("\n".join(lines))

    with open(output_json, "a") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")

    return record


def write_summary_with_gcs_core(
    *,
    model_name: str,
    seed: int,
    real_dirs: Dict[str, str],
    synth_dir: str,
    fid_cap_per_class: int,
    output_json: str,
    output_console: str,
    metrics: Dict,
    notes: str = "phase2-real",
    images_counts: dict | None = None,  # <-- add this
) -> Dict:
    from gcs_core.eval_common import evaluate_model_suite

    summary = evaluate_model_suite(
        model_name=model_name,
        seed=seed,
        real_dirs=real_dirs,
        synth_dir=synth_dir,
        fid_cap_per_class=fid_cap_per_class,
        evaluator="small_cnn_v1",
        output_json=output_json,
        output_console=output_console,
        metrics=metrics,
        notes=notes,
    )

    # If caller provided exact counts, override the auto-inferred ones
    if images_counts:
        imgs = summary.get("images", {})
        imgs.update({k: (int(v) if v is not None else None) for k, v in images_counts.items()})
        summary["images"] = imgs
    return summary

