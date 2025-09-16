# gcs-core/gcs_core/__init__.py
# =============================================================================
# gcs_core: GenCyberSynth “frozen core”
#
# Purpose
# -------
# A minimal, stable facade for the shared evaluation utilities used by all
# model repos (GAN, Diffusion, VAE, Autoregressive, RBM, GMM, MAF, etc.).
#
# Design
# ------
# • Stable API surface:
#       from gcs_core.val_common import evaluate_model_suite
#   Back-compat (deprecated but supported):
#       from gcs_core.eval_common import evaluate_model_suite
#
# • Lazy submodule loading: importing `gcs_core` is fast; submodules load
#   on first access.
#
# • Versioned: `__version__` comes from installed package metadata so runs
#   can record the exact core version used.
#
# Layout (flat package)
# ---------------------
# gcs-core/
#   gcs_core/
#     __init__.py          <-- this file
#     val_common.py        <-- preferred Phase-2 evaluation API
#     eval_common.py       <-- shim re-exporting from val_common (deprecated)
#     synth_loader.py
#     metrics.py
#     data.py
#     utils.py
#     schemas/             <-- JSON Schemas (not imported as modules)
# =============================================================================

from __future__ import annotations

from typing import TYPE_CHECKING
import importlib
import logging

# ---------------------------------------------------------------------------
# Package version (read from installed dist metadata; fallback for editable dev)
# ---------------------------------------------------------------------------
try:
    from importlib.metadata import PackageNotFoundError, version as _pkg_version
except Exception:  # pragma: no cover
    PackageNotFoundError = Exception  # type: ignore
    def _pkg_version(_: str) -> str:  # type: ignore
        return "0.0.0+dev"

try:
    # Distribution name from pyproject.toml -> [project].name = "gcs-core"
    __version__ = _pkg_version("gcs-core")
except PackageNotFoundError:  # e.g., editable install before metadata exists
    __version__ = "0.0.0+dev"

# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------
# We intentionally expose only top-level submodules; concrete functions
# (evaluate_model_suite, compute_all_metrics, etc.) live under those modules.
_SUBMODULES = (
    "val_common",     # preferred Phase-2 API
    "eval_common",    # deprecated shim; re-exports from val_common
    "synth_loader",
    "metrics",
    "data",
    "utils",
)

__all__ = ["__version__", *list(_SUBMODULES)]

# ---------------------------------------------------------------------------
# Logging: never configure global logging on import
# ---------------------------------------------------------------------------
_logger = logging.getLogger(__name__)
_logger.addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# Lazy import hook: `from gcs_core import val_common` stays fast
# ---------------------------------------------------------------------------
def __getattr__(name: str):
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def __dir__() -> list[str]:  # improves discoverability in REPLs
    return sorted(list(globals().keys()) + list(_SUBMODULES))

# ---------------------------------------------------------------------------
# Static type checkers (mypy/pyright) prefer real imports at type-time.
# ---------------------------------------------------------------------------
if TYPE_CHECKING:  # pragma: no cover
    from . import val_common, eval_common, synth_loader, metrics, data, utils  # noqa: F401
