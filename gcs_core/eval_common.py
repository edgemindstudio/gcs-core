# gcs-core/gcs_core/eval_common.py
"""
Deprecated shim for backward compatibility.

Old code that imports `gcs_core.eval_common` will continue to work,
but you should migrate to:

    from gcs_core.val_common import ...

This shim re-exports everything from `val_common` and emits a
DeprecationWarning once on import. It can be removed in a future
minor release after consumers migrate.
"""
from __future__ import annotations

import warnings as _warnings

# Re-export the public API from the canonical module
from .val_common import *  # noqa: F401,F403

# Mirror __all__ if it exists, else derive it from the current globals
try:  # pragma: no cover - trivial
    from .val_common import __all__ as __all__  # type: ignore
except Exception:  # pragma: no cover - safety net
    __all__ = [n for n in globals() if not n.startswith("_")]

_warnings.warn(
    "gcs_core.eval_common is deprecated; import gcs_core.val_common instead.",
    DeprecationWarning,
    stacklevel=2,
)
