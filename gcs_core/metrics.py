
from __future__ import annotations
from typing import Dict

def compute_generative_metrics(val_dir:str, synth_dir:str, fid_cap_per_class:int) -> Dict[str, float]:
    return {"fid_macro": 999.0, "cfid_macro": 999.0, "js": 0.0, "kl": 0.0, "diversity": 0.0}
