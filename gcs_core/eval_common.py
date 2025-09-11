
from __future__ import annotations
from typing import Dict, Optional
from pathlib import Path
from .utils import write_json_line, now_iso

def evaluate_model_suite(
    model_name: str,
    seed: int,
    real_dirs: Dict[str, str],
    synth_dir: str,
    fid_cap_per_class: int,
    evaluator: str,
    output_json: str,
    output_console: str,
    metrics: Optional[Dict] = None,
    notes: str = "",
) -> Dict:
    if metrics is None:
        metrics = {
            "fid_macro": 0.0, "cfid_macro": 0.0, "js": 0.0, "kl": 0.0, "diversity": 0.0,
            "real_only": {"accuracy":0.0,"macro_f1":0.0,"bal_acc":0.0,"macro_auprc":0.0,"recall_at_1pct_fpr":0.0,"ece":0.0,"brier":0.0},
            "real_plus_synth": {"accuracy":0.0,"macro_f1":0.0,"bal_acc":0.0,"macro_auprc":0.0,"recall_at_1pct_fpr":0.0,"ece":0.0,"brier":0.0},
        }
    deltas = {k: metrics["real_plus_synth"][k] - metrics["real_only"][k]
              for k in ["accuracy","macro_f1","bal_acc","macro_auprc","recall_at_1pct_fpr","ece","brier"]}
    record = {
        "timestamp": now_iso(),
        "model": model_name,
        "seed": int(seed),
        "real_train_dir": real_dirs.get("train",""),
        "val_dir": real_dirs.get("val",""),
        "test_dir": real_dirs.get("test",""),
        "synth_dir": synth_dir,
        "fid_cap_per_class": int(fid_cap_per_class),
        "metrics": metrics,
        "deltas": deltas,
        "notes": notes,
    }
    Path(output_console).parent.mkdir(parents=True, exist_ok=True)
    cb = f"""=== GenCyberSynth Eval ===
model: {model_name} | seed: {seed}
splits: train={record['real_train_dir']} | val={record['val_dir']} | test={record['test_dir']}
synth:  {synth_dir} | fid_cap: {fid_cap_per_class}

Generative: FID={metrics['fid_macro']:.3f} | cFID={metrics['cfid_macro']:.3f} | JS={metrics['js']:.3f} | KL={metrics['kl']:.3f} | Div={metrics['diversity']:.3f}
Downstream (R):  Acc={metrics['real_only']['accuracy']:.4f} | F1={metrics['real_only']['macro_f1']:.4f} | BalAcc={metrics['real_only']['bal_acc']:.4f} | AUPRC={metrics['real_only']['macro_auprc']:.4f} | R@1%FPR={metrics['real_only']['recall_at_1pct_fpr']:.4f} | ECE={metrics['real_only']['ece']:.4f} | Brier={metrics['real_only']['brier']:.4f}
Downstream (R+S):Acc={metrics['real_plus_synth']['accuracy']:.4f} | F1={metrics['real_plus_synth']['macro_f1']:.4f} | BalAcc={metrics['real_plus_synth']['bal_acc']:.4f} | AUPRC={metrics['real_plus_synth']['macro_auprc']:.4f} | R@1%FPR={metrics['real_plus_synth']['recall_at_1pct_fpr']:.4f} | ECE={metrics['real_plus_synth']['ece']:.4f} | Brier={metrics['real_plus_synth']['brier']:.4f}
Δ (R+S − R):     Acc={deltas['accuracy']:.4f} | F1={deltas['macro_f1']:.4f} | BalAcc={deltas['bal_acc']:.4f} | AUPRC={deltas['macro_auprc']:.4f} | R@1%FPR={deltas['recall_at_1pct_fpr']:.4f} | ECE={deltas['ece']:.4f} | Brier={deltas['brier']:.4f}
"""
    Path(output_console).write_text(cb, encoding="utf-8")
    write_json_line(record, output_json)
    return record
