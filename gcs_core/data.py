
from __future__ import annotations
from pathlib import Path
from typing import Dict

def load_splits(real_train_dir:str, val_dir:str, test_dir:str) -> Dict[str, str]:
    return {"train": str(Path(real_train_dir)), "val": str(Path(val_dir)), "test": str(Path(test_dir))}
