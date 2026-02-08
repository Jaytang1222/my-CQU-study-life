import os
from typing import Any, Dict

import jittor as jt


def save_checkpoint(model: jt.Module, optimizer: Any, epoch: int, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    jt.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        },
        path,
    )


def load_checkpoint(model: jt.Module, optimizer: Any, path: str) -> Dict[str, Any]:
    state = jt.load(path)
    model.load_parameters(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return state

