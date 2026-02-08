import importlib.util
import os
import random
from types import ModuleType
from typing import Dict, Any

import numpy as np
import jittor as jt


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    jt.set_global_seed(seed)


def load_config(config_path: str) -> Dict[str, Any]:
    spec = importlib.util.spec_from_file_location("cfg_module", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载配置文件: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    if not hasattr(module, "config"):
        raise AttributeError(f"{config_path} 中未找到 config 变量")
    return dict(module.config)

