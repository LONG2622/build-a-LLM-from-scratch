"""
GPT-2 从零构建项目
提供统一的配置管理和工具函数
"""

from .configs.model_configs import (
    ModelConfig,
    TrainingConfig,
    GenerationConfig,
    MODEL_CONFIGS,
    get_model_config,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_TRAINING_CONFIG,
    DEFAULT_GENERATION_CONFIG,
)

from .utils.generation import generate, generate_text_simple
from .utils.io import (
    download_file,
    download_and_load_file,
    save_json,
    load_json,
    save_model,
    load_model,
)

__all__ = [
    # 配置类
    "ModelConfig",
    "TrainingConfig",
    "GenerationConfig",
    "MODEL_CONFIGS",
    "get_model_config",
    "DEFAULT_MODEL_CONFIG",
    "DEFAULT_TRAINING_CONFIG",
    "DEFAULT_GENERATION_CONFIG",
    # 生成函数
    "generate",
    "generate_text_simple",
    # IO工具
    "download_file",
    "download_and_load_file",
    "save_json",
    "load_json",
    "save_model",
    "load_model",
]