from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class ModelConfig:
    """GPT模型配置类"""
    vocab_size: int = 50257
    context_length: int = 1024
    emb_dim: int = 768
    n_heads: int = 12
    n_layers: int = 12
    drop_rate: float = 0.1
    qkv_bias: bool = False


@dataclass
class TrainingConfig:
    """训练配置类"""
    batch_size: int = 4
    learning_rate: float = 5e-5
    num_epochs: int = 10
    weight_decay: float = 0.1
    eval_freq: int = 5
    eval_iter: int = 1
    gradient_accumulation_steps: int = 1


@dataclass
class GenerationConfig:
    """文本生成配置类"""
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_k: Optional[int] = None
    eos_id: Optional[int] = None


# 预定义模型配置
MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "gpt2-small": ModelConfig(
        vocab_size=50257,
        context_length=1024,
        emb_dim=768,
        n_heads=12,
        n_layers=12,
        drop_rate=0.1,
        qkv_bias=False
    ),
    "gpt2-medium": ModelConfig(
        vocab_size=50257,
        context_length=1024,
        emb_dim=1024,
        n_heads=16,
        n_layers=24,
        drop_rate=0.1,
        qkv_bias=False
    ),
    "gpt2-large": ModelConfig(
        vocab_size=50257,
        context_length=1024,
        emb_dim=1280,
        n_heads=20,
        n_layers=36,
        drop_rate=0.1,
        qkv_bias=False
    ),
    "gpt2-xl": ModelConfig(
        vocab_size=50257,
        context_length=1024,
        emb_dim=1600,
        n_heads=25,
        n_layers=48,
        drop_rate=0.1,
        qkv_bias=False
    ),
}


def get_model_config(model_name: str) -> ModelConfig:
    """
    获取预定义的模型配置
    
    Args:
        model_name: 模型名称，如 "gpt2-small", "gpt2-medium"
    
    Returns:
        ModelConfig: 对应的模型配置
    
    Raises:
        ValueError: 如果模型名称不存在
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_name]


# 默认配置
DEFAULT_MODEL_CONFIG = MODEL_CONFIGS["gpt2-small"]
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_GENERATION_CONFIG = GenerationConfig()