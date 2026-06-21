from typing import Optional

import torch
from torch import nn


def generate(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    eos_id: Optional[int] = None
) -> torch.Tensor:
    """
    文本生成函数（支持温度采样和Top-K采样）
    
    Args:
        model: 语言模型
        idx: 输入token IDs，形状为 (batch_size, seq_len)
        max_new_tokens: 最大生成token数
        context_size: 上下文窗口大小
        temperature: 温度系数，0表示贪婪搜索，>0表示随机采样
        top_k: Top-K采样，只保留概率最高的k个token
        eos_id: 结束符ID，遇到则停止生成
    
    Returns:
        torch.Tensor: 生成的完整token序列
    
    Example:
        >>> token_ids = generate(
        ...     model=model,
        ...     idx=text_to_token_ids("Hello", tokenizer).to(device),
        ...     max_new_tokens=50,
        ...     context_size=1024,
        ...     temperature=1.0,
        ...     top_k=50
        ... )
    """
    for _ in range(max_new_tokens):
        # 限制上下文长度
        idx_cond = idx[:, -context_size:]
        
        # 前向传播获取logits
        with torch.no_grad():
            logits = model(idx_cond)
        
        # 只取最后一个token的logits
        logits = logits[:, -1, :]
        
        # Top-K筛选
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf")).to(logits.device),
                logits
            )
        
        # 温度缩放和概率计算
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.argmax(probs, dim=-1, keepdim=True)
        
        # 遇到结束符停止
        if eos_id is not None and idx_next.item() == eos_id:
            break
        
        # 拼接新生成的token
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx


def generate_text_simple(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int
) -> torch.Tensor:
    """
    简单的贪婪文本生成函数
    
    Args:
        model: 语言模型
        idx: 输入token IDs
        max_new_tokens: 最大生成token数
        context_size: 上下文窗口大小
    
    Returns:
        torch.Tensor: 生成的token序列
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        
        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx