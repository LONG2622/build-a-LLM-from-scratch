from typing import Optional, Tuple, List

import torch
from torch import nn
from torch.utils.data import DataLoader


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: nn.Module,
    device: torch.device
) -> torch.Tensor:
    """
    计算单个批次的交叉熵损失（文本生成任务）
    
    Args:
        input_batch: 输入token IDs
        target_batch: 目标token IDs
        model: 语言模型
        device: 计算设备
    
    Returns:
        torch.Tensor: 损失值
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    
    logits = model(input_batch)
    loss = nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten()
    )
    
    return loss


def calc_loss_loader(
    data_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    num_batches: Optional[int] = None
) -> float:
    """
    计算数据加载器的平均损失
    
    Args:
        data_loader: 数据加载器
        model: 语言模型
        device: 计算设备
        num_batches: 计算的批次数（None表示全部）
    
    Returns:
        float: 平均损失
    """
    total_loss = 0.0
    count = 0
    
    if len(data_loader) == 0:
        return float("nan")
    
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
        count += 1
    
    return total_loss / count if count > 0 else float("nan")


def calc_accuracy_loader(
    data_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    num_batches: Optional[int] = None
) -> float:
    """
    计算分类任务的准确率
    
    Args:
        data_loader: 数据加载器
        model: 分类模型
        device: 计算设备
        num_batches: 计算的批次数（None表示全部）
    
    Returns:
        float: 准确率
    """
    model.eval()
    correct_predictions = 0
    num_examples = 0
    
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= num_batches:
                break
            
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            
            logits = model(input_batch)[:, -1, :]
            predicted_labels = torch.argmax(logits, dim=-1)
            
            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
    
    model.train()
    return correct_predictions / num_examples if num_examples > 0 else 0.0


def evaluate_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    eval_iter: int
) -> Tuple[float, float]:
    """
    评估模型在训练集和验证集上的损失
    
    Args:
        model: 语言模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 计算设备
        eval_iter: 评估的批次数
    
    Returns:
        Tuple[float, float]: (训练损失, 验证损失)
    """
    model.eval()
    
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    
    model.train()
    return train_loss, val_loss


def text_to_token_ids(text: str, tokenizer) -> torch.Tensor:
    """
    将文本转换为token IDs张量
    
    Args:
        text: 输入文本
        tokenizer: tiktoken分词器
    
    Returns:
        torch.Tensor: token IDs张量，形状为 (1, seq_len)
    """
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids: torch.Tensor, tokenizer) -> str:
    """
    将token IDs张量转换为文本
    
    Args:
        token_ids: token IDs张量
        tokenizer: tiktoken分词器
    
    Returns:
        str: 解码后的文本
    """
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())