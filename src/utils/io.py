import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import urllib.request


def download_file(url: str, save_path: Path) -> None:
    """
    下载文件并处理异常
    
    Args:
        url: 文件URL
        save_path: 保存路径
    
    Raises:
        RuntimeError: 下载或写入失败时
    """
    try:
        # 确保父目录存在
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with urllib.request.urlopen(url) as response:
            with open(save_path, "wb") as f:
                f.write(response.read())
        
        print(f"Successfully downloaded: {save_path}")
    
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP error {e.code}: {e.reason}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"URL error: {e.reason}") from e
    except IOError as e:
        raise RuntimeError(f"File write error: {e}") from e


def download_and_load_file(file_path: str, url: str) -> Any:
    """
    下载并加载JSON文件
    
    Args:
        file_path: 本地文件路径
        url: 远程URL
    
    Returns:
        Any: JSON解析后的数据
    """
    if not os.path.exists(file_path):
        download_file(url, Path(file_path))
    
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: Dict, file_path: str, indent: int = 4) -> None:
    """
    保存数据为JSON文件
    
    Args:
        data: 要保存的数据
        file_path: 保存路径
        indent: JSON缩进
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(file_path: str) -> Dict:
    """
    加载JSON文件
    
    Args:
        file_path: 文件路径
    
    Returns:
        Dict: JSON数据
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_model(model: torch.nn.Module, file_path: str) -> None:
    """
    保存模型权重
    
    Args:
        model: PyTorch模型
        file_path: 保存路径
    """
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to: {file_path}")


def load_model(model: torch.nn.Module, file_path: str, device: Optional[torch.device] = None) -> None:
    """
    加载模型权重
    
    Args:
        model: PyTorch模型
        file_path: 模型文件路径
        device: 设备（可选）
    """
    map_location = device if device else torch.device("cpu")
    model.load_state_dict(torch.load(file_path, map_location=map_location))
    model.eval()
    print(f"Model loaded from: {file_path}")