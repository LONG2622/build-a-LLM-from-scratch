## 📚 ch03 内容详解

这一章实现了 **完整的GPT模型架构**，从组件到整体，涵盖了Transformer解码器的核心模块。

---

### 1️⃣ GPT 模型配置

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,      # GPT2词汇表大小
    "context_length": 1024,   # 上下文窗口长度
    "emb_dim": 768,           # 嵌入维度
    "n_heads": 12,            # 注意力头数
    "n_layers": 12,           # Transformer块数量
    "drop_rate": 0.1,         # Dropout率
    "qkv_bias": False         # Q/K/V是否使用偏置
}
```

这是GPT-2 Small (124M参数)的标准配置。

---

### 2️⃣ 虚拟GPT模型（DummyGPTModel）

```python
class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])  # Token嵌入
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])  # 位置嵌入
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
    
    def forward(self, in_idx):
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
```

这是一个**结构完整但功能简化**的模型，用于演示GPT的整体架构。

---

### 3️⃣ 层归一化（LayerNorm）

```python
class Layernorm(nn.Module):
    def __init__(self, emb_dim):
        self.eps = 1e-5                    # 防止分母为0
        self.scale = nn.Parameter(torch.ones(emb_dim))  # 可学习缩放参数
        self.shift = nn.Parameter(torch.zeros(emb_dim)) # 可学习偏移参数
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift  # 仿射变换
```

**作用**：稳定训练过程，防止梯度消失/爆炸，加速收敛。

---

### 4️⃣ GELU 激活函数

```python
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))))
```

**特性**：
- 平滑的非线性函数
- 比ReLU更平滑，在负值区域有小的梯度
- 有助于缓解梯度消失问题

---

### 5️⃣ 前馈神经网络（FeedForward）

```python
class FeedForward(nn.Module):
    def __init__(self, cfg):
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 升维到4倍
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])   # 降维回原维度
        )
```

**设计原理**：先将维度扩展4倍，增加模型表达能力，再压缩回原维度。

---

### 6️⃣ 残差连接（Residual Connection）演示

```python
class ExampleDeepNeuralNetwork(nn.Module):
    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output  # 残差连接
            else:
                x = layer_output
        return x
```

**对比实验**：
- 无残差连接：梯度会随着层数加深而消失
- 有残差连接：梯度可以直接通过短路路径反向传播

---

### 7️⃣ Transformer 块

```python
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        self.att = MultiHeadAttention(...)  # 多头注意力
        self.ff = FeedForward(cfg)          # 前馈网络
        self.norm1 = Layernorm(cfg["emb_dim"])
        self.norm2 = Layernorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
    
    def forward(self, x):
        # 注意力残差
        shortcut = x
        x = self.norm1(x)           # 先归一化
        x = self.att(x)             # 再注意力
        x = self.drop_shortcut(x)
        x = x + shortcut            # 残差连接
        
        # 前馈网络残差
        shortcut = x
        x = self.norm2(x)           # 先归一化
        x = self.ff(x)              # 再前馈
        x = self.drop_shortcut(x)
        x = x + shortcut            # 残差连接
        return x
```

**注意**：这里使用的是 **Pre-LN** 架构（归一化在注意力/前馈之前）。

---

### 8️⃣ 完整 GPT 模型

```python
class GPTModel(nn.Module):
    def __init__(self, cfg):
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = Layernorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
```

---

### 9️⃣ 参数计算

```python
total_params = sum(p.numel() for p in model.parameters())  # 总参数
total_params_gpt2 = total_params - sum(p.numel() for p in model.out_head.parameters())  # 共享权重后
```

**关键点**：GPT-2通过**权重共享**（输出层与嵌入层共享权重）将参数从约1.5亿降到1.24亿。

---

### 🔟 文本生成

```python
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]  # 取最后context_size个token
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]          # 只取最后一个token的预测
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # greedy decoding
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

这是**贪婪解码**（greedy decoding），每次选择概率最高的token。

---

## 🎯 GPT 完整架构图

```
输入 token IDs [batch, seq_len]
        ↓
┌─────────────────────────┐
│   Token Embedding       │  [batch, seq_len, emb_dim]
└─────────────────────────┘
        ↓
┌─────────────────────────┐
│   Positional Embedding  │  [seq_len, emb_dim]
└─────────────────────────┘
        ↓ (相加)
┌─────────────────────────┐
│      Dropout            │
└─────────────────────────┘
        ↓
┌─────────────────────────┐
│  Transformer Block x 12 │  × n_layers
│  - LayerNorm           │
│  - MultiHeadAttention  │
│  - Residual            │
│  - LayerNorm           │
│  - FeedForward         │
│  - Residual            │
└─────────────────────────┘
        ↓
┌─────────────────────────┐
│   Final LayerNorm       │
└─────────────────────────┘
        ↓
┌─────────────────────────┐
│   Output Head (Linear)  │  [batch, seq_len, vocab_size]
└─────────────────────────┘
        ↓
输出 logits (预测下一个token的概率)
```

---

## 💡 核心要点总结

| 组件 | 作用 |
|------|------|
| **LayerNorm** | 稳定训练，加速收敛 |
| **GELU** | 平滑激活函数，缓解梯度消失 |
| **FeedForward** | 非线性变换，增加模型表达能力 |
| **残差连接** | 解决深度网络梯度消失问题 |
| **Transformer Block** | 注意力 + 前馈的组合单元 |
| **权重共享** | 减少参数数量，提升泛化能力 |

这一章完成了GPT模型的完整实现，下一章应该会涉及模型训练和优化。