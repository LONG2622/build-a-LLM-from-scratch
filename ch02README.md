## 📚 ch02 内容详解

这一章聚焦于 **Transformer的核心——注意力机制**，实现了从基础因果注意力到高效多头注意力的完整演进过程。

---

### 1️⃣ 因果注意力（Causal Attention）

```python
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        self.W_query = torch.nn.Linear(d_in, d_out)
        self.W_key = torch.nn.Linear(d_in, d_out)
        self.W_value = torch.nn.Linear(d_in, d_out)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))
```

**核心思想**：
- **Q/K/V 投影**：通过三个线性层将输入分别转换为 Query、Key、Value
- **因果掩码**：`torch.triu` 生成上三角矩阵，确保每个位置只能关注前面的位置（防止未来信息泄露）

**前向传播流程**：
```python
def forward(self, x):
    queries = self.W_query(x)    # [batch, tokens, d_out]
    keys = self.W_key(x)         # [batch, tokens, d_out]
    values = self.W_value(x)     # [batch, tokens, d_out]
    
    attn_scores = queries @ keys.transpose(1, 2)  # 计算相似度
    attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)  # 应用掩码
    attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)  # 归一化
    context_vec = attn_weights @ values  # 加权求和
    return context_vec
```

**关键公式**：
$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

---

### 2️⃣ 多头注意力封装（简单版）

```python
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        self.heads = nn.ModuleList([
            CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
            for _ in range(num_heads)
        ])
    
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
```

**设计思路**：创建多个独立的注意力头，并行计算后拼接结果。但这种实现效率较低。

---

### 3️⃣ 高效多头注意力（正式版）

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        
        self.W_query = nn.Linear(d_in, d_out)
        self.W_key = nn.Linear(d_in, d_out)
        self.W_value = nn.Linear(d_in, d_out)
        self.out_proj = nn.Linear(d_out, d_out)  # 输出投影
```

**高效实现技巧**：

```python
def forward(self, x):
    b, num_tokens, d_in = x.shape
    
    # 1. 线性投影
    keys = self.W_key(x)      # [b, tokens, d_out]
    queries = self.W_query(x)
    values = self.W_value(x)
    
    # 2. 重塑为多头格式 [b, num_heads, tokens, head_dim]
    keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
    queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
    values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
    
    # 3. 注意力计算（批量处理所有头）
    attn_scores = queries @ keys.transpose(2, 3)  # [b, heads, tokens, tokens]
    attn_scores.masked_fill_(mask_bool, -torch.inf)
    attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
    attn_weights = self.dropout(attn_weights)
    
    # 4. 输出拼接
    context_vec = (attn_weights @ values).transpose(1, 2)  # [b, tokens, heads, head_dim]
    context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)  # 拼接
    context_vec = self.out_proj(context_vec)  # 最终投影
    return context_vec
```

---

## 🎯 核心要点总结

| 组件 | 作用 | 关键特性 |
|------|------|----------|
| **Q/K/V 投影** | 将输入转换为三种向量表示 | 学习不同的语义视角 |
| **因果掩码** | 防止未来信息泄露 | 上三角矩阵，确保自回归特性 |
| **缩放点积** | 计算注意力权重 | 除以 $\sqrt{d_k}$ 防止梯度消失 |
| **多头机制** | 并行学习多种注意力模式 | 提升模型表达能力 |
| **输出投影** | 融合多头结果 | 线性变换整合信息 |

---

## 🔄 数据流可视化

```
输入 x [batch, tokens, d_in]
    ↓
Q/K/V 线性投影 [batch, tokens, d_out]
    ↓
重塑为多头 [batch, num_heads, tokens, head_dim]
    ↓
注意力分数计算 [batch, num_heads, tokens, tokens]
    ↓
因果掩码 + Softmax
    ↓
加权求和 [batch, num_heads, tokens, head_dim]
    ↓
转置拼接 [batch, tokens, d_out]
    ↓
输出投影 [batch, tokens, d_out]
```

这一章的注意力机制是GPT架构的核心，后续章节会在此基础上构建完整的Transformer解码器。