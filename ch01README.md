这是第一章的代码内容，主要介绍了构建LLM（大语言模型）的基础数据处理流程。

## 📚 ch01 内容概述

这一章主要涵盖了 **从原始文本到模型输入** 的完整数据处理流程，是构建LLM的基础准备工作。

***

### 1️⃣ 下载数据集

```python
if not os.path.exists("the-verdict.txt"):
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
    urllib.request.urlretrieve(url, "the-verdict.txt")
```

从GitHub下载一个小说文本作为训练数据，这是后续所有操作的数据源。

***

### 2️⃣ 基础文本预处理

```python
text = "hello , world . Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
```

使用正则表达式按标点符号和空格分词，这是最简单的分词方式。

***

### 3️⃣ 自定义简易分词器

```python
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}
    
    def encode(self, text): ...  # 文本转token ID
    def decode(self, ids): ...   # token ID转文本
```

实现了一个基于单词级别的分词器，包含：

- `encode()`: 文本 → 整数ID
- `decode()`: 整数ID → 文本

***

### 4️⃣ GPT2 官方 BPE 分词器

```python
gpt2_tokenizer = tiktoken.get_encoding("gpt2")
```

使用 OpenAI 官方的 `tiktoken` 库加载 GPT2 的 BPE（Byte Pair Encoding）分词器。

***

### 5️⃣ 滑动窗口构建样本

```python
context_size = 4
for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
```

展示了 **自回归语言建模** 的核心思想：用前 `i` 个token预测第 `i+1` 个token。

***

### 6️⃣ GPT 数据集类

```python
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i+1:i + max_length + 1]
```

将长文本切分为固定长度的训练样本，使用 `stride`（步长）实现重叠采样，提高数据利用率。

***

### 7️⃣ 数据加载器

```python
def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, ...):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, ...)
```

封装了PyTorch的 `DataLoader`，方便批量加载数据。

***

### 8️⃣ Token 嵌入 + 位置嵌入

```python
token_emb = torch.nn.Embedding(vocab_size, output_dim)
pos_emb = torch.nn.Embedding(4, output_dim)
input_embeddings = token_embeddings + pos_embeddings
```

这是Transformer架构的关键组件：

- **Token嵌入**: 将每个token ID映射到一个向量
- **位置嵌入**: 提供位置信息，因为Transformer本身是顺序无关的

***

## 🎯 核心要点

这一章的核心是理解 **如何将原始文本转换为模型可训练的格式**：

```
原始文本 → 分词 → 编码为ID → 构建(input, target)样本 → 嵌入层处理
```

这些都是构建GPT等LLM模型的基础准备工作，为后续的Transformer架构实现打下基础。
