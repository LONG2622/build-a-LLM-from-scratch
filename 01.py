import os
import urllib.request

if not os.path.exists("the-verdict.txt"):
    url = ("https:/raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/ch02/01_main-chapter-code/the-verdict.txt")
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_pth)

with open("the-verdict.txt" , "r", encoding = UTF-8") as f:
    raw_text = f.read()
raw_text
#指令会显示verdict原文
#检验字数
len(raw_text)
#随便检验一下
import re
text = "hello , worls. This is a test."
result = re.split(r'(\s)' , text)
print(result)
#复杂符号的处理
text = "hello , world . Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print(result)
#统计全文的token数
text = "hello , world . Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
result = [item.strip() for item in result if item.strip()]
preprocessed = result
print(result)
len(preprocessed)
#将这些token（词元）转化为 tokenID
#建立词汇表(构建vocabulary)
#先排序
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
#构建实际词汇表(类似于字典排序)
vocab = {token :integer for integer,token in enumerate(all_words)}
vocab
#标记器
class SimpletokenizerV1:
    def __init__(self , vocab):
        self.str_to_int = vocab
        self.int_to_str = {i :s for s , i in vocab.items() }
    def encode(self,text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)',text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    def decode(self,ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        #replace spaces before the specified punctuations(在指定的标点符号前替换空格)
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
tokenizer = SimpletokenizerV1(vocab)
text = """"It's the last he painted, you know,"
            Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)
#同样可以解码
tokenizer.decode(ids)
#引入特殊上下文token
text = "hello , do you like tea ?"
tokenizer.encode(text)
all_tokens  = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>","<|unk|>"])#添加token
#BPE byte pair encoding
import tiktoken
pip install tiktoken
tiktoken.__version__
# 加载 GPT-2 的分词器
tokenizer = tiktoken.get_encoding("gpt2")

# 测试一句话
text = "Hello, do you like tea?"
ids = tokenizer.encode(text)

print("编码结果：", ids)
print("解码结果：", tokenizer.decode(ids))
"""
# 一键恢复你之前的所有代码
import os
import re
import urllib.request

# 1. 下载数据
if not os.path.exists("the-verdict.txt"):
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
    urllib.request.urlretrieve(url, "the-verdict.txt")

# 2. 读取数据
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# 3. 构建词汇表
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
all_words = sorted(set(preprocessed))
vocab = {token:i for i, token in enumerate(all_words)}

# 4. 定义分词器
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        return [self.str_to_int[s] for s in preprocessed]
    
    def decode(self, ids):
        return " ".join([self.int_to_str[i] for i in ids])

tokenizer = SimpleTokenizerV1(vocab)


# 5. 现在导入 tiktoken 成功！
import tiktoken
print("✅ tiktoken 导入成功！")
"""
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")  # 这是 GPT 专用分词器！

#滑动窗口法数据采样
with open("the-verdict.txt","r",encoding = "utf-8") as f:
    raw_text = tokenizer.encode(raw_text)
enc_text = tokenizer.encode(raw_text)
print(enc_text[:20])
enc_sample = enc_text[50:]
context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x:{x}")
print(f"y:   {y}")

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))
#结果：
# and ----> established    
# and established ----> himself
# and established himself ----> in
# and established himself in ----> a


import torch
from torch.utils.data import Dataset
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i+1 : i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
class GPTDatasetV1(Dataset):
    def __init__ (self , txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt , allowed_special={"<|endoftext|>"})
        for i in range(0 , len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length +1]
            self.input_ids.append(torch.tensor(inout_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, ids):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt , batch= 4 ,max_length = 256,stride = 128, shuffle = True, drop_last = True, num_workers = 0):
    tokenizer = tiktoken.get_enoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = Dataloader(
        dataset, 
        batch_size = batch_size,
        shuffle = shuffle,
        drop_last = drop_last ,
        num_workers = num_workers
    )
    return dataloader
dataloader = create_dataloader_v1(raw_text , batch_size = 1,max_length = 4,stride = 1,shuffle =False)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)
#步幅为4 的数据加载器采样
dataloader = create_dataloader_v1(
    raw_text,
    tokenizer,
    batch_size=8,
    max_length=4,
    stride=4,
    shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Input:\n", inputs)
print("outputs:\n", targets)

#创建token嵌入
inputs_ids = torch.tensor([2  , 3, 5  , 1])
#创建嵌入层
vocab_size  = 6
output_dim= 3
embedding_layer = torch.nn.Embedding(vocab_size , output_dim)
print(embedding_layer.weight)

embedding_layer(torch.tensor([3]))

#encoding word positions
vocab_size=50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
max_length= 4

dataloader = create_dataloader_v1(
    raw_text,
    tokenizer,
    batch_size=8,
    max_length=max_length,
    stride=max_length,
    shuffle=False
)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

token_embeddings = token_embedding_layer(inputs)
token_embeddings.shape

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length , output_dim)
torch.arange(max_length)