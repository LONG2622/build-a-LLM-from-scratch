#编码注意力机制（encoding attention mechanisim）
import os
import torch
inputs = torch.tensor(
    [[0.43, 0.15, 0.89],
     [0.55, 0.87, 0.66],
     [0.57, 0.85, 0.64],
     [0.22, 0.58, 0.33],
     [0.77, 0.25, 0.10],
     [0.05, 0.80, 0.55]]
)
input_query = inputs[1]
input_query
input_1 = inputs[0]
input_1

#做点积
torch.dot(input_query ,input_1)

torch.empty(inputs.shape[0])
query = inputs[1]

attn_score_2 = torch.empty(inputs.shape[0])
for i , x_i in enumerate(inputs):
    attn_score_2[i]= torch.dot (x_i, input_query)
print(attn_score_2)

#归一化计算注意力权重
attn_weight_2_tmp = attn_score_2 / attn_score_2.sum()
print("attention weights:",attn_weight_2_tmp)
print("sum:",attn_weight_2_tmp.sum())

def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim = 0)
attn_weight_2_naive = softmax_naive(attn_score_2)
print("attention weight:" , attn_weight_2_naive)
print("sum:" , attn_weight_2_naive.sum())

attn_weight_2 = torch.softmax(attn_score_2, dim = 0)


query = inputs[1]
context_vec_2 = torch.zeros(query.shape)
for x, x_i in enumerate(inputs):
    context_vec_2 += attn_weight_2[i] * x_i
print(context_vec_2)

for i,x_i in enumerate(inputs):
    print(f"{attn_weight_2[i]} --->{inputs[i]}")
    context_vec_2 += attn_weight_2[i] * x_i

#无权重的自注意力机制
attn_scores = torch.empty(6 ,6)

for i, x_i in enumerate (inputs):
    for j , x_j in enumerate(inputs):
        attn_scores[i , j] = torch.dot(x_i , x_j)
print(attn_scores)

attn_weight = torch.softmax(attn_scores , dim = 1)
attn_weight
attn_weight.sum(dim = -1)
torch.sum(torch.tensor([0.1385, 0.2184, 0.2128, 0.1420, 0.0988, 0.1896]))

attn_scores = inputs @ inputs.T
attn_weight = torch.softmax(attn_scores, dim = 1)
print(attn_weight)

all_context_vecs = attn_weight @ inputs
all_context_vecs

#计算所有token的权重
#先找到张量
inputs
x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2
#生成权重参数
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in , d_out))
W_key = torch.nn.Parameter(torch.rand(d_in ,d_out))
W_value = torch.nn.Parameter(torch.rand(d_in ,d_out))
W_query
query_2 = x_2 @ W_query
query_2 

key = inputs @ W_key
value = inputs @ W_value

key_2 = key[1]
attn_score_22 = torch.dot(query_2 ,key_2)
attn_score_22

import math
attn_score_2 = query_2 @ key.T
attn_score_2
d_k = key.shape[1]
attn_weights =torch.softmax(attn_score_2 / d_k ** 0.5, dim = -1)
attn_weights

attn_weights.sum()
context_vec_2 = attn_weights @ value 
#自注意力类
import os
import torch
import torch.nn as nn
m = torch.nn.Linear(2,3)
m.bias
class SelfAttention_V1(nn.Module):
    def __init__(self , d_in ,d_out):
        super().__init__()
        W_query = torch.nn.Parameter(torch.rand(d_in , d_out))
        W_key = torch.nn.Parameter(torch.rand(d_in ,d_out))
        W_value = torch.nn.Parameter(torch.rand(d_in ,d_out))
    def forward(self , x):
        query = inputs @ W_query
        key = inputs @ W_key
        value = inputs @ W_value
        attn_score_2 = query_2 @ key.T
        attn_weights =torch.softmax(attn_score_2 / d_k ** 0.5, dim = -1)
        context_vec = attn_weights @value
        return context_vec

torch.manual_seed(123)
sa_v1 = SelfAttention_V1(d_in ,d_out)
sa_v1(inputs)
#添加了QKV偏差
m = torch.nn.Linear(2,3)
m.bias
class SelfAttention_V1(nn.Module):
    def __init__(self , d_in , d_out ,qkv_bias = False):
        super().__init__()
        self.W_query = torch.nn.Linear(d_in , d_out, bias =qkv_bias)
        self.W_key = torch.nn.Linear(d_in , d_out, bias =qkv_bias)
        self.W_value = torch.nn.Linear(d_in , d_out, bias =qkv_bias)
    def forward(self , x):
        query = self.W_query(inputs)
        key = self.W_key(inputs)
        value = self.W_value(inputs)
        attn_score_2 = query_2 @ key.T
        attn_weights =torch.softmax(attn_score_2 / d_k ** 0.5, dim = -1)
        context_vec = attn_weights @value
        return context_vec

torch.manual_seed(123)
sa_v2 = SelfAttention_V1(d_in ,d_out)
sa_v2(inputs)

query = sa_v2.W_query(inputs)
key = sa_v2.W_key(inputs)
value = sa_v2.W_value(inputs)
attn_score_2 = query @ key.T
attn_weights =torch.softmax(attn_score_2 / d_k ** 0.5, dim = -1)
attn_weights

context_length= attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length , context_length))
print(mask_simple)

mask_simple = attn_weights * mask_simple
mask_simple

mask = torch.triu(torch.ones(context_length, context_length), diagonal = 1)
masked = attn_scores.masked_fill(mask.bool() , -torch.inf)
print(masked)

#用dropout掩码额外的注意力权重
torch.manual_seed(123)
layer = torch.nn.Dropout(0.5)
example= torch.ones(6,6)
example
layer(example)
dropout_rate = 0.5
1/(1- dropout_rate)

#实现因果注意力类
batch = torch.stack((inputs,inputs), dim =0)
batch.shape
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, dropout,qkv_bias=False):
        super().__init__()
        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer("mask",torch.ones(context_length, context_length), diagonal = 1)

    def forward(self, x):
        b , num_tokens,d_in =x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        attn_scores =queries @ keys.transpose(1,2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens] , -torch.inf)
        attn_weights = torch.softmax(attn_scores / d_k**0.5, dim=-1)
        context_vec = attn_weights @ values

        return context_vec
torch.manual_seed(789)
context_length = batch.shape[1]
dropout = 0.0
ca = CausalAttention(d_in, d_out, context_length , dropout)
ca(batch)
"""
import torch
import torch.nn as nn

# 先把之前的变量准备好（你必须有这两个）
d_in = 256    # 输入维度
d_out = 256  # 输出维度

# 正确的因果注意力类
class CausalAttention(nn.Module):
    # 🔥 修复1：__init__ 双下划线
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()  # 🔥 修复2：正确写法
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        
        # 🔥 修复3：正确的上三角掩码
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        # 🔥 修复4：正确掩码
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        
        d_k = d_out
        attn_weights = torch.softmax(attn_scores / d_k**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context_vec = attn_weights @ values
        return context_vec

# 测试运行
torch.manual_seed(789)
context_length = 4  # 你的输入长度
dropout = 0.0

# 🔥 修复5：参数匹配
ca = CausalAttention(d_in=d_in, d_out=d_out, context_length=context_length, dropout=dropout)

# 构造测试数据
batch = torch.randn(2, context_length, d_in)
output = ca(batch)
print("输出形状:", output.shape)"""

class MultiHeadAttentionWrapper(nn.Module):
    def  __init__ (self, d_in, d_out, context_length, dropout, num_heads=2, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList([
            CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)
            ])
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
torch.manual_seed(123)
context_length=batch.shape[1]
d_in,d_out = batch.shape[0] ,2
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, dropout = 0.0, num_heads = 2)
mha(batch)