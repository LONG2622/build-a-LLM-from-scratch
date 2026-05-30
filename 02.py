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