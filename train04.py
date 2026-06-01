from L02 import *
import torch
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim" :768,
    "n_heads": 12,
    "n_layers":12,
    "drop_rate":0.1,
    "qkv_bias" : False
}
torch.manusl_seed(123)
model = (GPTModel(GPT_CONFIG_124M))
model.eval()