# 通过微调遵循人类指令 6.5
# 先为有监督微调准备数据集
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import urllib
import re
import time
import torch
import tiktoken
from importlib.metadata import version
from functools import partial
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
# 打印依赖版本
pkgs = [
    "numpy", "matplotlib", "tiktoken", "torch", "tqdm"
]
for p in pkgs:
    print(f"{p} version: {version(p)}")
# 下载数据集
def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data
file_path = "instruction-data.json"
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
data = download_and_load_file(file_path, url)
print("数据总量:", len(data))
# 提示词格式化
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text
# 划分数据集
train_portion = int(len(data) * 0.85)
test_portion = int(len(data) * 0.1)
val_portion = len(data) - train_portion - test_portion
train_data = data[:train_portion]
test_data = data[train_portion : train_portion + test_portion]
val_data = data[train_portion + test_portion :]
print(f"训练集:{len(train_data)}, 验证集:{len(val_data)}, 测试集:{len(test_data)}")
# 指令数据集类
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))
    def __getitem__(self, index):
        return self.encoded_texts[index]
    def __len__(self):
        return len(self.data)
# 自定义批次函数
def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    return (
        torch.stack(inputs_lst).to(device),
        torch.stack(targets_lst).to(device)
    )
# 初始化 tokenizer
tokenizer = tiktoken.get_encoding("gpt2")
# 加载数据
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
customized_collate_fn = partial(
    custom_collate_fn,
    device=device,
    allowed_max_length=1024
)
batch_size = 4
num_workers = 0

train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, collate_fn=customized_collate_fn,
    shuffle=True, drop_last=True, num_workers=num_workers
)
val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, collate_fn=customized_collate_fn,
    shuffle=False, drop_last=False, num_workers=num_workers
)
test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, collate_fn=customized_collate_fn,
    shuffle=False, drop_last=False, num_workers=num_workers
)
# 加载 GPT 模型
from gpt_download import download_and_load_gpt2
from ch03 import GPTModel
from ch04 import load_weights_into_gpt, generate, text_to_token_ids, token_ids_to_text
from ch04 import calc_loss_loader, train_model_simple, plot_losses
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True
}
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
}
CHOOSE_MODEL = "gpt2-small (124M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
model_size = CHOOSE_MODEL.split(" ")[-1].strip("()")
settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()
model.to(device)
# 测试生成
torch.manual_seed(123)
input_text = format_input(val_data[0])
print("\n输入提示词:\n", input_text)
token_ids = generate(
    model=model,
    idx=text_to_token_ids(input_text, tokenizer).to(device),
    max_new_tokens=256,
    context_size=BASE_CONFIG["context_length"],
    eos_id=50256
)
generated_text = token_ids_to_text(token_ids, tokenizer)
response = generated_text[len(input_text):].strip()
print("\n微调前模型回复:\n", response)
# 开始微调
model.train()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
num_epochs = 1
start_time = time.time()
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=1,
    start_context="### Instruction:\nHello", tokenizer=tokenizer
)
end_time = time.time()
print(f"训练耗时: {(end_time-start_time)/60:.2f} 分钟")
# 画损失曲线
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
# 生成测试集回复
model.eval()
torch.manual_seed(123)
for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
    input_text = format_input(entry)
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = generated_text[len(input_text):].replace("### Response:", "").strip()
    test_data[i]["model_response"] = response_text

# 保存结果
with open("instruction-data-with-response.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f, indent=4, ensure_ascii=False)
# 保存模型
file_name = f"{re.sub(r'[()]', '', CHOOSE_MODEL)}-sft.pth"
torch.save(model.state_dict(), file_name)
print(f"\n模型已保存: {file_name}")