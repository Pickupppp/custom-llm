import os
import random
import torch
from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from config import CustomConfig
from dataset import CustomDataset
from model import CustomForCausalLM
from utils import get_model_size, data_collator_closure

torch.manual_seed(42)
random.seed(42)

# tokenizer 部分
tokenizer = Tokenizer.from_file("tokenizer.json")
tokenizer.post_processor = TemplateProcessing(
    single="<|bos|> $A <|eos|>",
    special_tokens=[
        ("<|bos|>", tokenizer.token_to_id("<|bos|>")),
        ("<|eos|>", tokenizer.token_to_id("<|eos|>")),
    ],
)
tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<|eos|>"), pad_token="<|eos|>")

# 训练参数
num_epochs = 7
batch_size = 1
gradient_accumulation = 16
lr = 2e-4
weight_decay = 0.01
max_norm = 1.0
effective_batch_size = gradient_accumulation * batch_size

# 数据集准备
file_name = "pretrained_demo.txt"
dataset = CustomDataset(file_name, tokenizer, max_length=4096)
num_repeats = effective_batch_size - len(dataset) % effective_batch_size

if num_repeats > 0:
    repeated_data = random.sample(dataset.raw_datas, num_repeats)
    dataset.raw_datas += repeated_data


data_collator = data_collator_closure(tokenizer.token_to_id("<|eos|>"))
data_loader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator
)


assert len(data_loader) % gradient_accumulation == 0
num_training_steps = num_epochs * (len(data_loader) // gradient_accumulation)
print(f"Total training step is {num_training_steps}")


loggin_step = 20
saving_step = num_training_steps // 10
steps = 0
ckpt_dir = "models"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = CustomConfig(
    vocab_size=tokenizer.get_vocab_size(),
    max_position_embeddings=2048,
    num_hidden_layers=3,
    pad_token_id=tokenizer.token_to_id("<|eos|>"),
)
model = CustomForCausalLM(config)
print(f"Model size is {get_model_size(model)}")
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0.1 * num_training_steps,
    num_training_steps=num_training_steps,
)


model = model.to(device)
model.train()
progress_bar = tqdm(range(num_training_steps))
for epoch in range(num_epochs):
    for idx, batch in enumerate(data_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits, loss = outputs
        loss /= gradient_accumulation
        loss.backward()

        if (idx + 1) % gradient_accumulation == 0:
            # 确保梯度没有溢出
            # scaler.unscale_(optimizer)
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            # scaler.step(optimizer)
            # scaler.update()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        if (steps + 1) % (loggin_step * gradient_accumulation) == 0:
            print(
                f"steps: {(steps + 1) // gradient_accumulation}, loss: {loss * gradient_accumulation}"
            )
        if (steps + 1) % (saving_step * gradient_accumulation) == 0:
            ckpt_path = os.path.join(
                ckpt_dir,
                f"checkpoint-{(steps + 1) // gradient_accumulation}",
            )
            os.makedirs(ckpt_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(ckpt_path, "model.pt"))
        steps += 1


torch.save(model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
