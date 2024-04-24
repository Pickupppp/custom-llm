import os
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


tokenizer = Tokenizer.from_file("tokenizer.json")
tokenizer.post_processor = TemplateProcessing(
    single="<|bos|> $A <|eos|>",
    special_tokens=[
        ("<|bos|>", tokenizer.token_to_id("<|bos|>")),
        ("<|eos|>", tokenizer.token_to_id("<|eos|>")),
    ],
)
tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<|eos|>"), pad_token="<|eos|>")

file_name = "data/test.txt"
dataset = CustomDataset(file_name, tokenizer, max_length=4096)

data_collator = data_collator_closure(tokenizer.token_to_id("<|eos|>"))
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=data_collator)
for batch in data_loader:
    print(batch)
    break


# config = CustomConfig(
#     vocab_size=tokenizer.get_vocab_size(),
#     max_position_embeddings=2048,
#     num_hidden_layers=3,
#     pad_token_id=tokenizer.token_to_id("<|eos|>"),
# )
# model = CustomForCausalLM(config)
# print(f"Model size is {get_model_size(model)}")

# num_epochs = 3
# num_training_steps = num_epochs * len(data_loader)
# optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
# scheduler = get_linear_schedule_with_warmup(
#     optimizer=optimizer,
#     num_warmup_steps=0.1 * num_training_steps,
#     num_training_steps=num_training_steps,
# )

# progress_bar = tqdm(range(num_training_steps))

# loggin_step = 20
# saving_step = 100
# steps = 1
# ckpt_dir = "models"


# model.train()
# for epoch in range(num_epochs):
#     for batch in data_loader:
#         outputs = model(**batch)
#         logits, loss = outputs
#         loss.backward()

#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

#         optimizer.step()
#         scheduler.step()
#         optimizer.zero_grad()
#         if steps % loggin_step == 0:
#             print(f"steps: {steps}, loss: {loss}")
#         if steps % saving_step == 0:
#             ckpt_path = os.path.join(ckpt_dir, f"checkpoint-{steps}")
#             os.makedirs(ckpt_path, exist_ok=True)
#             torch.save(model.state_dict(), os.path.join(ckpt_path, "model.pt"))
#         steps += 1

#         progress_bar.update(1)

# torch.save(model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
