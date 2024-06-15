import json
import os
import random
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CustomConfig
from modeling_custom import CustomForCausalLM
from tokenization_custom import CustomTokenizer
from utils import get_model_size, SupervisedDataset

SEED = 42


def set_seed(seed: int):
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed=seed)
    np.random.seed(seed=seed)
    random.seed(seed)


def get_lr_warmup(warmup_steps: int):

    def lr_warmup(current_step: int):
        return float(current_step) / float(max(1, warmup_steps))

    return lr_warmup


@dataclass
class TrainingArgs:
    output_dir: str
    logging_steps: int = 500
    saving_steps: int = 500
    batch_size: int = 1
    epochs: int = 3
    lr: float = 1e-4
    weight_decay: float = 1e-4
    max_norm: float = 1.0
    warm_up_ratio: float = 0.1
    gradient_checkpointing: bool = False
    gradient_accumulation_steps: int = 24


def train(
    model: nn.Module,
    args: TrainingArgs,
    dataset: Dataset,
    device: Optional[Union[str, torch.device]] = None,
    data_collator=None,
):
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=8,
    )
    # 完整的有效步
    complete_steps_per_epoch = len(data_loader) // args.gradient_accumulation_steps
    # 不完整的有效步，最后剩余的小批量
    last_mini_steps = len(data_loader) % args.gradient_accumulation_steps
    # 一个 epoch 等效步
    if last_mini_steps != 0:
        steps_per_epoch = complete_steps_per_epoch + 1
    else:
        steps_per_epoch = complete_steps_per_epoch

    total_steps = steps_per_epoch * args.epochs

    # 优化器
    optimizer = AdamW(
        params=model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # 学习率调度
    warmup_steps = int(total_steps * args.warm_up_ratio)
    cosine_steps = total_steps - warmup_steps
    warmup_scheduler = LambdaLR(
        optimizer=optimizer, lr_lambda=get_lr_warmup(warmup_steps=warmup_steps)
    )
    cosine_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=cosine_steps)
    scheduler = SequentialLR(
        optimizer=optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    # 设备
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.output_dir, exist_ok=True)

    model = model.to(device=device)
    if args.gradient_checkpointing:
        model.enable_gradient_checkpoint()
    loggin_info = []
    current_step = 0

    progress_bar = tqdm(range(total_steps))
    scaler = GradScaler()
    for epoch in range(args.epochs):
        current_loss = 0.0
        for idx, batch in enumerate(data_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            if last_mini_steps == 0 or len(data_loader) - (idx + 1) > last_mini_steps:
                current_accumulation = args.gradient_accumulation_steps
            else:
                current_accumulation = last_mini_steps

            with autocast(dtype=torch.bfloat16):
                outputs = model(**batch)
                logits, loss = outputs
                loss /= current_accumulation
            current_loss += loss.item()
            # 反向传播
            scaler.scale(loss).backward()

            if (idx + 1) % args.gradient_accumulation_steps == 0 or (idx + 1) == len(
                data_loader
            ):
                # 梯度裁剪
                scaler.unscale_(optimizer=optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=args.max_norm
                )

                # 梯度更新
                scaler.step(optimizer=optimizer)

                # 更新缩放因子
                scaler.update()

                # 学习率更新
                scheduler.step()

                # 清除梯度
                optimizer.zero_grad()

                progress_bar.update(1)

                current_step += 1
                if current_step % args.logging_steps == 0:
                    current_epochs = current_step / steps_per_epoch
                    info = {
                        "Epoch": f"{current_epochs:.2f}/{args.epochs}",
                        "Step": f"{current_step}/{total_steps}",
                        "Loss": current_loss,
                        "LR": scheduler.get_last_lr()[0],
                    }
                    loggin_info.append(info)
                    print(info)

                if current_step % args.saving_steps == 0:
                    ckpt_path = os.path.join(
                        args.output_dir,
                        f"checkpoint-{current_step}.pt",
                    )
                    torch.save(model.state_dict(), ckpt_path)

                current_loss = 0.0

    ckpt_path = os.path.join(
        args.output_dir,
        "last.pt",
    )
    torch.save(model.state_dict(), ckpt_path)
    with open("logging.jsonl", "w", encoding="utf-8") as fw:
        for logging_data in loggin_info:
            fw.write(json.dumps(logging_data) + "\n")


if __name__ == "__main__":
    set_seed(SEED)

    tokenizer = CustomTokenizer.from_pretrained("tokenizer")
    config = CustomConfig(
        vocab_size=len(tokenizer.get_vocab()),
        max_position_embeddings=2048,
        hidden_size=4096,
        intermediate_size=16384,
        num_hidden_layers=3,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = CustomForCausalLM(config)
    print(f"Model size is {get_model_size(model)}")

    dataset = SupervisedDataset("alpaca_data.json", tokenizer, max_len=2048)

    args = TrainingArgs(
        output_dir="result",
        gradient_checkpointing=True,
        batch_size=4,
        logging_steps=50,
        warm_up_ratio=0.03,
        epochs=1,
        gradient_accumulation_steps=8,
        lr=1e-3,
        weight_decay=1e-5,
    )

    train(model=model, args=args, dataset=dataset)
