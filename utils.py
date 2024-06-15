import json
import os
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizer

IGNORE_TOKEN_ID = -100


def _preprocess(
    source: Dict, tokenizer: PreTrainedTokenizer, max_len: int
) -> Dict[str, torch.Tensor]:
    system_message = "You are a helpful assistant."
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("<|system|>").input_ids
    _user = tokenizer("<|user|>").input_ids
    _assistant = tokenizer("<|assistant|>").input_ids
    _end = tokenizer("<|end|>").input_ids

    input_ids, labels = [], []

    # 系统指令
    system = _system + tokenizer(system_message).input_ids + _end + nl_tokens
    input_ids += system
    labels += _system + [IGNORE_TOKEN_ID] * (len(system) - 3) + _end + nl_tokens
    assert len(input_ids) == len(labels)

    # 输入指令
    if source["input"] != "":
        _input_ids = (
            _user
            + tokenizer(source["instruction"]).input_ids
            + nl_tokens
            + tokenizer(source["input"]).input_ids
            + _end
            + nl_tokens
        )
    else:
        _input_ids = (
            _user + tokenizer(source["instruction"]).input_ids + _end + nl_tokens
        )
    _labels = _user + [IGNORE_TOKEN_ID] * (len(_input_ids) - 3) + _end + nl_tokens
    input_ids += _input_ids
    labels += _labels
    assert len(input_ids) == len(labels)

    # 输出
    _output = _assistant + tokenizer(source["output"]).input_ids + _end
    input_ids += _output
    labels += _output
    assert len(input_ids) == len(labels)

    attention_mask = [1] * len(input_ids)

    if len(input_ids) < max_len:
        diff = max_len - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * diff
        labels += [IGNORE_TOKEN_ID] * diff
        attention_mask += [0] * diff

    input_ids = input_ids[:max_len]
    labels = labels[:max_len]
    attention_mask = attention_mask[:max_len]

    return dict(
        input_ids=torch.tensor(input_ids, dtype=torch.int),
        attention_mask=torch.tensor(attention_mask, dtype=torch.int),
        labels=torch.tensor(labels, dtype=torch.int),
    )


class SupervisedDataset(Dataset):
    def __init__(self, path, tokenizer: PreTrainedTokenizer, max_len: int):
        """
        Args:
            path (str): 路径可以是一个文件或者一个包含JSON文件的目录。
            transform (callable, optional): 一个用于进行数据转换的可选函数。
        """
        self.path = path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.cache = {}
        self.data = []

        # 检查路径是文件还是目录
        if os.path.isfile(path):
            self._load_file(path)
        elif os.path.isdir(path):
            for filename in os.listdir(path):
                if filename.endswith(".json"):
                    file_path = os.path.join(path, filename)
                    self._load_file(file_path)

    def _load_file(self, file_path: str):
        """
        从指定的文件路径加载JSON数据。
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                self.data.extend(data)
            elif isinstance(data, dict):
                self.data.append(data)

    def __len__(self):
        """
        返回数据集中的样本数量。
        """
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        返回索引idx对应的数据项。
        """
        if idx in self.cache:
            return self.cache[idx]

        res = _preprocess(self.data[idx], self.tokenizer, self.max_len)
        self.cache[idx] = res
        return res


def get_model_size(model: nn.Module):
    """
    获取模型参数量
    """
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    import torch

    tensor_list = [torch.rand(3, 4), torch.rand(2, 4), torch.rand(5, 4)]
    pad_sequence = nn.utils.rnn.pad_sequence(
        tensor_list, batch_first=True, padding_value=0
    )
    print(pad_sequence)
