import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class CustomDataset(Dataset):
    def __init__(self, filename, tokenizer, max_length):
        # 初始化数据集
        self.raw_datas = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache = {}
        with open(filename, "r", encoding="utf-8") as f:
            for line in tqdm(f):
                self.raw_datas.append(line)

    def __len__(self):
        # 返回数据集的大小
        return len(self.raw_datas)

    def __getitem__(self, index):
        if index in self.cache:
            return self.cache[index]
        else:
            raw_data = self.raw_datas[index]
            outputs = self.tokenizer.encode(raw_data)
            input_ids = outputs.ids[: self.max_length]
            # if len(input_ids) < self.max_length:
            #     input_ids += [self.tokenizer.token_to_id("<|eos|>")] * (
            #         self.max_length - len(input_ids)
            #     )
            attention_mask = outputs.attention_mask[: self.max_length]
            # if len(attention_mask) < self.max_length:
            #     attention_mask += [0] * (self.max_length - len(attention_mask))
            # assert len(input_ids) >= self.max_length
            # assert len(attention_mask) >= self.max_length
            assert len(input_ids) == len(attention_mask)
            data = {
                "input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor(attention_mask),
                "labels": torch.tensor(input_ids),
            }
            self.cache[index] = data
            return data
