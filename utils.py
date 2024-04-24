import torch.nn as nn


def get_model_size(model: nn.Module):
    """
    获取模型参数量
    """
    return sum(p.numel() for p in model.parameters())


def data_collator_closure(pad_token_id, no_attention=0):
    """
    数据收集器闭包
    """

    def batch_padding(batch):

        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]
        input_ids = nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=pad_token_id
        )
        attention_mask = nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=no_attention
        )
        labels = nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=pad_token_id
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return batch_padding


if __name__ == "__main__":
    import torch

    tensor_list = [torch.rand(3, 4), torch.rand(2, 4), torch.rand(5, 4)]
    pad_sequence = nn.utils.rnn.pad_sequence(
        tensor_list, batch_first=True, padding_value=0
    )
    print(pad_sequence)
