from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
from datasets import load_dataset, Dataset

tokenizer = Tokenizer(models.BPE())
tokenizer.normalizer = normalizers.NFC()
tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
    [
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False),
    ]
)
trainer = trainers.BpeTrainer(
    vocab_size=65535,
    special_tokens=["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"],
    min_frequency=1500,
)

tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
tokenizer.decoder = decoders.ByteLevel()

dataset: Dataset = load_dataset(
    "json",
    data_files=[
        "nlp_datas/part-000020-a894b46e.jsonl.tar.gz",
        "nlp_datas/part-000065-a894b46e.jsonl.tar.gz",
    ],
    split="train",
)


def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["content"]


tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)


# 保存 tokenizer
tokenizer.save("tokenizer_new.json")
