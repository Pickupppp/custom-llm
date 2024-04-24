from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import ByteLevel, Digits, Punctuation, Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

# 初始化 tokenizer
tokenizer = Tokenizer(BPE())

# 规范化
normalizer = normalizers.Sequence([NFD(), StripAccents()])
tokenizer.normalizer = normalizer

# 设置预分词器
pre_tokenizer = pre_tokenizers.Sequence(
    [
        Whitespace(),
        Punctuation(),
        Digits(individual_digits=True),
        ByteLevel(),
    ]
)
tokenizer.pre_tokenizer = pre_tokenizer

tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel()


# 训练
trainer = BpeTrainer(
    special_tokens=["<|bos|>", "<|eos|>", "<|system|>", "<|user|>", "<|assistant|>"]
)


# 训练 tokenizer
tokenizer.train(files=["data/train.csv", "data/test.csv"], trainer=trainer)

# 保存 tokenizer
tokenizer.save("tokenizer.json")
