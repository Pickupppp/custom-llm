from tokenizers import Tokenizer
from tokenizers import normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, Digits
from tokenizers.normalizers import NFD, StripAccents, Lowercase
from tokenizers.processors import TemplateProcessing


# 初始化 tokenizer
tokenizer = Tokenizer(BPE(unk_token="<unk>"))

# 规范化
normalizer = normalizers.Sequence([NFD(), StripAccents(), Lowercase()])
tokenizer.normalizer = normalizer

# 设置预分词器
pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Digits(individual_digits=True)])
tokenizer.pre_tokenizer = pre_tokenizer

# 设置分词模型
trainer = BpeTrainer(special_tokens=["<s>", "</s>", "<pad>", "<unk>"])

# # 后处理
# tokenizer.post_processor = TemplateProcessing(
#     single="<s> $A </s>",
#     special_tokens={
#         "<s>": tokenizer.token_to_id("<s>"),
#         "</s>": tokenizer.token_to_id("</s>"),
#         "<pad>": tokenizer.token_to_id("<pad>"),
#     },
# )

# 训练 tokenizer
tokenizer.train(files=["data/train.csv", "data/test.csv"], trainer=trainer)

# 保存 tokenizer
tokenizer.save("tokenizer.json")
