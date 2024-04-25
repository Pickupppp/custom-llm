import random
import torch
from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing

from config import CustomConfig
from model import CustomForCausalLM

torch.manual_seed(42)
random.seed(42)


tokenizer = Tokenizer.from_file("tokenizer.json")
tokenizer.post_processor = TemplateProcessing(
    single="<|bos|> $A <|eos|>",
    special_tokens=[
        ("<|bos|>", tokenizer.token_to_id("<|bos|>")),
        ("<|eos|>", tokenizer.token_to_id("<|eos|>")),
    ],
)
tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<|eos|>"), pad_token="<|eos|>")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = CustomConfig(
    vocab_size=tokenizer.get_vocab_size(),
    max_position_embeddings=2048,
    num_hidden_layers=3,
    pad_token_id=tokenizer.token_to_id("<|eos|>"),
)
model = CustomForCausalLM(config)
model = model.to(device)

res = model.generate("hello", tokenizer=tokenizer, max_new_tokens=10)

print(res)
print(tokenizer.decode(res))


model.load_state_dict(torch.load("xxx"))
model = model.to(device)
res = model.generate("hello", tokenizer=tokenizer, max_new_tokens=10)

print(res)
print(tokenizer.decode(res))
