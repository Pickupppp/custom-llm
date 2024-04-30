from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast


tokenizer = Tokenizer.from_file("tokenizer.json")

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="<|bos|>",
    eos_token="<|eos|>",
    pad_token="<|eos|>",
)

# wrapped_tokenizer.save_pretrained("./test")

print(
    wrapped_tokenizer(
        ["I have a watch.", "I like eating apple and strawberry."],
        return_tensors="pt",
        padding=True,
    )
)

# print(tokenizer.get_vocab_size())

# sentence = "Hello, how are you?我很好"
# outputs = tokenizer.encode(sentence)
# print(outputs.tokens)
# print(outputs.ids)
# print(outputs.type_ids)
