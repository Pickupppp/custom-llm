from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer.json")
print(tokenizer.get_vocab_size())

sentence = "Hello, how are you?我很好"
outputs = tokenizer.encode(sentence)
print(outputs.tokens)
print(outputs.ids)
print(outputs.type_ids)