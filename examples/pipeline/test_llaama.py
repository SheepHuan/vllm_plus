import transformers

tokenizer = transformers.AutoTokenizer.from_pretrained("/root/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct")

messages = [
  {"role": "system", "content": "You are a bot that responds to weather queries."},
  {"role": "user", "content": "Hey, what's the temperature in Paris right now?"}
]

inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
print(tokenizer.decode(inputs))