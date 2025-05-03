from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
 
model_name= "/root/.cache/modelscope/hub/models/01ai/Yi-34B-Chat-4bits"
 
 
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda",
    torch_dtype=torch.bfloat16
).eval()

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prompt content: "hi"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "hi"}
]

input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
print(tokenizer.decode(input_ids[0]))
output_ids = model.generate(input_ids.to('cuda'))
response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

# Model response: "Hello! How can I assist you today?"
print(response)