import time
from sentence_transformers import SentenceTransformer
from vllm import LLM
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"

def benchmark_embedding(model:SentenceTransformer,text:str,rounds:int=200):
    tokens = model.tokenize(text)
    # warm up
    for _ in range(10):
        model.encode(text)
    
    avg_time = []
    # benchmark
    for i in range(rounds):
        start_time = time.time()
        model.encode(text)
        end_time = time.time()
        avg_time.append(end_time - start_time)
    print(f"token num: {tokens['input_ids'].shape[0]}, avg embed time: {sum(avg_time) / len(avg_time)} seconds")

def benchmark_prefill(model_name:str,model:LLM,text:str,rounds:int=200):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer.encode(text)
    for _ in range(10):
        model.generate(text,use_tqdm=False)
    avg_time = []
    for i in range(rounds):
        output= model.generate(text,use_tqdm=False)
        ttft = output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time
        avg_time.append(ttft)
    print(f"token num: {len(tokens)}, avg prefill time: {sum(avg_time) / len(avg_time)} seconds")
        

def test_embedding():
    # model_name = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
    # model_name = "Linq-AI-Research/Linq-Embed-Mistral"
    # model_name = "BAAI/bge-m3"
    # model_name = "all-MiniLM-L6-v2"
    model_name = "Alibaba-NLP/gte-Qwen2-7B-instruct"
    model = SentenceTransformer(model_name,device="cuda:0")
    texts = [
        "Hello," * 10,
        "Hello," * 20,
        "Hello," * 50,
        "Hello," * 60,
        "Hello," * 70,
        "Hello," * 80,
        "Hello," * 90,
        "Hello," * 100,
        "Hello," * 200,
        "Hello," * 300,
        "Hello," * 400,
        "Hello," * 500,
        "Hello," * 600,
        "Hello," * 700,
        "Hello," * 800,
        "Hello," * 900,
    ]
    for text in texts:
        benchmark_embedding(model,text,rounds=100)


def test_prefill():
    # model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    model_name = "/root/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
    model = LLM(model=model_name,device="cuda:0",disable_async_output_proc=True,dtype="bfloat16",multi_step_stream_outputs=True,gpu_memory_utilization=0.8,max_model_len=2048)
    texts = [
        "Hello," * 10,
        "Hello," * 20,
        "Hello," * 50,
        "Hello," * 60,
        "Hello," * 70,
        "Hello," * 80,
        "Hello," * 90,
        "Hello," * 100,
        "Hello," * 200,
        "Hello," * 300,
        "Hello," * 400,
        "Hello," * 500,
        "Hello," * 600,
        "Hello," * 700,
        "Hello," * 800,
        "Hello," * 900,
    ]
    for text in texts:
        benchmark_prefill(model_name,model,text)

if __name__ == "__main__":
    test_embedding()
    # test_prefill()