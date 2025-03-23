import datasets
import torch
import json
from tqdm import tqdm
# import chromadb  # 假设您已经安装了chromadb库
import uuid
import time
import concurrent.futures  # 导入concurrent.futures模块
from sentence_transformers import SentenceTransformer
import hashlib
import numpy as np  # 导入numpy库
from transformers import AutoTokenizer
from concurrent.futures import ProcessPoolExecutor
import os
from functools import partial

def generate_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()  # 使用SHA-256生成哈希

def wild_chat():
    dataset_name = "allenai/WildChat-1M"
    
    def process_item(dataset):
        json_data =[]
        start_index = 0  # 设置开始节点
        for index, item in tqdm(enumerate(dataset), total=len(dataset)):
            if index < start_index:
                continue  # 跳过已处理的项
            # 假设item是句子
            conversation = item['conversation']
            for i in conversation:
                role = i['role']
                if role == "user":
                    text = i['content']
                    unique_id = str(uuid.uuid4())[:8] + str(time.time())  # 生成UUID并结合当前时间戳
                    json_data.append({
                        "id": generate_hash(unique_id),
                        "text": text,
                            "hash": item['conversation_hash']
                        })
        return json_data
    dataset = datasets.load_dataset(dataset_name, split="train")
    data = process_item(dataset)
    # dataset = datasets.load_dataset(dataset_name, split="test")
    json.dump(data, open("examples/bench_cache/data/wild_chat.json", "w"))


def lccc():
    dataset_name = "thu-coai/lccc"
    
    def process_item(dataset):
        json_data =[]
        start_index = 0  # 设置开始节点
        for index, item in tqdm(enumerate(dataset), total=len(dataset)):
            if index < start_index:
                continue  # 跳过已处理的项
            # 假设item是句子
            conversation = item['dialog']
            for i in conversation:
                role = i['role']
                if role == "user":
                    text = i['content']
                    unique_id = str(uuid.uuid4())[:8] + str(time.time())  # 生成UUID并结合当前时间戳
                    json_data.append({
                        "id": generate_hash(unique_id),
                        "text": text,
                            "hash": item['conversation_hash']
                        })
        return json_data
    dataset = datasets.load_dataset(dataset_name, "large",trust_remote_code=True,split="train")
    data = process_item(dataset)
    # dataset = datasets.load_dataset(dataset_name, split="test")
    json.dump(data, open("examples/bench_cache/data/lccc.json", "w"))

def belle():
    dataset_name = "BelleGroup/multiturn_chat_0.8M"
    
    def process_item(dataset):
        json_data =[]
        start_index = 0  # 设置开始节点
        for index, item in tqdm(enumerate(dataset), total=len(dataset)):
            instruction = item['instruction']
            ins_items = instruction.split("\n")
            for ins in ins_items:
                if "Human" in ins:
                    text = ins.replace("Human","")
                    text = text.replace(":","")
                    text = text.replace("：","")
                    text = text.strip()
                    unique_id = str(uuid.uuid4())[:8] + str(time.time())  # 生成UUID并结合当前时间戳
                    json_data.append({
                        "id": generate_hash(unique_id),
                        "text": text,
                    })
        return json_data
    dataset = datasets.load_dataset(dataset_name,trust_remote_code=True,split="train")
    data = process_item(dataset)
    # dataset = datasets.load_dataset(dataset_name, split="test")
    json.dump(data, open("examples/bench_cache/data/belle.json", "w"),indent=4,ensure_ascii=False)

def sharegpt52k():
    dataset_name = "liyucheng/ShareGPT90K"
    # dataset = datasets.load_dataset(dataset_name,split="train")
    
    def process_item(dataset):
        json_data =[]
        start_index = 0  # 设置开始节点
        for index, item in tqdm(enumerate(dataset),total=len(dataset)):
            if index < start_index:
                continue  # 跳过已处理的项
            # 假设item是句子
            conversation = item['conversations']
            for i in range(len(conversation["value"])):
                if conversation["from"][i] == "human":
                    text = conversation["value"][i]
                    unique_id = str(uuid.uuid4())[:8] + str(time.time())  # 生成UUID并结合当前时间戳
                    json_data.append({
                        "id": generate_hash(unique_id),
                        "text": text,
                        "hash": item['id']
                    })
        return json_data
    dataset = datasets.load_dataset(dataset_name,split="train")
    data1 = process_item(dataset)
    # dataset = datasets.load_dataset(dataset_name,split="test")
    # data2 = process_item(dataset)
    # dataset = datasets.load_dataset(dataset_name,split="valid")
    # data3 = process_item(dataset)
    # data = data1 + data2 + data3
    json.dump(data1, open("examples/bench_cache/data/sharegpt90k.json", "w"),indent=4)

def load_jsonl(file_path):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    obj = json.loads(line)
                    data.append(obj)
                except json.JSONDecodeError:
                    print(f"解析 JSON 时出错，跳过行: {line}")
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 未找到。")
    except Exception as e:
        print(f"发生未知错误: {e}")
    return data

def instruction_wildv2():
    import glob
    data_path = glob.glob("examples/bench_cache/data/ins_wildv2/*.jsonl")
    save_data = []
    for path in data_path:
        # 如何加载jsonl文件
        data = load_jsonl(path)
        for item in data:
            text = item['instruction']
            unique_id = str(uuid.uuid4())[:8] + str(time.time())  # 生成UUID并结合当前时间戳
            save_data.append({
                "id": generate_hash(unique_id),
                "text": text,
            })
    json.dump(save_data, open("examples/bench_cache/data/instruction_wildv2_ppl.json", "w"),indent=4)
    

def lmsys_chat():
    dataset_name = "lmsys/lmsys-chat-1m"
    
    
    start_index = 0
    def process_item(dataset):
        json_data =[]
        # 设置开始节点
        for index, item in tqdm(enumerate(dataset),total=len(dataset)):
            if index < start_index:
                continue  # 跳过已处理的项
            if item['text'] == '':
                continue
            unique_id = str(uuid.uuid4())[:8] + str(time.time())  # 生成UUID并结合当前时间戳
            json_data.append({
                "id": generate_hash(unique_id),
                "text": item['text'],
                "hash": ''
            })
        return json_data
    dataset = datasets.load_dataset(dataset_name,'wikitext-103-v1',split="train")
    data1 = process_item(dataset)
    dataset = datasets.load_dataset(dataset_name,'wikitext-103-v1',split="test")
    data2 = process_item(dataset)
    dataset = datasets.load_dataset(dataset_name,'wikitext-103-v1',split="validation")
    data3 = process_item(dataset)
    data = data1 + data2 + data3
    
    json.dump(data, open("examples/bench_cache/data/lmsys_chat_1m.json", "w"),indent=4)

def moss():
    dataset_name = "YeungNLP/moss-003-sft-data"
    dataset = datasets.load_dataset(dataset_name,split="train")
    def process_item(dataset):
        json_data =[]
        for item in tqdm(dataset,total=len(dataset)):
            conversations = item['conversation']
            for conversation in conversations:
                if 'human' in conversation:
                    text = conversation['human']
                    unique_id = str(uuid.uuid4())[:8] + str(time.time())  # 生成UUID并结合当前时间戳
                    json_data.append({
                        "id": generate_hash(unique_id),
                        "text": text,
                    })
        return json_data
    data = process_item(dataset)
    json.dump(data, open("examples/bench_cache/data/moss_ppl.json", "w"),indent=4)

def chatbot_arena():
    dataset_name = "lmsys/chatbot_arena_conversations"
    def process_item(dataset):
        json_data =[]
        start_index = 0  # 设置开始节点
        for index, item in tqdm(enumerate(dataset),total=len(dataset)):
            if index < start_index:
                continue  # 跳过已处理的项
            conversation = item['conversation_a']
            for i in conversation:
                role = i['role']
                if role == "user":
                    text = i['content']
                    unique_id = str(uuid.uuid4())[:8] + str(time.time())  # 生成UUID并结合当前时间戳
                    json_data.append({
                        "id": generate_hash(unique_id),
                        "text": text,
                    })
        return json_data
    dataset = datasets.load_dataset(dataset_name,split="train")
    
    data = process_item(dataset)
    json.dump(data, open("examples/bench_cache/data/chatbot_arena.json", "w"),indent=4,ensure_ascii=False)



def process_chunk(chunk, tokenizer=None):
    """处理数据块的函数，移到全局作用域"""
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    new_chunk = []
    for item in chunk:
        text = item["text"]
        tokens = tokenizer.encode(text, add_special_tokens=True)
        if len(tokens) >= 10:
            new_chunk.append(item)
    return new_chunk

def clean_by_token_length(path, new_path):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    json_data = json.load(open(path, "r"))

    # 将数据分成若干个块
    num_processes = 64
    chunk_size = len(json_data) // num_processes + 1
    chunks = [json_data[i:i + chunk_size] for i in range(0, len(json_data), chunk_size)]

    new_json_data = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        # 使用functools.partial来传递tokenizer参数
        process_chunk_with_tokenizer = partial(process_chunk, tokenizer=tokenizer)
        results = executor.map(process_chunk_with_tokenizer, chunks)

    # 合并所有结果
    for result in results:
        new_json_data.extend(result)

    json.dump(new_json_data, open(new_path, "w"), indent=4)
    
def clean_by_ppl(path,new_path,device="cuda:0"):
    from ppl import PerplexityMetric
    ppl_metric = PerplexityMetric(model_id="Qwen/Qwen2.5-0.5B-Instruct", device=device)
    json_data = json.load(open(path, "r"))
    new_json_data = []
    batch_size = 512  # 设置批处理大小
    num = 0
    for i in tqdm(range(0, len(json_data), batch_size), desc="Processing clean"):
        batch = json_data[i:i + batch_size]  # 获取当前批次
        text_list = [item["text"] for item in batch]
        ppls = ppl_metric.compute(text_list,max_length=512)
        for item,ppl in zip(batch,ppls):
            item["ppl"] = ppl
            
            if ppl < 100:
                item["num"] = num
                num += 1
                new_json_data.append(item)
    json.dump(new_json_data, open(new_path, "w"), indent=4,ensure_ascii=False)

if __name__ == "__main__":

    # wild_chat()
    # gte("examples/bench_cache/data/wild_chat.json")
    # edit_data("examples/bench_cache/data/wild_chat.json")
    
    # sharegpt52k()
    # lmsys_chat()
    # belle()
    # chatbot_arena()
    # instruction_wildv2()
    moss()
    
    
    # connection_name = "wild_chat_embeddings"
    # gte("examples/bench_cache/data/wild_chat.json",connection_name,device="cuda:0")
    # search_data(connection_name,"examples/dataset/data/wild_chat_ppl.json","examples/dataset/data/wild_chat_sim.json",device="cuda:1")
    
    # connection_name = "sharegpt90k_embeddings"
    # gte("examples/bench_cache/data/sharegpt90k.json",connection_name,device="cuda:1",batch_size=2048)
    # search_data(connection_name,"examples/dataset/data/sharegpt90k_ppl.json","examples/dataset/data/sharegpt90k_sim.json",device="cuda:1")

    # connection_name = "lmsys_chat_embeddings"
    # gte("examples/bench_cache/data/lmsys_chat_1m.json",connection_name,device="cuda:1",batch_size=2048)
    # search_data(connection_name,"examples/bench_cache/data/lmsys_chat_1m.json","examples/dataset/data/lmsys_chat_1m_sim.json",device="cuda:1")
    
    # clean_by_token_length("examples/bench_cache/data/wild_chat.json","examples/dataset/data/wild_chat_token_length.json")
    # clean_by_token_length("examples/bench_cache/data/sharegpt90k.json","examples/dataset/data/sharegpt90k_token_length.json")
    # clean_by_token_length("examples/bench_cache/data/lmsys_chat_1m.json","examples/dataset/data/lmsys_chat_1m_token_length.json")
    
    # clean_by_ppl("examples/dataset/data/wild_chat_token_length.json","examples/dataset/data/wild_chat_ppl.json",device="cuda:1")
    
    # clean_by_ppl("examples/dataset/data/sharegpt90k_token_length.json","examples/dataset/data/sharegpt90k_ppl.json",device="cuda:0")
    
    # clean_by_ppl("examples/dataset/data/lmsys_chat_1m_token_length.json","examples/dataset/data/lmsys_chat_1m_ppl.json",device="cuda:0")
    
    # clean_by_ppl("examples/bench_cache/data/belle.json","examples/dataset/data/belle_ppl.json",device="cuda:0")