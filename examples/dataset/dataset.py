import datasets
import torch
import json
from tqdm import tqdm
import chromadb  # 假设您已经安装了chromadb库
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


def gte(path,connection_name,device="cuda:0",s=None,e=None,batch_size=512):
    
    model_name = "Alibaba-NLP/gte-modernbert-base"
    model = SentenceTransformer(model_name, trust_remote_code=True, device=device).to(torch.bfloat16).eval()
    # In case you want to reduce the maximum length:
    model.max_seq_length = 8192
    
    json_data = json.load(open(path, "r"))
    if s is not None and e is not None:
        json_data = json_data[s:e]
    # 初始化chromadb数据库
    client = chromadb.PersistentClient()
    collection = client.get_or_create_collection(connection_name,metadata={"hnsw:space": "cosine"})
    for index in tqdm(range(0, len(json_data), batch_size)):
        # 批量处理
        batch_texts = [item['text'] for item in json_data[index:index + batch_size]]
        batch_ids = [item['id'] for item in json_data[index:index + batch_size]]
        # pass
        embeddings = model.encode(batch_texts)  # 批量计算embedding
        
        collection.add(
            ids=batch_ids, 
            embeddings=embeddings,
            documents=batch_texts,
        )  # 将每个向量独立添加到chromadb

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


# def edit_data(path):
#     json_data = json.load(open(path, "r"))
    
#     results = []  # 用于存储结果

#     def compute_distance(item1, item2):
#         id1, id2 = item1["id"], item2["id"]
#         return {
#             "id1": id1,
#             "id2": id2,
#             "distance": levenshtein_distance(item1["text"], item2["text"])
#         }

#     # 使用进程池加速计算
#     with concurrent.futures.ProcessPoolExecutor(max_workers=80) as executor:
#         futures = []
#         total_comparisons = len(json_data) * (len(json_data) - 1) // 2  # 计算总比较次数
#         # 使用tqdm显示进度
#         for i, item1 in tqdm(enumerate(json_data), total=len(json_data), desc="Submitting tasks: "):
#             for item2 in json_data[i + 1:]:  # 只与后面的项计算
#                 futures.append(executor.submit(compute_distance, item1, item2))

#         # 使用tqdm显示进度
#         while futures:  # 外部循环检查futures是否为空
#             for future in futures[:]:  # 内部循环检查每个future
#                 if future.done():  # 如果计算已完成
#                     results.append(future.result())  # 收集结果
#                     futures.remove(future)  # 从futures中移除已完成的future
#             # 更新进度条
#             tqdm.write(f"Completed: {len(results)} / {total_comparisons}")

#     # 将结果记录到新的json中
#     with open("examples/bench_cache/data/wild_chat_levenshtein_results.json", "w") as f:
#         json.dump(results, f)
        


def search_data(connection_name, path, save_path,device="cuda:0"):
    json_data = json.load(open(path, "r"))
    model_name="Alibaba-NLP/gte-modernbert-base"
    model = SentenceTransformer(model_name, trust_remote_code=True, device=device).to(torch.bfloat16).eval()
    model.max_seq_length = 8192
    client = chromadb.PersistentClient()
    collection = client.get_collection(connection_name)

    
    num_sim = [0,0,0,0,0]
    num_same =[0,0,0,0,0]
    import random
    # 随机在json_data中选择10000条数据
    items = random.sample(json_data, 20000)
    # items = json_data
    # print(len(items))
    save_data = []

    
    for item in tqdm(items, desc="Processing items", total=len(items)):  # 添加进度条
        embeddings = model.encode(item["text"])
        cache_hash_list = []
        if embeddings is None:
            continue
        threshold = 0.05
        query_result = collection.query(
            query_embeddings=[embeddings],
            n_results=6,
            include=["embeddings","documents","distances"]  # 确保包含嵌入向量
        )
        distance = query_result["distances"][0]
        doc = query_result["documents"][0]
        
        sim_count = 0
        for i in range(0,5):
            try:
                hash_value = hashlib.sha256(doc[i].encode()).hexdigest()
                if hash_value not in cache_hash_list and distance[i] < threshold and i > 0:
                    sim_count += 1
                    num_sim[sim_count-1] += 1
                    if doc[i] == item["text"]:
                        num_same[sim_count-1] += 1
                    save_data.append({
                        "id1": item["id"],
                        "text1": item["text"],
                        "text2": doc[i],
                        "distance": distance[i],
                    })
                    cache_hash_list.append(hash_value)
            except:
                continue
        
    json.dump(save_data, open(save_path, "w"),indent=4)
    print("total: ", len(items))
    print(f"num_sim_one: {num_sim[0]}")
    print(f"num_sim_two: {num_sim[1]}")
    print(f"num_sim_three: {num_sim[2]}")
    print(f"num_sim_four: {num_sim[3]}")
    print(f"num_sim_more: {num_sim[4]}")
    
    print(f"num_same_one: {num_same[0]}")
    print(f"num_same_two: {num_same[1]}")
    print(f"num_same_three: {num_same[2]}")
    print(f"num_same_four: {num_same[3]}")
    print(f"num_same_more: {num_same[4]}")

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
    from data.ppl import PerplexityMetric
    ppl_metric = PerplexityMetric(model_id="Qwen/Qwen2.5-0.5B-Instruct", device=device)
    json_data = json.load(open(path, "r"))
    new_json_data = []
    batch_size = 16  # 设置批处理大小
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
    json.dump(new_json_data, open(new_path, "w"), indent=4)

if __name__ == "__main__":

    # wild_chat()
    # gte("examples/bench_cache/data/wild_chat.json")
    # edit_data("examples/bench_cache/data/wild_chat.json")
    
    # sharegpt52k()
    # lmsys_chat()
    
    connection_name = "wild_chat_embeddings"
    # gte("examples/bench_cache/data/wild_chat.json",connection_name,device="cuda:0")
    search_data(connection_name,"examples/dataset/data/wild_chat_ppl.json","examples/dataset/data/wild_chat_sim.json",device="cuda:1")
    
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