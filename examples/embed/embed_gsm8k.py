import datasets
import transformers
from tqdm import tqdm
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import random
from embed_dataset import DatasetEmbeder
from sentence_transformers import SentenceTransformer
from pymilvus import connections,MilvusClient
from transformers import AutoTokenizer
import os
    
def process_gsm8k_dataset(save_path):
    dataset_path = "openai/gsm8k"
    tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    main_train_dataset = list(datasets.load_dataset(dataset_path,"main",split="train"))
    main_test_dataset = list(datasets.load_dataset(dataset_path,"main",split="test"))
    socratic_train_dataset = list(datasets.load_dataset(dataset_path,"socratic",split="train"))
    socratic_test_dataset = list(datasets.load_dataset(dataset_path,"socratic",split="test"))
    dataset = main_train_dataset + main_test_dataset + socratic_train_dataset + socratic_test_dataset
    
    save_data = []
    global_id = 1
    for item in tqdm(dataset):
        # item["id"] = global_id
        source_text = item["question"]
        source_token_ids = tokenizer.encode(source_text, add_special_tokens=False)
        if len(source_token_ids) < 10:
            continue
        data = {
            "id": global_id,
            "question": item["question"],
            "answer": item["answer"]
        }
        global_id += 1
        save_data.append(data)
    json.dump(save_data, open(save_path, "w"), ensure_ascii=False, indent=4)
    
def embed_gsm8k_dataset(dataset_path):
    tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    dataset = json.load(open(dataset_path, "r"))
    dataset_embedder = DatasetEmbeder(
        model_name="all-MiniLM-L6-v2",
        collection_name="gsm8k",
        database_path="examples/dataset/data/database/gsm8k.db"
    )
     # 初始化批处理列表
    batch_items = []
    batch_texts = []
    global_id = 1
    batch_size = 64
    for item in tqdm(dataset, desc="Processing documents"):
        source_text = item["question"]
        
        # 添加到当前批次
        batch_items.append(item)
        batch_texts.append(source_text)
        
        # 当达到批处理大小时，处理当前批次
        if len(batch_texts) >= batch_size:
            try:
                # 批量计算embeddings
                batch_embeddings = dataset_embedder.embed_text(batch_texts)
                
                # 准备批量插入数据
                batch_data = []
                for idx, (item, embedding) in enumerate(zip(batch_items, batch_embeddings)):
                    item_data = {
                        "id": item["id"],
                        "vector": embedding.tolist(),  # 确保embedding是列表格式
                        "question": item["question"],
                        "answer": item["answer"]
                    }
                    batch_data.append(item_data)
                
                # 批量插入数据库
                dataset_embedder.insert_dataset(batch_data)
                
                # 更新global_id
                global_id += len(batch_items)
                
                # 清空批处理列表
                batch_items = []
                batch_texts = []
                
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue

def find_gsm8k_similar_docs(dataset_path,save_path):
    from edit2 import find_text_differences
    dataset = json.load(open(dataset_path, "r"))
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct",local_files_only=True)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = MilvusClient("/root/code/vllm_plus/examples/dataset/data/database/gsm8k.db")
    collection_name = "gsm8k"
    save_data = {
        "all_data": {},
        "similar_pairs": {}
    }
    for item in tqdm(dataset):
        source_text = item["question"]
        embeddings = model.encode(source_text)
        results = client.search(collection_name=collection_name,data=[embeddings],limit=50,output_fields=["question","answer"])
        if len(results[0]) > 0:
            saved_item = {
                "id": item["id"],
                "similar_items": []
            }
            for result in results[0]:
                reused_token_num = 0
                
                source_tokens = tokenizer.encode(result["entity"]["question"])
                target_tokens = tokenizer.encode(item["question"])
                diff_report = find_text_differences(source_tokens,target_tokens)
                
                for move in diff_report["moves"]:
                    reused_token_num += len(move["to_position"])
                for move in diff_report["moves"]:
                    reused_token_num += len(move["to_position"])
                saved_item["similar_items"].append({
                    "id": result["id"],
                    "similarity": result["distance"],
                    "reused_token_num": reused_token_num
                })
                if result["id"] not in save_data["all_data"]:
                    save_data["all_data"][result["id"]] = {
                        "question": result["entity"]["question"],
                        "answer": result["entity"]["answer"]
                    }
            save_data["all_data"][item["id"]] = {
                "question": item["question"],
                "answer": item["answer"]
            }
            # saved_item["similar_items"] 排序,选择similarity top5保存
            new_item = {
                "id": item["id"],
                "cosine_similarity_top5": sorted(saved_item["similar_items"],key=lambda x: x["similarity"],reverse=True)[1:6],
                "reused_token_num_top5": sorted(saved_item["similar_items"],key=lambda x: x["reused_token_num"],reverse=True)[1:6]
            }
            save_data["similar_pairs"][item["id"]] = new_item
            
            
    json.dump(save_data, open(save_path, "w"), ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # process_gsm8k_dataset(save_path="examples/dataset/data/gsm8k/gsm8k_dataset.json")
    # embed_gsm8k_dataset(dataset_path="examples/dataset/data/gsm8k/gsm8k_dataset.json")
    
    dataset_path = "examples/dataset/data/gsm8k/gsm8k_dataset.json"
    save_path = "examples/dataset/data/gsm8k/gsm8k_dataset_similar_docs_top5.json"
    find_gsm8k_similar_docs(dataset_path=dataset_path,save_path=save_path)