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
    
def process_opus_dataset(save_path,config):
    dataset_path = "Helsinki-NLP/opus-100"
    tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    # dataset_embeder = DatasetEmbeder(model_name,collection_name=collection_name,database_path=database_path)
    # dataset = json.load(open(dataset_path, "r"))
    global_id = 1
    validation_dataset = list(datasets.load_dataset(dataset_path,config,split="validation"))
    test_dataset = list(datasets.load_dataset(dataset_path,config,split="test"))
    train_dataset = list(datasets.load_dataset(dataset_path,config,split="train"))
    dataset = validation_dataset + train_dataset + test_dataset
    
    save_data = []
    for item in tqdm(dataset):
        source_text = item["translation"][config.split("-")[0]]
        target_text = item["translation"][config.split("-")[1]]
        # embeddings = dataset_embeder.embed_text(source_text)
        source_token_ids = tokenizer.encode(source_text, add_special_tokens=False)
        target_token_ids = tokenizer.encode(target_text, add_special_tokens=False)
        if len(source_token_ids) < 50:
            continue
        data = {
            "id": global_id,
            "translation": item["translation"],
        }
        global_id += 1
        save_data.append(data)
    json.dump(save_data, open(save_path, "w"), ensure_ascii=False, indent=4)
    
def embed_opus_dataset(dataset_path: str, config: str, batch_size: int = 32):
    """批量计算embedding并插入数据库
    
    Args:
        dataset_path: 数据集路径
        config: 语言配置，如 "en-zh"
        batch_size: 批处理大小
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    dataset = json.load(open(dataset_path, "r"))
    dataset_embedder = DatasetEmbeder(
        model_name="all-MiniLM-L6-v2",
        collection_name="opus",
        database_path="examples/dataset/data/database/opus.db"
    )
    
    # 初始化批处理列表
    batch_items = []
    batch_texts = []
    global_id = 1
    
    for item in tqdm(dataset, desc="Processing documents"):
        source_text = item["translation"][config.split("-")[0]]
        
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
                        "translation": item["translation"]
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
    
    # 处理最后一个不完整的批次
    if batch_texts:
        try:
            batch_embeddings = dataset_embedder.embed_text(batch_texts)
            batch_data = []
            for idx, (item, embedding) in enumerate(zip(batch_items, batch_embeddings)):
                item_data = {
                    "id": item["id"],
                    "vector": embedding.tolist(),
                    "translation": item["translation"]
                }
                batch_data.append(item_data)
            dataset_embedder.insert_dataset(batch_data)
        except Exception as e:
            print(f"Error processing final batch: {e}")
    
    print(f"Processed {global_id} documents in total")

def find_opus_similar_docs(dataset_path: str, save_path: str):
    dataset = json.load(open(dataset_path, "r"))
    client = MilvusClient("/root/code/vllm_plus/examples/dataset/data/database/opus.db")
    collection_name = "opus"
    model = SentenceTransformer("all-MiniLM-L6-v2",device="cuda:0")
    save_data = {
        "all_translations": {},
        "similar_pairs": {}
    }
    # dataset = random.sample(dataset,min(len(dataset),500))
    for item in tqdm(dataset):
        source_text = item["translation"][config.split("-")[0]]
        embeddings = model.encode(source_text)
        results = client.search(collection_name=collection_name,data=[embeddings],limit=50,output_fields=["translation"])
        if len(results[0]) > 0:
            saved_item = {
                "id": item["id"],
                "similar_items": []
            }
            if item["id"] not in save_data["all_translations"]:
                save_data["all_translations"][item["id"]] = item["translation"]
            for result in results[0]:
                saved_item["similar_items"].append({
                    "id": result["id"],
                    "similarity": result["distance"],
                })
                if result["id"] not in save_data["all_translations"]:
                    save_data["all_translations"][result["id"]] = result["entity"]["translation"]
            save_data["similar_pairs"][item["id"]] = saved_item
    json.dump(save_data, open(save_path, "w"), ensure_ascii=False, indent=4)
    
def find_opus_similar_docs_topk(dataset_path: str, save_path: str):
    dataset = json.load(open(dataset_path, "r"))
    from edit2 import find_text_differences
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct",local_files_only=True)
    data = json.load(open(dataset_path,"r"))
    all_translations = data["all_translations"]
    similar_pairs = data["similar_pairs"]
    save_data = []
    # similar_pairs = random.sample(similar_pairs,min(len(similar_pairs),500))
    for id,item in tqdm(similar_pairs.items()):
        save_items = []
        for similar_item in item["similar_items"]:
            source_tokens = tokenizer.encode(all_translations[str(similar_item["id"])]["en"])
            target_tokens = tokenizer.encode(all_translations[str(item["id"])]["en"])
            diff_report = find_text_differences(source_tokens,target_tokens)
            reused_token_num = 0
            for move in diff_report["moves"]:
                reused_token_num += len(move["to_position"])
            similar_item["reused_token_num"] = reused_token_num
            save_items.append(similar_item)

        # 选择similarity top5保存
        new_item = {
            "id": id,
            "cosine_similarity_top5": sorted(save_items,key=lambda x: x["similarity"],reverse=True)[1:6],
            "reused_token_num_top5": sorted(save_items,key=lambda x: x["reused_token_num"],reverse=True)[1:6]
        }
        save_data.append(new_item)
        
    json.dump({
        "all_translations": all_translations,
        "similar_pairs": save_data
    }, open(save_path, "w"), ensure_ascii=False, indent=4)
    
    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = "en-zh"
    dataset_path = f"examples/dataset/data/opus/opus_dataset_{config}.json"
    # process_opus_dataset(save_path=dataset_path, config=config)
    # embed_opus_dataset(dataset_path=dataset_path, config=config, batch_size=64)
    
    similar_path = f"examples/dataset/data/opus/opus_dataset_{config}_similar_docs_top50.json"
    save_path = f"examples/dataset/data/opus/opus_dataset_{config}_similar_docs_top50_test1.json"
    # find_opus_similar_docs(dataset_path=dataset_path, save_path=save_path)
    find_opus_similar_docs_topk(dataset_path=similar_path, save_path=save_path)
