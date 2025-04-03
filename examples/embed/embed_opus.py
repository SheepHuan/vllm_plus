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

def find_opus_similar_docs(dataset_path: str, save_path: str, config: str):
    """查找相似文档并计算重用token数，一次完成相似度搜索和token重用分析
    
    Args:
        dataset_path: 数据集路径
        save_path: 保存路径
        config: 语言配置，如 "en-zh"
    """
    from edit2 import find_text_differences
    
    # 初始化必要的组件
    source_documents = json.load(open(dataset_path, "r"))
    vector_db = MilvusClient("/root/code/vllm_plus/examples/dataset/data/database/opus.db")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda:0")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", local_files_only=True)
    
    # 定义源语言和目标语言
    source_lang = config.split("-")[0]
    target_lang = config.split("-")[1]
    
    result_data = {
        "all_translations": {},  # 存储所有文档的翻译
        "similar_pairs": []     # 存储相似文档对
    }
    
    for source_doc in tqdm(source_documents, desc="Processing documents"):
        try:
            # 1. 获取源文本并计算embedding
            source_text = source_doc["translation"][source_lang]
            text_embedding = embedding_model.encode(source_text)
            
            # 2. 在向量数据库中搜索相似文档
            search_results = vector_db.search(
                collection_name="opus",
                data=[text_embedding],
                limit=50,
                output_fields=["translation"]
            )
            
            if len(search_results[0]) == 0:
                continue
                
            # 3. 保存原始文档的翻译
            source_doc_id = source_doc["id"]
            if source_doc_id not in result_data["all_translations"]:
                result_data["all_translations"][source_doc_id] = source_doc["translation"]
            
            # 4. 处理相似文档
            similar_docs = []
            source_target_tokens = tokenizer.encode(source_doc["translation"][target_lang])
            
            for candidate in search_results[0]:
                candidate_id = candidate["id"]
                # 跳过自身
                if candidate_id == source_doc_id:
                    continue
                    
                # 保存相似文档的翻译
                if candidate_id not in result_data["all_translations"]:
                    result_data["all_translations"][candidate_id] = candidate["entity"]["translation"]
                
                # 计算token重用数量
                candidate_tokens = tokenizer.encode(candidate["entity"]["translation"][target_lang])
                diff_report = find_text_differences(candidate_tokens, source_target_tokens)
                reused_tokens = sum([move["to_position"][1]-move["to_position"][0] for move in diff_report["moves"]])
                
                # 保存相似度和重用token信息
                similar_docs.append({
                    "id": candidate_id,
                    "similarity": candidate["distance"],
                    "reused_token_num": reused_tokens
                })
            
            # 5. 选择top5并保存
            if similar_docs:
                doc_pair = {
                    "id": source_doc_id,
                    "cosine_similarity_top5": sorted(similar_docs, 
                                                   key=lambda x: x["similarity"], 
                                                   reverse=True)[:5],
                    "reused_token_num_top5": sorted(similar_docs, 
                                                  key=lambda x: x["reused_token_num"], 
                                                  reverse=True)[:5]
                }
                result_data["similar_pairs"].append(doc_pair)
                
        except Exception as e:
            print(f"Error processing document {source_doc['id']}: {e}")
            continue
        
        # # 定期保存结果
        # if len(result_data["similar_pairs"]) % 100 == 0:
        #     json.dump(result_data, open(save_path, "w"), ensure_ascii=False, indent=4)
    
    # 最终保存
    json.dump(result_data, open(save_path, "w"), ensure_ascii=False, indent=4)
    print(f"Successfully processed {len(result_data['similar_pairs'])} documents")

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = "en-zh"
    dataset_path = f"examples/dataset/data/opus/opus_dataset_{config}.json"
    save_path = f"examples/dataset/data/opus/opus_dataset_{config}_similar_docs_250403.json"
    #process_opus_dataset(save_path,config)
    #embed_opus_dataset(dataset_path: str, config: str, batch_size: int = 32)
    find_opus_similar_docs(dataset_path=dataset_path, save_path=save_path, config=config)
