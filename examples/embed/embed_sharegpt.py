import json
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
from embed_dataset import DatasetEmbeder
from sentence_transformers import SentenceTransformer
from pymilvus import connections,MilvusClient
from tqdm import tqdm
from transformers import AutoTokenizer
import os
import random
from edit2 import find_text_differences

def embed_sharegpt_dataset(dataset_path: str, batch_size: int = 32):
    """批量计算embedding并插入数据库
    
    Args:
        dataset_path: 数据集路径
        batch_size: 批处理大小
    """
    data = json.load(open(dataset_path, "r"))
    dataset_embedder = DatasetEmbeder(
        model_name="all-MiniLM-L6-v2",
        collection_name="sharegpt",
        database_path="examples/dataset/data/database/milvus_sharegpt.db"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    
    # 初始化批处理列表
    batch_items = []
    batch_texts = []
    global_id = 1
    
    for item in tqdm(data, desc="Processing documents"):
        text = item["text"]
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        
        if len(token_ids) < 10:
            continue
            
        # 添加到当前批次
        batch_items.append(item)
        batch_texts.append(text)
        
        # 当达到批处理大小时，处理当前批次
        if len(batch_texts) >= batch_size:
            try:
                # 批量计算embeddings
                batch_embeddings = dataset_embedder.embed_text(batch_texts)
                
                # 准备批量插入数据
                batch_data = []
                for item, embedding in zip(batch_items, batch_embeddings):
                    item_data = {
                        "id": global_id,
                        "text": item["text"],
                        "vector": embedding.tolist()  # 确保embedding是列表格式
                    }
                    batch_data.append(item_data)
                    global_id += 1
                
                # 批量插入数据库
                dataset_embedder.insert_dataset(batch_data)
                
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
            for item, embedding in zip(batch_items, batch_embeddings):
                item_data = {
                    "id": global_id,
                    "text": item["text"],
                    "vector": embedding.tolist()
                }
                batch_data.append(item_data)
                global_id += 1
            dataset_embedder.insert_dataset(batch_data)
        except Exception as e:
            print(f"Error processing final batch: {e}")
    
    print(f"Processed {global_id-1} documents in total")

def process_sharegpt_dataset(save_path):
    dataset = json.load(open(save_path, "r"))
    global_id = 1
    for item in tqdm(dataset):
        item["id"] = global_id
        global_id += 1
    json.dump(dataset, open(save_path, "w"), ensure_ascii=False, indent=4)
    
def find_sharegpt_similar_docs(dataset_path: str, save_path: str):
    """查找ShareGPT数据集中的相似文档并计算token重用数
    
    Args:
        dataset_path: 数据集路径
        save_path: 结果保存路径
    """
    # 初始化必要的组件
    vector_db = MilvusClient("examples/dataset/data/database/milvus_sharegpt.db")
    collection_name = "sharegpt"
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda:0")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    
    # 加载数据集
    source_documents = json.load(open(dataset_path, "r"))
    # 随机采样以减少处理量
    # source_documents = random.sample(source_documents, 100)
    
    result_data = {
        "all_texts": {},        # 存储所有文档的文本
        "similar_pairs": []     # 存储相似文档对
    }
    
    for source_doc in tqdm(source_documents, desc="Processing documents"):
        try:
            source_text = source_doc["text"]
            if len(tokenizer.encode(source_text)) < 50:
                continue
            text_embedding = embedding_model.encode(source_text)
            
            # 搜索相似文档
            search_results = vector_db.search(
                collection_name=collection_name,
                data=[text_embedding],
                limit=50,
                output_fields=["text"]
            )
            
            if len(search_results) == 0:
                continue
                
            # 存储原始文档文本
            source_doc_id = source_doc["id"]
            result_data["all_texts"][source_doc_id] = source_text
            
            # 处理相似文档
            similar_docs = []
            source_tokens = tokenizer.encode(source_text)
            
            for candidate in search_results[0]:
                # 跳过完全相同的文档
                if candidate["distance"] >= 1.0:
                    continue
                    
                candidate_text = candidate["entity"]["text"]
                candidate_id = candidate["id"]
                
                # 计算token重用数量
                diff_report = find_text_differences(
                    tokenizer.encode(candidate_text), 
                    source_tokens
                )
                reused_tokens = sum(len(move["to_position"]) for move in diff_report["moves"])
                
                # 保存相似度和重用token信息
                doc_info = {
                    "id": candidate_id,
                    "similarity": candidate["distance"],
                    "token_reused": reused_tokens
                }
                similar_docs.append(doc_info)
                
                # 保存相似文档的文本
                if candidate_id not in result_data["all_texts"]:
                    result_data["all_texts"][candidate_id] = candidate_text
            
            # 选择top5并保存
            if similar_docs:
                doc_pair = {
                    "id": source_doc_id,
                    "high_similarity_top5": sorted(
                        similar_docs, 
                        key=lambda x: x["similarity"], 
                        reverse=True
                    )[:5],
                    "high_token_reused_top5": sorted(
                        similar_docs, 
                        key=lambda x: x["token_reused"], 
                        reverse=True
                    )[:5]
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
    dataset_path = "examples/dataset/data/sharegpt/sharegpt90k_ppl.json"
    save_path = "examples/dataset/data/sharegpt/sharegpt90k_similar_250331.json"
    # embed_sharegpt_dataset(dataset_path, batch_size=128)
    # process_sharegpt_dataset(dataset_path)
    find_sharegpt_similar_docs(dataset_path, save_path)
    