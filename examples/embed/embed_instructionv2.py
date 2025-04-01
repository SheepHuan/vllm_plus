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
def embed_instructionv2_dataset(dataset_path: str, batch_size: int = 32):
    """批量计算embedding并插入数据库
    
    Args:
        dataset_path: 数据集路径
        batch_size: 批处理大小
    """
    data = json.load(open(dataset_path, "r"))
    dataset_embedder = DatasetEmbeder(
        model_name="all-MiniLM-L6-v2",
        collection_name="instructionv_wild_v2",
        database_path="examples/dataset/data/database/milvus_instructionv_wild_v2.db"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    
    # 初始化批处理列表
    batch_items = []
    batch_texts = []
    # global_id = 1
    
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
                        "id": item["id"],
                        "text": item["text"],
                        "vector": embedding.tolist()  # 确保embedding是列表格式
                    }
                    batch_data.append(item_data)
                
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

    # print(f"Processed {global_id-1} documents in total")

def find_instructionv2_similar_docs(dataset_path: str, save_path: str):
    from edit2 import find_text_differences
    client = MilvusClient("examples/dataset/data/database/milvus_instructionv_wild_v2.db")
    connection_name = "instructionv_wild_v2"
    model = SentenceTransformer("all-MiniLM-L6-v2")
    dataset = json.load(open(dataset_path, "r"))
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    
    save_data = {
        "all_texts": {},
        "similar_pairs": []
    }
    dataset = random.sample(dataset,5000)
    for item in tqdm(dataset):
        text = item["text"]
        if len(tokenizer.encode(text)) < 50:
            continue
        embedding = model.encode(text)
        results = client.search(collection_name=connection_name,data=[embedding],limit=50,output_fields=["text"])
        if len(results) > 0:
            temp = {
                "high_similarity": [],
                "high_token_reused": []
            }
            save_data["all_texts"][item["id"]] = text
            for result in results[0]:
                if result["distance"] >=1.0:
                    continue
                text = result["entity"]["text"]
                
                diff_report = find_text_differences(tokenizer.encode(text), tokenizer.encode(item["text"]))
                reused_token_num =0
                for move in diff_report["moves"]:
                   reused_token_num += len(move["to_position"])
                
                temp["high_similarity"].append(
                    {
                        "id": result["id"],
                         "similarity": result["distance"],
                        "token_reused": reused_token_num,
                    }
                )
                temp["high_token_reused"].append(
                    {
                        "id": result["id"],
                        "similarity": result["distance"],
                        "token_reused": reused_token_num,
                    }
                )
                
                if result["id"] not in save_data["all_texts"]:
                    save_data["all_texts"][result["id"]] = text
                
            save_item = {
                "id": item["id"],
                "high_similarity_top5": sorted(temp["high_similarity"], key=lambda x: x["similarity"], reverse=True)[:5],
                "high_token_reused_top5": sorted(temp["high_token_reused"], key=lambda x: x["token_reused"], reverse=True)[:5]
            }
            save_data["similar_pairs"].append(save_item)

    json.dump(save_data, open(save_path, "w"), ensure_ascii=False, indent=4)


def process_instructionv2_dataset(save_path):
    dataset = json.load(open(save_path, "r"))
    global_id = 1
    for item in tqdm(dataset):
        item["id"] = global_id
        global_id += 1
    json.dump(dataset, open(save_path, "w"), ensure_ascii=False, indent=4)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    dataset_path = "examples/dataset/data/insturctionv2/instruction_wildv2_ppl.json"
    # embed_instructionv2_dataset(dataset_path, batch_size=128)
    find_instructionv2_similar_docs(dataset_path, "examples/dataset/data/insturctionv2/instruction_wildv2_similar_250331.json")
    # process_instructionv2_dataset(dataset_path)