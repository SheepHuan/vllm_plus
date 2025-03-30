import datasets
import transformers
# from ppl import PerplexityMetric
from tqdm import tqdm
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import random
import os
from embed_dataset import DatasetEmbeder
from pymilvus import connections,MilvusClient
from sentence_transformers import SentenceTransformer

def embed_samsum(model_name,database_path,collection_name,global_id=0):
    dataset_embeder = DatasetEmbeder(model_name,collection_name=collection_name,database_path=database_path)
    train_data = json.load(open("examples/dataset/data/samsum/train.json","r"))
    validation_data = json.load(open("examples/dataset/data/samsum/val.json","r"))
    test_data = json.load(open("examples/dataset/data/samsum/test.json","r"))
    data_list = train_data + validation_data + test_data
    for item in tqdm(data_list):
        try:
            global_id += 1
            document = item["dialogue"]
            embeddings = dataset_embeder.embed_text(document)
            data = {
                "id": global_id,
                "vector":embeddings,
                "document": item["dialogue"],
                "summary": item["summary"]
            }
            dataset_embeder.insert_dataset(data)
        except Exception as e:
            print(e)
            continue
    
def find_samsum_similar_docs(model_name,database_path,save_path):
    client = MilvusClient(database_path)
    collection_name = "samsum"
    model = SentenceTransformer(model_name,device="cuda:0")
    train_data = json.load(open("examples/dataset/data/samsum/train.json","r"))
    validation_data = json.load(open("examples/dataset/data/samsum/val.json","r"))
    test_data = json.load(open("examples/dataset/data/samsum/test.json","r"))
    # 将数据集转换为列表，然后再随机采样
    data_list = train_data + validation_data + test_data
    all_documents = dict()
    save_data = []
    for item in tqdm(data_list):
        try:
            dialogue = item["dialogue"]
            embeddings = model.encode(dialogue)
            results = client.search(collection_name=collection_name,data=[embeddings],limit=50,output_fields=["document","summary"])
            if len(results[0]) > 0:
                saved_item = {
                    "document": dialogue,
                    "summary": item["summary"],
                    "similar_docs": []
                }
                for result in results[0]:
                    saved_item["similar_docs"].append({
                        "id": result["id"],
                        "similarity": result["distance"],
                    })
                    all_documents[result["id"]] = {
                        "document": result["entity"]["document"],
                        "summary": result["entity"]["summary"]
                    }
                save_data.append(saved_item)
            
        except Exception as e:
            print(e)
            continue
    json.dump({
        "all_documents": all_documents,
        "similar_docs": save_data
    },open(save_path,"w"),indent=4,ensure_ascii=False)
    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # group_samsum("examples/dataset/data/samsum/train.json","examples/dataset/data/samsum_group.json")
    model_name = "all-MiniLM-L6-v2"
    database_path = "examples/dataset/data/database/milvus_samsum.db"
    collection_name = "samsum"
    # embed_samsum(model_name=model_name,database_path=database_path,collection_name=collection_name,global_id=0)
    find_samsum_similar_docs(model_name=model_name,database_path=database_path,save_path="examples/dataset/data/samsum/all-mini-l6-v2_samsum_similar_docs_topk50.json")