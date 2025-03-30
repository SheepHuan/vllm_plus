from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from pymilvus import MilvusClient
import hashlib
import datasets
import time
import random
import os
from tqdm import tqdm
from transformers import AutoTokenizer
import json
def get_unique_id():
    """生成一个唯一的整数ID
    
    Args:
        data (str): 需要生成唯一ID的输入字符串
        
    Returns:
        int: 一个基于输入数据的唯一整数ID
    """
     # 获取当前时间戳（微秒级）
    timestamp = int(time.time() * 1000000)
    # 添加3位随机数以避免同一微秒内的冲突
    random_num = random.randint(0, 999)
    # 组合时间戳和随机数
    unique_id = timestamp * 1000 + random_num
    return unique_id

class DatasetEmbeder:
    def __init__(self,model_name,database_path,collection_name,device="cuda:0"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name,device=device)
        
        self.client = MilvusClient(database_path)
        self.collection_name = collection_name
        
        if self.client.has_collection(collection_name=collection_name):
            self.client.drop_collection(collection_name=collection_name)
        self.client.create_collection(
            collection_name=collection_name,
            dimension=384,  # The vectors we will use in this demo has 768 dimensions
        )

    def embed_text(self,text):
        return self.model.encode(text)
    
    def insert_dataset(self,data):
        self.client.upsert(collection_name=self.collection_name,data=data)
        
    def query(self,embeddings,topk=10):
        self.client.search(collection_name=self.collection_name,data=embeddings,limit=topk)
        
# class DatasetFinder:
#     def __init__(self,database_path,collection_name):
#         self.client = MilvusClient(database_path)
#         self.collection_name = collection_name
        
#     def find_similar_docs(self,embeddings,topk=10):
#         return self.client.query(collection_name=self.collection_name,data=embeddings,limit=topk)
    
    
def embedding_xsum_dataset(model_name,dataset_path,tag="train",global_id=2100000):
    dataset_embeder = DatasetEmbeder(model_name,collection_name="xsum",database_path="examples/dataset/data/database/milvus_xsum.db")
    dataset = datasets.load_dataset(dataset_path,trust_remote_code=True,split=tag)
    for item in tqdm(dataset):
        try:
            global_id += 1
            document = item["document"]
            embeddings = dataset_embeder.embed_text(document)
            data = {
                "id": global_id,
                "vector":embeddings,
                "document": item["document"],
                "summary": item["summary"]
            }
            dataset_embeder.insert_dataset(data)
        except Exception as e:
            print(e)
            continue
    print(f"embedding {tag} dataset done,max id is {global_id}")

def find_xsum_similar_docs(model_name,dataset_path,database_path,save_path):
    client = MilvusClient(database_path)
    collection_name = "xsum"
    model = SentenceTransformer(model_name,device="cuda:0")
    train_dataset = datasets.load_dataset(dataset_path,trust_remote_code=True,split="train")
    validation_dataset = datasets.load_dataset(dataset_path,trust_remote_code=True,split="validation")
    test_dataset = datasets.load_dataset(dataset_path,trust_remote_code=True,split="test")
    # 将数据集转换为列表，然后再随机采样
    data_list = list(train_dataset) + list(validation_dataset) + list(test_dataset)
    # data_list = random.sample(data_list, min(100000, len(data_list)))
    all_documents = dict()
    save_data = []
    for item in tqdm(data_list):
        try:
            document = item["document"]
            embeddings = model.encode(document)
            results = client.search(collection_name=collection_name,data=[embeddings],limit=50,output_fields=["document","summary"])
            if len(results[0]) > 0:
                saved_item = {
                    "document": document,
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

def find_the_most_similar_and_max_resued_docs(dataset_path,save_path):
    from edit2 import find_text_differences
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct",local_files_only=True)
    data = json.load(open(dataset_path,"r"))
    all_documents = data["all_documents"]
    similar_docs = data["similar_docs"]
    # 对similar_docs进行排序
  
    
    for item in tqdm(similar_docs):
        item["similar_docs"] = sorted(item["similar_docs"],key=lambda x: x["similarity"],reverse=True)
        target_tokens = tokenizer.encode(item["document"])
        max_resued_token_num = 0
        max_resued_doc_id = -1
        for similar_doc in item["similar_docs"]:
            if similar_doc["similarity"] > 0.9999:
                continue
            source_tokens = tokenizer.encode(all_documents[str(similar_doc["id"])]["document"])
            if len(source_tokens) < 10 or len(target_tokens) < 10:
                continue
            diff_report = find_text_differences(source_tokens,target_tokens)
            resued_token_num =0
            for move in diff_report["moves"]:
                resued_token_num += len(move["to_position"])
            if resued_token_num > max_resued_token_num:
                max_resued_token_num = resued_token_num
                max_resued_doc_id = similar_doc["id"]
        # if max_resued_doc_id is not None:
        item["max_resued_doc_id"] = max_resued_doc_id
        item["max_resued_token_num"] = max_resued_token_num
    json.dump(data,open(save_path,"w"),indent=4,ensure_ascii=False)

def select_xsum_dataset(dataset_path,save_path):
    data = json.load(open(dataset_path,"r"))
    all_documents = data["all_documents"]
    similar_docs = data["similar_docs"]
    
    save_data = {
        "all_documents":all_documents,
        "similar_docs":[]
    }
    
    for item in tqdm(similar_docs):
        item["similar_docs"] = sorted(item["similar_docs"],key=lambda x: x["similarity"],reverse=True)
        if item["similar_docs"][1]["similarity"]<=0.7:
            continue
        new_item = {
            "document":item["document"],
            "summary":item["summary"],
            "high_similarity_doc":item["similar_docs"][1],
            "max_resued_doc":item["max_resued_doc_id"],
            "max_resued_token_num":item["max_resued_token_num"]
        }
        save_data["similar_docs"].append(new_item)
    json.dump(save_data,open(save_path,"w"),indent=4,ensure_ascii=False)
    print(f"select {len(save_data['similar_docs'])} items from {len(similar_docs)} items")

if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model_name = "all-MiniLM-L6-v2"
    
    
    dataset_path = "EdinburghNLP/xsum"
    database_path = "examples/dataset/data/database/milvus_xsum.db"
    save_path = f"examples/dataset/data/xsum/{model_name}_train_similar_docs_topk50.json"
    save_path_high_sim = f"examples/dataset/data/xsum/{model_name}_train_similar_docs_topk50_high_similarity.json"
    # embedding_xsum_dataset(model_name="all-MiniLM-L6-v2",dataset_path="EdinburghNLP/xsum",tag="train",global_id=0)
    # embedding_xsum_dataset(model_name="all-MiniLM-L6-v2",dataset_path="EdinburghNLP/xsum",tag="test",global_id=2100000)                                                     
    # embedding_xsum_dataset(model_name="all-MiniLM-L6-v2",dataset_path="EdinburghNLP/xsum",tag="validation",global_id=2111400)
    # find_xsum_similar_docs(model_name=model_name,dataset_path=dataset_path,database_path=database_path,save_path=save_path)
    # find_the_most_similar_and_max_resued_docs(dataset_path=save_path,save_path=save_path)
    select_xsum_dataset(dataset_path=save_path,save_path=save_path_high_sim)
    
    