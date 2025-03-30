import datasets
import transformers
from tqdm import tqdm
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import random
from embed_dataset import DatasetEmbeder

    
    
def embed_wmt_dataset(model_name,dataset_path,database_path,collection_name,global_id=0,config="zh-en"):
    dataset_embeder = DatasetEmbeder(model_name,collection_name=collection_name,database_path=database_path)
    train_dataset = list(datasets.load_dataset(dataset_path,config,split="train"))
    validation_dataset = list(datasets.load_dataset(dataset_path,config,split="validation"))
    
    dataset = train_dataset + validation_dataset 
    for item in tqdm(dataset):
        try:
            global_id += 1
            document = item["translation"][config.split("-")[0]]
            embeddings = dataset_embeder.embed_text(document)
            data = {
                "id": global_id,
                "vector":embeddings,
                # "document": document,
            }
        except Exception as e:
            print(e)
            continue
    print(f"embedding {config} dataset done,max id is {global_id}")    
    # print(f"embedding {tag} dataset done,max id is {global_id}")    
            
if __name__ == "__main__":
    dataset_path = "wmt/wmt19"
    database_path = "examples/dataset/data/database/milvus_wmt19.db"
    collection_name = "wmt19"
    model_name = "all-MiniLM-L6-v2"
    config = "zh-en"
    embed_wmt_dataset(model_name=model_name,dataset_path=dataset_path,database_path=database_path,collection_name=collection_name,global_id=0,config=config)

