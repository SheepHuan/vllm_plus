import datasets
import transformers
from tqdm import tqdm
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import random
from embed_dataset import DatasetEmbeder
import os


def process_chunk(chunk_data, config, tokenizer_name="Qwen/Qwen2.5-1.5B-Instruct"):
    """处理数据集的一个子块"""
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    gt_key, input_key = config.split("-")
    long_size_items = []
    
    for item in chunk_data:
        gt_text = item[gt_key]
        input_text = item[input_key]
        input_token_ids = tokenizer.encode(input_text, add_special_tokens=False)
        if len(input_token_ids) > 64:
            item["length"] = len(input_token_ids)
            long_size_items.append(item)
            
    return long_size_items

def split_wmt_dataset(config: str="zh-en", 
                     save_path: str="examples/bench_cache/data/wmt_dataset.json",
                     num_processes: int=8):
    """
    多进程处理WMT数据集
    
    Args:
        config: 数据集配置
        save_path: 保存路径
        num_processes: 进程数量
    """
    dataset :datasets.DatasetDict = datasets.load_dataset("wmt/wmt19", config)
    train_dataset: datasets.Dataset = dataset['train']
    validation_dataset: datasets.Dataset = dataset['validation']
    
    real_dataset = train_dataset
    # 计算每个进程处理的数据量
    chunk_size = len(real_dataset) // num_processes
    # print(len(real_dataset),chunk_size)
    train_chunks = [real_dataset[i:i + chunk_size]["translation"] for i in range(0, len(real_dataset), chunk_size)]
    validation_chunks = [validation_dataset[i:i + chunk_size]["translation"] for i in range(0, len(validation_dataset), chunk_size)]
    chunks = train_chunks + validation_chunks
    
    
    # 使用进程池处理数据
    long_size_items = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        # 提交任务
        for chunk in chunks:
            future = executor.submit(process_chunk, chunk, config)
            futures.append(future)
        
        # 收集结果
        for future in tqdm(futures, desc=f"Processing {config} dataset", total=len(futures)):
            chunk_results = future.result()
            long_size_items.extend(chunk_results)
    
    # 保存结果
    json.dump(long_size_items, 
             open(save_path, "w"), 
             ensure_ascii=False, 
             indent=4)
    
    return long_size_items


def combine_wmt_dataset(input_path: str="examples/dataset/data/wmt_dataset_long.json",
                       output_path: str="examples/dataset/data/wmt_dataset_benchmark.json"):
    wmt_dataset = json.load(open(input_path, "r"))
    
    new_dataset = random.sample(wmt_dataset, 1000)
    
    bench_dataset = []
    # 每10条组合成一个数据，组合，200个数据
    for i in range(200):
        items = random.sample(new_dataset, 10)
        input_text = " ".join([item["zh"] for item in items])
        gt_text = " ".join([item["en"] for item in items])
        bench_dataset.append({
            "input": input_text,
            "gt": gt_text
        })
    
    json.dump(bench_dataset, open(output_path, "w"), ensure_ascii=False, indent=4)

def merge_wmt_dataset(save_path: str="examples/dataset/data/wmt_dataset_long.json",
                      config: str="zh-en",
                      dataset_path: str="wmt/wmt19"):
    tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    validation_dataset = list(datasets.load_dataset(dataset_path,config,split="validation"))
    train_dataset = list(datasets.load_dataset(dataset_path,config,split="train"))
    dataset = validation_dataset + train_dataset
    global_id = 0
    save_data = []
    for item in dataset:
        source_text = item["translation"][config.split("-")[0]]
        target_text = item["translation"][config.split("-")[1]]
        source_token_ids = tokenizer.encode(source_text, add_special_tokens=False)
        target_token_ids = tokenizer.encode(target_text, add_special_tokens=False)
        if len(source_token_ids) < 50:
            continue
        # data["length"] = len(source_token_ids)
        global_id += 1
        data = {
            "id": global_id,
            "translation": item["translation"],
        }
        save_data.append(data)
    json.dump(save_data, open(save_path, "w"), ensure_ascii=False, indent=4)        
    
    
    
def embed_wmt_dataset(model_name,dataset_path,database_path,collection_name,global_id=0,config="zh-en",batch_size=64):
    dataset = json.load(open(dataset_path, "r"))
    dataset_embeder = DatasetEmbeder(model_name,collection_name=collection_name,database_path=database_path)
    
    for i in tqdm(range(0,len(dataset),batch_size),desc=f"embedding {config} dataset"):
        try:
            items = dataset[i:i+batch_size]
            source_texts = []
            target_texts = []
            for item in items:
                source_texts.append(item["translation"][config.split("-")[0]])
                target_texts.append(item["translation"][config.split("-")[1]])
            embeddings = dataset_embeder.embed_text(source_texts)
            datas = [{"id":global_id+j,"vector":embeddings[j],"translation":items[j]["translation"]} for j in range(len(items))]
            dataset_embeder.insert_dataset(datas)
            global_id += len(items)
        except Exception as e:
            print(e)
            continue
    print(f"embedding {config} dataset done,max id is {global_id}")    

def find_wmt_similar_docs(dataset_path: str,save_path: str):
    from edit2 import find_text_differences
    from sentence_transformers import SentenceTransformer
    from pymilvus import MilvusClient
    dataset = json.load(open(dataset_path, "r"))
    tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = MilvusClient("/root/code/vllm_plus/examples/dataset/data/database/milvus_wmt19.db")
    collection_name = "wmt19"
    save_data = {
        "all_data": {},
        "similar_pairs": []
    }
    
    for item in dataset:
        id = item["id"]
        translation = item["translation"]
        embeddings = model.encode(translation["zh"])
        save_data["all_data"][id] = translation
        results = client.search(collection_name=collection_name,data=[embeddings],limit=50,output_fields=["translation"])
        if len(results[0]) > 0:
            tmp_item = {
                "id": id,
                "translation": translation,
                "similar_items": []
            }
            for result in results[0]:
                
                reused_token_num = 0
                source_tokens = tokenizer.encode(result["entity"]["translation"]["zh"])
                target_tokens = tokenizer.encode(translation["zh"])
                diff_report = find_text_differences(source_tokens,target_tokens)
                for move in diff_report["moves"]:
                    reused_token_num += len(move["to_position"])
                tmp_item["similar_items"].append({
                    "id": result["id"],
                    "similarity": result["distance"],
                    "reused_token_num": reused_token_num
                })
            new_item = {
                "id": item["id"],
                "cosine_similarity_top5": sorted(tmp_item["similar_items"],key=lambda x: x["similarity"],reverse=True)[1:6],
                "reused_token_num_top5": sorted(tmp_item["similar_items"],key=lambda x: x["reused_token_num"],reverse=True)[1:6]
            }
            save_data["similar_pairs"].append(new_item)
        
    json.dump(dataset, open(save_path, "w"), ensure_ascii=False, indent=4)
    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dataset_path = "wmt/wmt19"
    database_path = "examples/dataset/data/database/milvus_wmt19.db"
    collection_name = "wmt19"
    model_name = "all-MiniLM-L6-v2"
    # model_name = "BAAI/bge-m3"
    config = "zh-en"
    
    # merge_wmt_dataset(save_path=f"examples/dataset/data/wmt19/wmt19_dataset_zh-en.json",
    #                   config="zh-en")
    # merge_wmt_dataset(save_path=f"examples/dataset/data/wmt19/wmt19_dataset_gu-en.json",
    #                   config="gu-en")
    # embed_wmt_dataset(model_name=model_name,dataset_path=dataset_path,database_path=database_path,collection_name=collection_name,global_id=0,config=config)
    
    find_wmt_similar_docs(dataset_path=f"examples/dataset/data/wmt19/wmt19_dataset_zh-en.json",
                          save_path=f"examples/dataset/data/wmt19/wmt19_dataset_zh-en_similar_docs.json")
