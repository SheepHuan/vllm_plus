import datasets
import transformers
from ppl import PerplexityMetric
from tqdm import tqdm
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import random

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
if __name__ == "__main__":
    input_path = "examples/dataset/data/wmt_dataset_long.json"
    output_path = "examples/dataset/data/wmt_dataset_benchmark.json"
    combine_wmt_dataset(input_path=input_path, output_path=output_path)

