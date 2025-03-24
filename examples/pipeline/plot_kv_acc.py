import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager 
from tqdm import tqdm
import numpy as np
import random
import json
from typing import List
import time
from multiprocessing import Process, Queue, cpu_count
import queue
from multiprocessing import Pool

font_path = "/root/code/vllm_plus/examples/dataset/data/fonts"
 
font_files = font_manager.findSystemFonts(fontpaths=font_path)
 
for file in font_files:
    font_manager.fontManager.addfont(file)

# 设置字体
matplotlib.rcParams['font.family'] = 'Arial'  # 设置字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"

def get_key_value(model:LLM,prompt: List[str]):
    model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["check"] = False
    model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata['collect'] = True
    template = "<|im_start|>user\n{prompt}\n<|im_end|>"
    prompt = template.format(prompt=prompt)
    
    sampling_params = SamplingParams(temperature=0, max_tokens=1)
    output = model.generate(prompt, sampling_params,use_tqdm=False)
    
    llm_layers = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers
    
    past_key_values = []
    num_layer = len(llm_layers)
    for j in range(num_layer):
        hack_kv = llm_layers[j].self_attn.hack_kv
        temp_key_cache = hack_kv[0].clone()
        temp_value_cache = hack_kv[1].clone()
        past_key_values.append(torch.stack([temp_key_cache,temp_value_cache],dim=0))
    past_key_values = torch.stack(past_key_values,dim=0)
    return past_key_values

def clean_text(json_path:str,clean_path:str):
    import hashlib
    from edit2 import apply_change,find_text_differences
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    data = json.load(open(json_path))
    new_data = []
    global_id = 0  # 全局ID计数器
    bin_count = [[] for _ in range(10)] # 统计不同token复用率的paire
    
    # 给每个cluster的每一个meber添加一个独特的整数id
    for key,cluster in tqdm(data["clusters"].items()):
        # hash每一个text
        members = cluster["members"]
        hash_set = set()
        # 给第一个member添加global_id
        global_id += 1
        members[0]["global_id"] = global_id
        
        new_members = [members[0]]
        for item in members[1:]:
            # 找到source_item和target_item的差异
            source_token = tokenizer.encode(members[0]["text"])
            target_token = tokenizer.encode(item["text"])
            diff_report = find_text_differences(tokenizer,source_token,target_token,window_size=3)
            token_reuse_rate = diff_report["summary"]["reuse_ratio"]
            
            hash_value = hashlib.md5(item["text"].encode()).hexdigest()
            if hash_value not in hash_set:
                global_id += 1
                hash_set.add(hash_value)
                item["global_id"] = global_id
                new_members.append(item)
                item["token_reuse_rate"] = token_reuse_rate
                
                bin_count[int((token_reuse_rate-0.01)//10)].append(item)
        # new_data.append(new_members)
    json.dump(bin_count,open(clean_path,"w"),indent=4)

def calculate_token_reuse_rate(pair):
    """
    计算token复用率，使用共享的tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    from edit2 import find_text_differences
    
    source_token = tokenizer.encode(pair["source"])
    target_token = tokenizer.encode(pair["target"])
    diff_report = find_text_differences(tokenizer, source_token, target_token, window_size=3)
    return diff_report["summary"]["reuse_ratio"]


def select_similar_pairs(json_path: str, save_path: str):
    """选择相似文本对并计算复用率"""
    # 加载数据
    data = json.load(open(json_path))
    clusters = data["clusters"]
    unique_pairs = []
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    # 收集文本对
    print("收集文本对...")
    pair_id = 0
    for key, cluster in tqdm(clusters.items()):
        members = cluster["members"]
        for i in range(len(members)):
            for j in range(i+1, len(members)):
                unique_pairs.append({
                    "pair_id": pair_id,
                    "source": members[i]["text"],
                    "target": members[j]["text"],
                    "source_token": tokenizer.encode(members[i]["text"]),
                    "target_token": tokenizer.encode(members[j]["text"]),
                })
                pair_id += 1
    # 保存结果
    print(f"\n保存结果到 {save_path}")
    with open(save_path, "w") as f:
        json.dump(unique_pairs, f, indent=4)
    
       

def compute_similarity(clean_path:str,save_path:str):
    """
    计算相似度
    """
    from tqdm import tqdm
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",device="cuda:1")
    data = json.load(open(clean_path))
    for index,pair in tqdm(enumerate(data)):
        source_embedding = model.encode(pair["source"])
        target_embedding = model.encode(pair["target"])
        similarity = np.dot(source_embedding, target_embedding) / (np.linalg.norm(source_embedding) * np.linalg.norm(target_embedding))
        pair["cosine_similarity"] = float(similarity)
    json.dump(data,open(save_path,"w"),indent=4)


            
def calculate_window_sizes(pair_data):
    """
    计算单个pair在不同窗口大小下的复用率
    """
    from edit2 import find_text_differences
    try:
        # 确保token是列表类型
        source_token = pair_data["source_token"] if isinstance(pair_data["source_token"], list) else [pair_data["source_token"]]
        target_token = pair_data["target_token"] if isinstance(pair_data["target_token"], list) else [pair_data["target_token"]]
        pair_id = pair_data["pair_id"]
        window_results = []
        
        for window_size in [1, 2, 3, 4, 5]:
            if window_size > len(source_token) or window_size > len(target_token):
                continue
            diff_report = find_text_differences(source_token, target_token, window_size=window_size)
            window_results.append([window_size, float(diff_report["summary"]["reuse_ratio"])])
        
        return pair_id, window_results
        
    except Exception as e:
        print(f"Error processing pair {pair_data.get('pair_id', 'unknown')}: {str(e)}")
        return pair_id, []

def get_windowsize_similarity(data_path: str, save_path: str):
    """使用进程池处理不同窗口大小的计算"""
    from multiprocessing import Pool
    from edit2 import find_text_differences
    # 加载数据
    print("加载数据...")
    data = json.load(open(data_path))
    
    # 准备任务数据，确保token是列表类型
    new_data = []
    for idx, pair in tqdm( enumerate(data)):
        # 检查并转换token类型
        source_token = pair["source_token"]
        target_token = pair["target_token"]
        pair["window_size"] = []
        for window_size in [1,2,3,4,5]:
            if window_size >= len(source_token) or window_size >= len(target_token):
                continue
            start_time = time.time()
            diff_report = find_text_differences(source_token,target_token,window_size=window_size)
            end_time = time.time()
            pair["reuse_rate"] = diff_report["summary"]["reuse_ratio"]
            pair["window_size"].append([window_size,diff_report["summary"]["reuse_ratio"],(end_time-start_time)*1000])
        new_data.append(pair) 
    # 保存结果
    print(f"保存结果到 {save_path}")
    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)


def plot_similarity_reuse_rate(data):
    # 创建图表
    plt.figure(figsize=(12, 8))
    def plot_data(data_path,tag,color):
        points_x = []
        points_y = []
        data = json.load(open(data_path))
        for pair in tqdm(data):
            if len(pair["window_size"]) == 0:
                continue
            window_size,reuse_rate,time = pair["window_size"][0]
            similarity = pair["cosine_similarity"]
            points_x.append(reuse_rate)
            points_y.append(similarity)
        plt.scatter(points_x,points_y,s=10,c=color,alpha=0.7,label=tag)
    for item in data:
        plot_data(item["path"],item["tag"],item["color"])
    
    plt.xlabel("reuse_rate")
    plt.ylabel("similarity")
    plt.legend()
    plt.title("similarity and reuse_rate")
    plt.savefig("examples/pipeline/images/similarity_reuse_rate.png",dpi=300,bbox_inches="tight")
    plt.close()

def plot_window_size_reuse_rate(data_path:str):
    # 创建图表
    plt.figure(figsize=(12, 8))
    points_x = []
    points_y = []
    data = json.load(open(data_path))
    for pair in tqdm(data):
        if len(pair["window_size"]) == 0:
            continue
        window_size,reuse_rate,time = pair["window_size"][0]
        target_length = len(pair["target_token"])
        points_x.append(window_size)
        points_y.append(reuse_rate)
    plt.scatter(points_x,points_y,s=10,c="blue",alpha=0.7)
    
    
    plt.xlabel("window_size")
    plt.ylabel("reuse_rate")
    plt.legend()
    plt.title("window_size and reuse_rate")
    plt.savefig("examples/pipeline/images/window_size_reuse_rate.png",dpi=300,bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    # 设置环境变量
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # raw_path = "examples/dataset/data/similar/instruction_wildv2/instruction_wildv2_batch_embeddings_clusters.json"
    # clean_path = "examples/dataset/data/similar/instruction_wildv2/instruction_wildv2_batch_embeddings_clusters_similar_pairs.json"
    similar_path = "examples/dataset/data/similar/instruction_wildv2/instruction_wildv2_batch_embeddings_clusters_similar_pairs_cosine_similarity.json"
    sim_reuse_path = "examples/dataset/data/similar/instruction_wildv2/instruction_wildv2_batch_embeddings_clusters_similar_pairs_reuse_rate.json"
    # select_similar_pairs(raw_path, clean_path)
    # compute_similarity(clean_path,save_path)
    # get_windowsize_similarity(sim_reuse_path,sim_reuse_path)
    # plot_similarity_reuse_rate(sim_reuse_path)
    plot_window_size_reuse_rate(sim_reuse_path)
    # raw_path = "examples/dataset/data/similar/sharegpt/sharegpt90k_batch_embeddings_clusters.json"
    # clean_path = "examples/dataset/data/similar/sharegpt/sharegpt90k_batch_embeddings_clusters_similar_pairs.json"
    # similar_path = "examples/dataset/data/similar/sharegpt/sharegpt90k_batch_embeddings_clusters_similar_pairs_cosine_similarity.json"
    # sim_reuse_path = "examples/dataset/data/similar/sharegpt/sharegpt90k_batch_embeddings_clusters_similar_pairs_reuse_rate.json"
    # select_similar_pairs(raw_path, clean_path)
    # compute_similarity(clean_path,similar_path)
    # get_windowsize_similarity(similar_path,sim_reuse_path)
    
    # raw_path = "examples/dataset/data/similar/belle/belle_batch_embeddings_clusters.json"
    # clean_path = "examples/dataset/data/similar/belle/belle_batch_embeddings_clusters_similar_pairs.json"
    # similar_path = "examples/dataset/data/similar/belle/belle_batch_embeddings_clusters_similar_pairs_cosine_similarity.json"
    # sim_reuse_path = "examples/dataset/data/similar/belle/belle_batch_embeddings_clusters_similar_pairs_reuse_rate.json"
    # select_similar_pairs(raw_path, clean_path)
    # compute_similarity(clean_path,similar_path)
    # get_windowsize_similarity(similar_path,sim_reuse_path)
    
    
    # data = [
    #     {
    #         "tag": "InstructionWild v2",
    #         "path": "examples/dataset/data/similar/instruction_wildv2/instruction_wildv2_batch_embeddings_clusters_similar_pairs_reuse_rate.json",
    #         "color": "blue"
    #     },
    #     # {
    #     #     "tag": "ShareGPT-90k",
    #     #     "path": "examples/dataset/data/similar/sharegpt/sharegpt90k_batch_embeddings_clusters_similar_pairs_reuse_rate.json",
    #     #     "color": "red"
    #     # }
    # ]
    # plot_similarity_reuse_rate(data)