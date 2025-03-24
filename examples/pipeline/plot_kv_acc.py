import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager 
from tqdm import tqdm

font_path = "/root/code/vllm_plus/examples/dataset/data/fonts"
 
font_files = font_manager.findSystemFonts(fontpaths=font_path)
 
for file in font_files:
    font_manager.fontManager.addfont(file)

# 设置字体
matplotlib.rcParams['font.family'] = 'Arial'  # 设置字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
import json
from typing import List
import time
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

def compare_kv_error(source_kv: List[torch.Tensor], target_kv: List[torch.Tensor]):
    pass


def clean_text(json_path:str,clean_path:str):
    import hashlib
    data = json.load(open(json_path))
    new_data = []
    global_id = 0  # 全局ID计数器
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
            hash_value = hashlib.md5(item["text"].encode()).hexdigest()
            if hash_value not in hash_set:
                global_id += 1
                hash_set.add(hash_value)
                item["global_id"] = global_id
                new_members.append(item)
        new_data.append(new_members)
    json.dump(new_data,open(clean_path,"w"),indent=4)

def test(json_path:str):
    from edit2 import apply_change,find_text_differences
    from vdb_cls import VectorDB
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    model = LLM(model=model_name,dtype="float16")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data = json.load(open(json_path))
    db = VectorDB(collection_name="instruction_wildv2",dimension=768,database_path="examples/pipeline/data/milvus_kvacc.db")
    for cluster in tqdm(data):
        source_item = cluster[0]
        for target_item in cluster[1:]:
            source_cached_res = db.search_by_id(source_item["global_id"])
            if len(source_cached_res) > 0:
                pass
            else:
                source_kv = get_key_value(model,source_item["text"])
                db.insert({
                    "id": source_item["global_id"],
                    "vector": np.random.randn(768).astype(np.float32),
                    "key_value_cache": source_kv.detach().cpu().numpy().astype(np.float32)
                })
            target_cached_res = db.search_by_id(target_item["global_id"])
            if len(target_cached_res) > 0:
                pass
            else:
                target_kv = get_key_value(model,target_item["text"])
                db.insert({
                    "id": target_item["global_id"],
                    "vector": np.random.randn(768).astype(np.float32),
                    "key_value_cache": target_kv.detach().cpu().numpy().astype(np.float32)
                })

if __name__ == "__main__":
    # test("examples/dataset/data/similar/instruction_wildv2/instruction_wildv2_batch_embeddings_clusters.json")
    raw_path = "examples/dataset/data/similar/instruction_wildv2/instruction_wildv2_batch_embeddings_clusters.json"
    clean_path = "examples/dataset/data/similar/instruction_wildv2/instruction_wildv2_batch_embeddings_clusters_clean.json"
    # clean_text(raw_path,clean_path)
    
    test(clean_path)
    
    