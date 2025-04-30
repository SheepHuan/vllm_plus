import json
from libs.pipeline import KVShareNewPipeline
from libs.edit2 import KVEditor
from vllm.sampling_params import SamplingParams
from tqdm import tqdm
from transformers import AutoModelForCausalLM,AutoTokenizer
import torch
import os
import random
from sentence_transformers import SentenceTransformer
import numpy as np
from matplotlib import pyplot as plt
from evaluate import load
import seaborn as sns
from scipy.stats import zscore
import math
import re
import evaluate
import matplotlib
from matplotlib import font_manager 
import traceback
import multiprocessing as mp
from functools import partial
import uuid
import torch


# XSUM_KVCACHE_DIR="examples/pipeline/kvcache/xsum"
# os.makedirs(XSUM_KVCACHE_DIR,exist_ok=True)

class BenchmarkTest:
    
    TEMPLATE ={
        "Qwen/Qwen2.5-1.5B-Instruct":"""<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant. <|im_end|>\n
<|im_start|>user\nSummarize and condense the following text into a short single sentence.\n{text}\n<|im_end|>\n<|im_start|>assistant\n""",
"Qwen/Qwen2.5-7B-Instruct":"""<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant. <|im_end|>\n
<|im_start|>user\nSummarize and condense the following text into a short single sentence.\n{text}\n<|im_end|>\n<|im_start|>assistant\n""",
        "llama3":"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>
Summarize and condense the following text into a short single sentence.\n{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    }
    def __init__(self,model_name):
        self.model_name = model_name
        self.metric = load("rouge")
        self.template = self.TEMPLATE[model_name]
    
    def compute_metric(self,pred_output,target_output):
        return self.metric.compute(predictions=pred_output,references=target_output)
    
    
    def generate_kvcache(self,pipeline:KVShareNewPipeline,input_path,output_path,kvcache_save_dir,batch_size=8):
        data = json.load(open(input_path))
        # if os.path.exists(output_path):
        #     save_data = json.load(open(output_path))
        # else:
        save_data = {}
        os.makedirs(kvcache_save_dir,exist_ok=True)
        
        # 准备批处理数据
        batch_items = []

        
        # 收集需要处理的数据
        for item in tqdm(data, desc="收集待处理数据"):
            uid = item["uuid"]
            # if uid in save_data:
            #     continue
            batch_items.append(item)
        
        if not batch_items:
            print("没有需要处理的新数据")
            return
        
        sample_params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            # top_p=0.95,
            # top_k=0,
            # repetition_penalty=1.0,
        )
        # 批量处理
        for batch_idx in tqdm(range(0,len(batch_items),batch_size), desc="批量生成KV缓存"):
            batch_items_part = batch_items[batch_idx:batch_idx+batch_size]
            batch_prompts_part = [self.template.format(text=item["candidate_doc"]) for item in batch_items_part]
            batch_kvcaches, outputs, keys = pipeline.get_kvcache_by_full_compute(pipeline.model,sample_params,batch_prompts_part)
            
            for sub_batch_idx,kvcache in enumerate(batch_kvcaches):
                kvcache_save_path = os.path.join(kvcache_save_dir,f"{batch_items_part[sub_batch_idx]['uuid']}.pt")
                torch.save(kvcache.detach().cpu(),kvcache_save_path)
                item = batch_items_part[sub_batch_idx]
                item["kvcache"] = kvcache_save_path
                item["candidate_doc"] = batch_prompts_part[sub_batch_idx]
                item["target_doc"] = self.template.format(text=batch_items_part[sub_batch_idx]["target_doc"])
                save_data[item["uuid"]] = item
        json.dump(save_data, open(output_path, "w"), indent=4)
        print(f"处理完成，已保存 {len(batch_items)} 条数据")
        
    def generate_with_cacheblend(self,pipeline:KVShareNewPipeline,input_path,output_path,kvcache_save_dir,batch_size=8):
        data = json.load(open(input_path))
       
        save_data = {}
        os.makedirs(kvcache_save_dir,exist_ok=True)
        
        # 准备批处理数据
        batch_items = []

        
        # 收集需要处理的数据
        for key,item in tqdm(data.items(), desc="收集待处理数据"):
            uid = item["uuid"]
            # if uid in save_data:
            #     continue
            batch_items.append(item)
        
        if not batch_items:
            print("没有需要处理的新数据")
            return
        
        sample_params = SamplingParams(
            max_tokens=512,
            temperature=0.0,
        )
        # 批量处理
        tokenizer = pipeline.model.get_tokenizer()
        max_request_id = 0
        all_scores = []
        for batch_idx in range(0,len(batch_items),batch_size):
            batch_items_part = batch_items[batch_idx:batch_idx+batch_size]
            
            batch_candidate_token_ids = [tokenizer.encode(item["candidate_doc"]) for item in batch_items_part]
            batch_target_token_ids =  [tokenizer.encode(item["target_doc"]) for item in batch_items_part]
            batch_target_prompts =   [item["target_doc"] for item in batch_items_part]
            batch_candidate_kvcache = [
                torch.load(item["kvcache"], weights_only=True) for item in batch_items_part
            ]
            batch_target_kvcache,batch_reused_map_indices,batch_unreused_map_indices= KVEditor.batch_kvedit_v2(
                batch_target_token_ids,
                batch_candidate_token_ids,
            batch_candidate_kvcache,
            tokenizer=None,
            window_size=5)
            next_batch_request_ids = [max_request_id+i for i in range(len(batch_target_prompts))]
            batch_pc_outputs = pipeline.partial_compute(
                pipeline.model,
                sample_params,
                batch_target_prompts,
                batch_target_kvcache,
                batch_reused_map_indices,
                batch_unreused_map_indices,
                next_batch_request_ids,
                enable_kvshare=False,
                enable_cacheblend=True,
                enable_only_compute_unreused=False,
                has_additional_value_error = False,
                las_additional_value_error = False,
                enable_compute_as=False
            )
            max_request_id = max([int(pc_outputs.request_id) for pc_outputs in batch_pc_outputs])+1
            for sub_batch_idx,output in enumerate(batch_pc_outputs):
                item = batch_items_part[sub_batch_idx]
                item["caceblend_output"] = output.outputs[0].text
                item["score"] = self.metric.compute(predictions=[item["caceblend_output"]],references=[item["answer"]])["rougeL"]
                save_data[item["uuid"]] = item
                all_scores.append(item["score"])
        json.dump(save_data, open(output_path, "w"), indent=4)
        print(f"处理完成{len(batch_items)} 条数据, 平均: {sum(all_scores)/len(all_scores)}")
    
    
    def generate_full_compute(self,pipeline:KVShareNewPipeline,input_path,output_path,kvcache_save_dir,batch_size=8):
        
        data = json.load(open(input_path))
       
        save_data = {}
        os.makedirs(kvcache_save_dir,exist_ok=True)
        
        # 准备批处理数据
        batch_items = []

        
        # 收集需要处理的数据
        for key,item in tqdm(data.items(), desc="收集待处理数据"):
            uid = item["uuid"]
            # if uid in save_data:
            #     continue
            batch_items.append(item)
        
        if not batch_items:
            print("没有需要处理的新数据")
            return
        
        sample_params = SamplingParams(
            max_tokens=512,
            temperature=0.0,
        )
        all_scores = []
        for batch_idx in range(0,len(batch_items),batch_size):
            batch_items_part = batch_items[batch_idx:batch_idx+batch_size]

            batch_target_prompts =   [item["target_doc"] for item in batch_items_part]
            batch_fc_outputs = pipeline.full_compute(
                pipeline.model,
                sample_params,
                batch_target_prompts,
            )
            for sub_batch_idx,output in enumerate(batch_fc_outputs):
                item = batch_items_part[sub_batch_idx]
                item["full_compute_output"] = output.outputs[0].text
                item["full_compute_score"] = self.metric.compute(predictions=[item["full_compute_output"]],references=[item["answer"]])["rougeL"]
                save_data[item["uuid"]] = item
                all_scores.append(item["full_compute_score"])
        json.dump(save_data, open(output_path, "w"), indent=4)
        print(f"处理完成{len(batch_items)} 条数据, FULL COMPUTE平均: {sum(all_scores)/len(all_scores)}")
    
        
        
        
    
    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["VLLM_USE_MODELSCOPE"]="True"

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    benchmark_xsum = "examples/dataset/data/xsum/benchmark_xsum.json"
    kvcache_path = "examples/pipeline/kvcache/xsum"
    benchmark_xsum_with_kvcache = "examples/dataset/data/xsum/benchmark_xsum_qwen_kvcache.json"
    benchmark_xsum_cacheblend = "examples/dataset/data/xsum/benchmark_xsum_cachblend.json"
    benchmark_xsum_full_compute = "examples/dataset/data/xsum/benchmark_xsum_full_compute.json"
    pipeline = KVShareNewPipeline(model_name,device="cuda:0")
    benchmark_test = BenchmarkTest(model_name)
    
    benchmark_test.generate_kvcache(pipeline, benchmark_xsum, benchmark_xsum_with_kvcache, kvcache_path,batch_size=16)
    
    benchmark_test.generate_full_compute(pipeline, benchmark_xsum_with_kvcache, benchmark_xsum_full_compute, kvcache_path,batch_size=16)
    
    # benchmark_test.generate_with_cacheblend(
    #     pipeline, benchmark_xsum_with_kvcache, benchmark_xsum_cacheblend, kvcache_path,batch_size=4
    # )
