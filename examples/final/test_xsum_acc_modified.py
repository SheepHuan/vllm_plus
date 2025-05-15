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
import argparse


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
        # self.template = self.TEMPLATE[model_name]
    
    def compute_metric(self,pred_output,target_output):
        return self.metric.compute(predictions=[pred_output],references=[target_output])["rougeL"]
    
    
    def generate_kvcache(self,pipeline:KVShareNewPipeline,input_path,output_path,kvcache_save_dir,batch_size=8):
        data = json.load(open(input_path))

        save_data =[]
        os.makedirs(kvcache_save_dir,exist_ok=True)
        
        # 准备批处理数据
        batch_items = []


        sample_params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
        )
        # 批量处理
        # special_token = pipeline.model.get_tokenizer().encode("\n , \n")
        for item in tqdm(data, desc="批量生成KV缓存"):
            candidate = item["candidates"]
            # batch_items_part = data["candidates"].values()[batch_idx:batch_idx+batch_size]
            # 第一个可以使用模板，后面不是使用模板
            batch_prompts_part = [text for idx,text in enumerate(candidate)]
            batch_kvcaches, outputs, keys = pipeline.get_kvcache_by_full_compute(pipeline.model,sample_params,batch_prompts_part)
            batch_candidate_kvcache = []
            for sub_batch_idx,kvcache in enumerate(batch_kvcaches):
                kvcache_save_path = os.path.join(kvcache_save_dir,f"{str(uuid.uuid4())}.pt")
                torch.save(kvcache.detach().cpu(),kvcache_save_path)
                batch_candidate_kvcache.append(kvcache_save_path)
            item["kvcache_path"] = batch_candidate_kvcache
            item["candidates"] = batch_prompts_part
            save_data.append(item)

        json.dump(save_data, open(output_path, "w"), indent=4,ensure_ascii=False)
        print(f"处理完成，已保存 {len(save_data)} 条数据")
        
    def generate_with_partial_compute(self,pipeline:KVShareNewPipeline,input_path,output_path,kvcache_save_dir,batch_size=8,
                                    enable_kvshare=False,
                                    enable_epic=False,
                                    enable_cacheblend=False,
                                    enable_only_compute_unreused=False,
                                    has_additional_value_error = False,
                                    las_additional_value_error = False,
                                    enable_compute_as=False,
                                    enable_kvshare_decode=False,
                                    enable_cacheblend_decode=False,
                                    cacheblend_recomp_ratio=0.15,
                                    has_top_ratio=0.15,
                                    max_tokens=512):
        data = json.load(open(input_path))
       
        save_data = []
        os.makedirs(kvcache_save_dir,exist_ok=True)
        
    
        sample_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
        )
        # 批量处理
        tokenizer = pipeline.model.get_tokenizer()
        max_request_id = 0
        all_scores = []
        for batch_idx in tqdm(range(0,len(data),batch_size), desc="批量生成缓存混合"):

            batch_items_part = data[batch_idx:batch_idx+batch_size]
            
            
            # 准备KVEDIT
            batch_candidate_token_ids = []
            batch_candidate_kvcache = []
            batch_target_token_ids = []
            batch_target_prompts= []
            batch_answer = []
            for item in batch_items_part:
                
                sub_candidate_token_ids = [tokenizer.encode(item["candidates"][i]) for i,text in enumerate(item["candidates"])]
                
                sub_candidate_kvcache = [torch.load(item["kvcache_path"][i], weights_only=True) for i,text in enumerate(item["candidates"])]
                if len(sub_candidate_token_ids)==0:
                    continue
                batch_candidate_token_ids.append(sub_candidate_token_ids)
                batch_candidate_kvcache.append(sub_candidate_kvcache)
                batch_target_prompts.append(item["target_doc"])
                batch_answer.append(item["answer"])
            
            batch_target_token_ids =  [tokenizer.encode(item) for item in batch_target_prompts]
            batch_target_kvcache,batch_reused_map_indices,batch_unreused_map_indices= KVEditor.batch_kvedit_v2(
                batch_target_token_ids,
                batch_candidate_token_ids,
                    batch_candidate_kvcache,
                    tokenizer=None,
                    window_size=6)
            # 计算复用率
            reused_rate = [len(batch_reused_map_indices[i])/len(batch_target_token_ids[i]) for i in range(len(batch_target_token_ids))]
            print(f"复用率: {reused_rate}")
            
            
            
            next_batch_request_ids = [max_request_id+i for i in range(len(batch_target_prompts))]
            batch_pc_outputs = pipeline.partial_compute(
                pipeline.model,
                sample_params,
                batch_target_prompts,
                batch_target_kvcache,
                batch_reused_map_indices,
                batch_unreused_map_indices,
                next_batch_request_ids,
                enable_kvshare=enable_kvshare,
                enable_cacheblend=enable_cacheblend,
                enable_only_compute_unreused=enable_only_compute_unreused,
                has_additional_value_error = has_additional_value_error,
                las_additional_value_error = las_additional_value_error,
                enable_compute_as=enable_compute_as,
                enable_kvshare_decode=enable_kvshare_decode,
                cacheblend_recomp_ratio = cacheblend_recomp_ratio,
                has_top_ratio = has_top_ratio,
                enable_cacheblend_decode=enable_cacheblend_decode,
                enable_epic=enable_epic
            )
            max_request_id = max([int(pc_outputs.request_id) for pc_outputs in batch_pc_outputs])+1
            for sub_batch_idx,output in enumerate(batch_pc_outputs):
                item = batch_items_part[sub_batch_idx]
                item["partial_compute_output"] = output.outputs[0].text
                item["score"] = self.compute_metric(item["partial_compute_output"],batch_answer[sub_batch_idx])
                # save_data[item["uid"]] = item
                save_data.append(item)
                all_scores.append(item["score"])
        json.dump(save_data, open(output_path, "w"), indent=4,ensure_ascii=False)
        print(f"enable_only_compute_unreused: {enable_only_compute_unreused}")
        print(f"enable_cacheblend: {enable_cacheblend}")
        print(f"enable_kvshare: {enable_kvshare}")
        # print(f"enable_compute_as: {enable_compute_as}")
        print(f"处理完成{len(data)} 条数据, 平均: {sum(all_scores)/len(all_scores)}")
    
    
    def generate_with_kvshare(self,pipeline:KVShareNewPipeline,input_path,output_path,kvcache_path,batch_size=8,enable_kvshare_decode=False,has_top_ratio=0.15,max_tokens=512):
        self.generate_with_partial_compute(pipeline,input_path,output_path,kvcache_path,batch_size=batch_size,
                                       enable_kvshare=True,
                                       enable_kvshare_decode=enable_kvshare_decode,
                                        enable_cacheblend=False,
                                        enable_only_compute_unreused=False,
                                        has_additional_value_error = False,
                                        las_additional_value_error = False,
                                        enable_compute_as=True,
                                        has_top_ratio=has_top_ratio,
                                        max_tokens=max_tokens)
    
    def generate_with_cacheblend(self,pipeline:KVShareNewPipeline,input_path,output_path,kvcache_path,batch_size=8,
                                 cacheblend_recomp_ratio=0.15,
                                 enable_cacheblend_decode=False,
                                 max_tokens=512):
        self.generate_with_partial_compute(pipeline,input_path,output_path,kvcache_path,batch_size=batch_size,
                enable_kvshare=False,
                enable_cacheblend=True,
                enable_only_compute_unreused=False,
                has_additional_value_error = False,
                las_additional_value_error = False,
                enable_compute_as=False,
                enable_cacheblend_decode=enable_cacheblend_decode,
                cacheblend_recomp_ratio=cacheblend_recomp_ratio,
                max_tokens=max_tokens)
        
    def generate_with_epic(self,pipeline:KVShareNewPipeline,input_path,output_path,kvcache_path,batch_size=8,
                                 cacheblend_recomp_ratio=0.15,
                                 enable_cacheblend_decode=False,
                                 max_tokens=512):
        self.generate_with_partial_compute(pipeline,input_path,output_path,kvcache_path,batch_size=batch_size,
                enable_kvshare=False,
                enable_cacheblend= False ,
                enable_epic=True,
                enable_only_compute_unreused=False,
                has_additional_value_error = False,
                las_additional_value_error = False,
                enable_compute_as=False,
                enable_cacheblend_decode=enable_cacheblend_decode,
                cacheblend_recomp_ratio=cacheblend_recomp_ratio,
                max_tokens=max_tokens)
        
    
    def generate_with_only_compute_unreused(self,pipeline:KVShareNewPipeline,input_path,output_path,kvcache_path,batch_size=8,max_tokens=512):
        self.generate_with_partial_compute(pipeline,input_path,output_path,kvcache_path,batch_size=batch_size,
                enable_kvshare=False,
                enable_cacheblend=False,
                enable_only_compute_unreused=True,
                enable_compute_as=False,
                max_tokens=max_tokens)
        
    def generate_full_compute(self,pipeline:KVShareNewPipeline,input_path,output_path,batch_size=8,max_tokens=512):
        
        data = json.load(open(input_path))
       
        save_data = []

        sample_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
        )
        all_scores = []
        for batch_idx in tqdm(range(0,len(data),batch_size), desc="批量-Full Compute"):
            batch_items_part = data[batch_idx:batch_idx+batch_size]

            batch_target_prompts =   [item["target_doc"] for item in batch_items_part]
            batch_fc_outputs = pipeline.full_compute(
                pipeline.model,
                sample_params,
                batch_target_prompts,
            )
            for sub_batch_idx,output in enumerate(batch_fc_outputs):
                item = batch_items_part[sub_batch_idx]
                item["full_compute_output"] = output.outputs[0].text
                item["full_compute_score"] = self.compute_metric(item["full_compute_output"],item["answer"])
                # save_data[item["uid"]] = item
                save_data.append(item)
                all_scores.append(item["full_compute_score"])
        json.dump(save_data, open(output_path, "w"), indent=4,ensure_ascii=False)
        print(f"处理完成{len(all_scores)} 条数据, FULL COMPUTE平均: {sum(all_scores)/len(all_scores)}")
    
        
        
        
    
    
def generate_output_filename(base_path, prefix, args):
    """根据参数生成输出文件名"""
    params = []
    
    # 添加模型信息
    model_name = args.model.split('/')[-1]
    params.append(f"model_{model_name}")
    
    # 添加批处理大小
    params.append(f"bs_{args.batch_size}")
    
    # 根据不同的生成方法添加参数
    if args.generate_cacheblend:
        params.append(f"cacheblend_{args.cacheblend_recomp_ratio}")
        if args.enable_cacheblend_decode:
            params.append("cacheblend_decode")
    
    if args.generate_kvshare:
        params.append(f"kvshare_{args.has_top_ratio}")
        if args.enable_kvshare_decode:
            params.append("kvshare_decode")
    
    if args.generate_only_compute_unreused:
        params.append("only_unreused")
    
    # 组合所有参数
    param_str = "_".join(params)
    filename = f"{prefix}_{param_str}.json"
    return os.path.join(os.path.dirname(base_path), filename)

def cli(benchmark_path, kvshare_save_path, benchmark_cls=BenchmarkTest):
    parser = argparse.ArgumentParser(description='运行不同的生成方法')
    parser.add_argument('--gpu', type=str, default="1", help='使用的GPU ID')
    parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-7B-Instruct", help='使用的模型名称')
    parser.add_argument('--batch_size', type=int, default=16, help='批处理大小')
    
    # 添加各种生成方法的参数
    parser.add_argument('--generate_kvcache', action='store_true', help='是否生成KV缓存')
    parser.add_argument('--generate_full_compute', action='store_true', help='是否执行完整计算')
    parser.add_argument('--generate_cacheblend', action='store_true', help='是否执行缓存混合')
    parser.add_argument('--generate_kvshare', action='store_true', help='是否执行KV共享')
    parser.add_argument('--generate_only_compute_unreused', action='store_true', help='是否只计算未重用的部分')
    
    # 添加可选参数
    parser.add_argument('--cacheblend_recomp_ratio', type=float, default=0.15, help='缓存混合重计算比例')
    parser.add_argument('--has_top_ratio', type=float, default=0.15, help='KV共享的top比例')
    parser.add_argument('--enable_kvshare_decode', action='store_true', help='是否启用KV共享解码')
    parser.add_argument('--enable_cacheblend_decode', action='store_true', help='是否启用缓存混合解码')
    
    args = parser.parse_args()
    
    # 设置环境变量
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["VLLM_USE_MODELSCOPE"] = "True"
    
    # 设置文件路径
    benchmark_xsum = benchmark_path
    kvcache_path = kvshare_save_path
    
    # 生成带参数的文件名
    benchmark_xsum_with_kvcache = generate_output_filename(benchmark_path, "benchmark_xsum_kvcache", args)
    benchmark_xsum_cacheblend = generate_output_filename(benchmark_path, "benchmark_xsum_cacheblend", args)
    benchmark_xsum_full_compute = generate_output_filename(benchmark_path, "benchmark_xsum_full_compute", args)
    benchmark_xsum_kvshare = generate_output_filename(benchmark_path, "benchmark_xsum_kvshare", args)
    benchmark_xsum_only_compute_unreused = generate_output_filename(benchmark_path, "benchmark_xsum_only_compute_unreused", args)
    
    # 初始化pipeline和benchmark
    pipeline = KVShareNewPipeline(args.model, device="cuda:0")
    benchmark_test = benchmark_cls(args.model)
    
    # 根据参数执行相应的函数
    if args.generate_kvcache:
        print("正在生成KV缓存...")
        benchmark_test.generate_kvcache(pipeline, benchmark_xsum, benchmark_xsum_with_kvcache, kvcache_path, batch_size=args.batch_size)
    
    if args.generate_full_compute:
        print("正在执行完整计算...")
        benchmark_test.generate_full_compute(pipeline, benchmark_xsum_with_kvcache, benchmark_xsum_full_compute, batch_size=args.batch_size)
    
    if args.generate_cacheblend:
        print("正在执行缓存混合...")
        benchmark_test.generate_with_cacheblend(
            pipeline, 
            benchmark_xsum_with_kvcache, 
            benchmark_xsum_cacheblend, 
            kvcache_path,
            batch_size=args.batch_size,
            cacheblend_recomp_ratio=args.cacheblend_recomp_ratio,
            enable_cacheblend_decode=args.enable_cacheblend_decode
        )
    
    if args.generate_kvshare:
        print("正在执行KV共享...")
        benchmark_test.generate_with_kvshare(
            pipeline, 
            benchmark_xsum_with_kvcache, 
            benchmark_xsum_kvshare, 
            kvcache_path,
            batch_size=args.batch_size,
            has_top_ratio=args.has_top_ratio,
            enable_kvshare_decode=args.enable_kvshare_decode
        )
    
    if args.generate_only_compute_unreused:
        print("正在执行仅计算未重用部分...")
        benchmark_test.generate_with_only_compute_unreused(
            pipeline, 
            benchmark_xsum_with_kvcache, 
            benchmark_xsum_only_compute_unreused, 
            kvcache_path,
            batch_size=args.batch_size
        )

if __name__ == "__main__":
    benchmark_path = "examples/dataset/data/xsum/benchmark_xsum.json"
    kvshare_save_path = "examples/pipeline/kvcache/xsum"
    cli(benchmark_path, kvshare_save_path, BenchmarkTest)