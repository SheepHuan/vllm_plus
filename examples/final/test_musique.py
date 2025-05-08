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
from test_xsum_acc_modified import BenchmarkTest
import collections
import string
from typing import List

def normalize_answer(s):
    """Normalize answer text for better matching."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def parse_generation(generation):
    """Parse generated text to extract the answer."""
    return generation.strip()

def compute_f1_token(pred_tokens: List[int], ans_tokens: List[int]) -> float:
    """基于Tokenizer输出计算F1分数（词表存在性判断版）
    
    Args:
        pred_tokens: 预测token的ID列表，如[101, 2023, 2003, ...]
        ans_tokens: 答案token的ID列表
        
    Returns:
        F1分数值，范围[0,1]
    """
    # 转换为集合消除重复（根据NLP常规评估逻辑）
    pred_set = set(pred_tokens)  
    ans_set = set(ans_tokens)
    
    # 计算匹配指标
    tp = len(pred_set & ans_set)   # 共同存在的唯一token数
    fp = len(pred_set - ans_set)   # 预测多余token数
    fn = len(ans_set - pred_set)   # 答案未覆盖token数
    
    # 防止除零错误的稳健计算（参考最佳实践[3,7](@ref)）
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # 调和平均计算（核心公式[2,6](@ref)）
    if (precision + recall) == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
def compute_f1(pred_text, ans_text):
    token1 = tokenizer.encode(pred_text.lower())
    token2 = tokenizer.encode(ans_text.lower())
    
    return compute_f1_token(token1,token2)


class MUSIQUEBenchmarkTest(BenchmarkTest):
    
    TEMPLATE ={
        "Qwen/Qwen2.5-1.5B-Instruct":
            """<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant. <|im_end|>\n
        <|im_start|>user\nText: {text}\n<|im_end|>\n<|im_start|>assistant\n""",
        "Qwen/Qwen2.5-7B-Instruct": 
            """<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant. <|im_end|>\n
        <|im_start|>user\nText: {text}\n<|im_end|>\n<|im_start|>assistant\n""",
        "LLM-Research/Meta-Llama-3.1-8B-Instruct":
        """<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\nText: {text}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        "01ai/Yi-34B-Chat-4bits":
            """<|im_start|>system\nYou are a helpful assistant. <|im_end|>\n<|im_start|>user\nText: {text}\n<|im_end|>\n<|im_start|>assistant\n""",
    }
    def __init__(self,model_name):
        super().__init__(model_name)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        self.template = self.TEMPLATE[model_name]
    
    def compute_metric(self,pred_output,target_output):
        return compute_f1(pred_output, target_output)
    
    

    
        
        
        
    
    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["VLLM_USE_MODELSCOPE"]="True"

    # model_name = "Qwen/Qwen2.5-7B-Instruct"
    # model_name = "01ai/Yi-34B-Chat-4bits"
    model_name = "LLM-Research/Meta-Llama-3.1-8B-Instruct"
    # batch_size = 16
    max_model_len = 4096
    
    benchmark_opus = "examples/dataset/data/musique/musique_benchmark_gpt_chunk.json"
    kvcache_path = "examples/pipeline/kvcache/musique"
    benchmark_opus_with_kvcache = "examples/dataset/data/musique/musique_benchmark_gpt_chunk_kvcache.json"
    benchmark_opus_cacheblend = "examples/dataset/data/musique/musique_benchmark_cachblend.json"
    benchmark_opus_full_compute = "examples/dataset/data/musique/musique_benchmark_full_compute.json"
    benchmark_opus_kvshare = "examples/dataset/data/musique/musique_benchmark_kvshare.json"
    benchmark_opus_only_compute_unreused = "examples/dataset/data/musique/musique_benchmark_only_compute_unreused.json"
    pipeline = KVShareNewPipeline(model_name,max_model_len=4096)
    benchmark_test = MUSIQUEBenchmarkTest(model_name)
    
    # benchmark_test.generate_kvcache(pipeline, benchmark_opus, benchmark_opus_with_kvcache, kvcache_path,batch_size=16)
    
    # benchmark_test.generate_full_compute(pipeline, benchmark_opus, benchmark_opus_full_compute,batch_size=16)
    
    # benchmark_test.generate_with_cacheblend(
    #     pipeline, benchmark_opus_with_kvcache, benchmark_opus_cacheblend, kvcache_path,batch_size=16,
    #     cacheblend_recomp_ratio=0.30
    # ) 
    benchmark_test.generate_with_kvshare(
        pipeline, benchmark_opus_with_kvcache, benchmark_opus_kvshare, kvcache_path,batch_size=16,
        enable_kvshare_decode=True,
        has_top_ratio=0.20
    ) 
    # benchmark_test.generate_with_only_compute_unreused(
    #     pipeline, benchmark_opus_with_kvcache, benchmark_opus_only_compute_unreused, kvcache_path,batch_size=16
    # ) 