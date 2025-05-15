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
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

from collections import Counter

def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def qa_f1_score(prediction, ground_truth, **kwargs):
    # 尝试提取 |Answer: xxxx| 格式的答案
    answer_match = re.search(r'\|Answer:\s*(.*?)\|', prediction)
    if answer_match:
        prediction = answer_match.group(1).strip()
    
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)


# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
# def compute_f1(pred_text, ans_text):
#     token1 = tokenizer.encode(pred_text.lower())
#     token2 = tokenizer.encode(ans_text.lower())
    
#     return compute_f1_token(token1,token2)


class MUSIQUEBenchmarkTest(BenchmarkTest):
    
  
    def __init__(self,model_name):
        super().__init__(model_name)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        # self.template = self.TEMPLATE[model_name]
    
    def compute_metric(self,pred_output,target_output):
        return qa_f1_score(pred_output, target_output)
    
    

    
        
        
        
    
    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["VLLM_USE_MODELSCOPE"]="True"

    # model_name = "Qwen/Qwen2.5-7B-Instruct"
    # model_name = "01ai/Yi-34B-Chat-4bits"
    model_name = "LLM-Research/Meta-Llama-3.1-8B-Instruct"
    # batch_size = 16
    # max_model_len = 4096
    
    benchmark_opus = "examples/dataset/data/musique/musique_benchmark.json"
    kvcache_path = "examples/pipeline/kvcache/musique"
    benchmark_opus_with_kvcache = "examples/dataset/data/musique/musique_benchmark_gpt_chunk_kvcache.json"
    benchmark_opus_cacheblend = "examples/dataset/data/musique/musique_benchmark_cachblend.json"
    benchmark_opus_full_compute = "examples/dataset/data/musique/musique_benchmark_full_compute.json"
    benchmark_opus_kvshare = "examples/dataset/data/musique/musique_benchmark_kvshare.json"
    benchmark_opus_only_compute_unreused = "examples/dataset/data/musique/musique_benchmark_only_compute_unreused.json"
    pipeline = KVShareNewPipeline(model_name)
    benchmark_test = MUSIQUEBenchmarkTest(model_name)
    
    # benchmark_test.generate_kvcache(pipeline, benchmark_opus, benchmark_opus_with_kvcache, kvcache_path,batch_size=32)
    
    # benchmark_test.generate_full_compute(pipeline, benchmark_opus, benchmark_opus_full_compute,batch_size=32,max_tokens=16)
    
    benchmark_test.generate_with_cacheblend(
        pipeline, benchmark_opus_with_kvcache, benchmark_opus_cacheblend, kvcache_path,batch_size=16,
        cacheblend_recomp_ratio=0.40,enable_cacheblend_decode=False,max_tokens=16
    ) 
    # benchmark_test.generate_with_kvshare(
    #     pipeline, benchmark_opus_with_kvcache, benchmark_opus_kvshare, kvcache_path,batch_size=16,
    #     enable_kvshare_decode=True,
    #     has_top_ratio=0.20
    # ) 
    # benchmark_test.generate_with_only_compute_unreused(
    #     pipeline, benchmark_opus_with_kvcache, benchmark_opus_only_compute_unreused, kvcache_path,batch_size=16
    # ) 