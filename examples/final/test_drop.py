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
from test_xsum_acc_modified import BenchmarkTest,cli
from drop_eval import drop_metric
import re


class DropBenchmarkTest(BenchmarkTest):
    
    
    def __init__(self,model_name):
        super().__init__(model_name)
        self.model_name = model_name
        self.metric = load("rouge")
        # self.template = self.TEMPLATE[model_name]
    
    def compute_metric(self,pred_output,target_output):
        # 检查
        ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"
        match = re.search(ANSWER_PATTERN, pred_output)
        extracted_answer = match.group(1) if match else pred_output
        
        span = target_output["spans"]
        em_score, f1_score = drop_metric(extracted_answer, span)
        if f1_score>0:
            return 1 
        else:
            return 0
        # return f1_score
        
if __name__ == "__main__":
    # benchmark_path = "examples/dataset/data/drop/sim_drop_benchmark_dataset_chunk.json"
    # kvshare_save_path = "examples/pipeline/kvcache/drop"
    # cli(benchmark_path, kvshare_save_path, DropBenchmarkTest)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["VLLM_USE_MODELSCOPE"]="True"

    # model_name = "Qwen/Qwen2.5-7B-Instruct"
    # model_name = "LLM-Research/Meta-Llama-3.1-8B-Instruct"
    model_name = "01ai/Yi-34B-Chat-4bits"
    batch_size = 16
    max_model_len = 4096
    
    benchmark_opus = "examples/dataset/data/drop/sim_drop_benchmark_dataset_chunk.json"
    kvcache_path = "examples/pipeline/kvcache/drop"
    benchmark_opus_with_kvcache = "examples/dataset/data/drop/sim_drop_benchmark_dataset_gpt_chunk_kvcache.json"
    benchmark_opus_cacheblend = "examples/dataset/data/drop/drop_benchmark_cachblend.json"
    benchmark_opus_full_compute = "examples/dataset/data/drop/drop_benchmark_full_compute.json"
    benchmark_opus_kvshare = "examples/dataset/data/drop/drop_benchmark_kvshare.json"
    benchmark_opus_only_compute_unreused = "examples/dataset/data/drop/drop_benchmark_only_compute_unreused.json"
    pipeline = KVShareNewPipeline(model_name,max_model_len=4096)
    benchmark_test = DropBenchmarkTest(model_name)
    
    # benchmark_test.generate_kvcache(pipeline, benchmark_opus, benchmark_opus_with_kvcache, kvcache_path,batch_size=16)
    
    # benchmark_test.generate_full_compute(pipeline, benchmark_opus, benchmark_opus_full_compute,batch_size=16,max_tokens=512)
    
    benchmark_test.generate_with_cacheblend(
        pipeline, benchmark_opus_with_kvcache, benchmark_opus_cacheblend, kvcache_path,batch_size=32,
        cacheblend_recomp_ratio=0.30,enable_cacheblend_decode=True,max_tokens=512
    ) 
    # benchmark_test.generate_with_kvshare(
    #     pipeline, benchmark_opus_with_kvcache, benchmark_opus_kvshare, kvcache_path,batch_size=32,
    #     enable_kvshare_decode=True,
    #     has_top_ratio=0.40,max_tokens=512
    # ) 
    # benchmark_test.generate_with_only_compute_unreused(
    #     pipeline, benchmark_opus_with_kvcache, benchmark_opus_only_compute_unreused, kvcache_path,batch_size=16
    # ) 