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

def compute_f1(a_pred, a_gold, tokenizer):
    a_pred = parse_generation(a_pred)
    gold_toks = tokenizer.encode(normalize_answer(a_gold))[1:]
    pred_toks = tokenizer.encode(normalize_answer(a_pred))[1:]
    #gold_toks = tokenizer.encode_chat_completion(ChatCompletionRequest(messages=[UserMessage(content=normalize_answer(a_gold))])).tokens[4:-4]
    #pred_toks = tokenizer.encode_chat_completion(ChatCompletionRequest(messages=[UserMessage(content=normalize_answer(a_pred))])).tokens[4:-4]
    #pdb.set_trace()
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

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
        return compute_f1(pred_output, target_output, self.tokenizer)
    
    

    
        
        
        
    
    
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
    #     pipeline, benchmark_opus_with_kvcache, benchmark_opus_cacheblend, kvcache_path,batch_size=8,
    #     cacheblend_recomp_ratio=0.60
    # ) 
    benchmark_test.generate_with_kvshare(
        pipeline, benchmark_opus_with_kvcache, benchmark_opus_kvshare, kvcache_path,batch_size=16,
        enable_kvshare_decode=False,
        has_top_ratio=0.6
    ) 
    # benchmark_test.generate_with_only_compute_unreused(
    #     pipeline, benchmark_opus_with_kvcache, benchmark_opus_only_compute_unreused, kvcache_path,batch_size=8
    # ) 