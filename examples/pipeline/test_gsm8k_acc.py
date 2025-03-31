import json
from kvshare_new_pipeline import KVShareNewPipeline
from nll_demo import calculate_nll
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

def extract_answer(s):
    _PAT_LAST_DIGIT = re.compile(
        r"([+-])?(?=([0-9]|\.[0-9]))(0|([1-9](\d{0,2}(,\d{3})*)|\d*))?(\.\d*)?(?=\D|$)"
    )
    match = list(_PAT_LAST_DIGIT.finditer(s))
    if match:
        last_digit = match[-1].group().replace(",", "").replace("+", "").strip()
        # print(f"The last digit in {s} is {last_digit}")
    else:
        last_digit = None
        print(f"No digits found in {s!r}", flush=True)
    return last_digit

def is_correct(completion, answer):
    gold = extract_answer(answer)
    assert gold is not None, "No ground truth answer found in the document."

    def number_equal(answer, pred):
        if pred is None:
            return False
        try:
            return math.isclose(eval(answer), eval(pred), rel_tol=0, abs_tol=1e-4)
        except:
            print(
                f"cannot compare two numbers: answer={answer}, pred={pred}", flush=True
            )
            return False

    return number_equal(gold, extract_answer(completion))

def generate_output_data(input_path: str, output_path: str):
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    device = "cuda:0"
    pipeline = KVShareNewPipeline(model_name,device)
    
    with open(input_path, "r") as f:
        data = json.load(f)
    save_data = []
    
    template="""<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant. <|im_end|>\n
<|im_start|>user\n
Question: In 2004, there were 60 kids at a cookout. In 2005, half the number of kids came to the cookout as compared to 2004. In 2006, 2/3 as many kids came to the cookout as in 2005. How many kids came to the cookout in 2006?
Let's think step by step
In 2005, 60/2=30 kids came to the cookout.
In 2006, 30/3*2=20 kids came to the cookout.
The answer is 20

Question: Jame gets a raise to $20 per hour and works 40 hours a week.  His old job was $16 an hour for 25 hours per week.  How much more money does he make per year in his new job than the old job if he works 52 weeks a year?
Let's think step by step
He makes 20*40=$800 per week
He used to make 16*25=$400 per week
So his raise was 800-400=$400 per week
So he makes 400*52=$20,800 per year more
The answer is 20800

Question: {question}.\nLet's think step by step<|im_end|>\n<|im_start|>assistant\n """
    
    all_data = data["all_data"]
    similar_pairs = data["similar_pairs"]
    save_data = []
    key_list = list(similar_pairs.keys())
    sample_keys = random.sample(key_list,1000)
    for key,item in tqdm(similar_pairs.items(),total=len(similar_pairs)):
        if key not in sample_keys:
            continue
        
        try:
            question = all_data[str(item["id"])]["question"]
            answer = all_data[str(item["id"])]["answer"]
            
            # 1. Full Compute
            sampling_params = SamplingParams(temperature=0, max_tokens=256)
            target_text = template.format(question=question)
            full_compute_output,target_token_ids,ttft_time = KVShareNewPipeline.full_compute(pipeline.model,sampling_params,target_text)
            
            item["output"] = full_compute_output
            item["is_correct"] = is_correct(full_compute_output,answer)
           
            profile_similar_top5_docs = []
            for index in range(0,len(item["cosine_similarity_top5"]),1):
                if str(item["cosine_similarity_top5"][index]["id"]) == str(key):
                    continue
                source_doc = all_data[str(item["cosine_similarity_top5"][index]["id"])]
              
                source_text = source_doc["question"]
                source_text = template.format(question=source_text)
                target_text = template.format(question=question)
                
                sampling_params_only_one = SamplingParams(temperature=0, max_tokens=1)
                source_kvcache,source_token_ids = KVShareNewPipeline.get_kvcache_by_full_compute(
                    pipeline.model,sampling_params_only_one,source_text)
                
                diff_report = KVShareNewPipeline.find_texts_differences(source_token_ids,target_token_ids)
                modified_kvcache,reused_map_indices,unused_map_indices = KVShareNewPipeline.apply_changes2kvcache(
                    source_token_ids, target_token_ids, source_kvcache, diff_report)
                
                high_sim_output,_,_ = KVShareNewPipeline.partial_compute(
                    pipeline.model, sampling_params, target_text,
                    reused_map_indices, unused_map_indices, modified_kvcache)
                
                
                # source_doc["output"] = high_sim_output
                is_correct_high_sim = is_correct(high_sim_output,answer)
                profile_similar_top5_docs.append({
                    "id":item["cosine_similarity_top5"][index]["id"],
                    "output":high_sim_output,
                    "is_correct":is_correct_high_sim,
                    "cosine_similarity":item["cosine_similarity_top5"][index]["similarity"]
                })
            item["cosine_similarity_top5"] = profile_similar_top5_docs
            
            # 3. High Similarity Partial Compute
            profile_reused_token_num_top5_docs = []
            for index in range(0,len(item["reused_token_num_top5"]),1):
                if str(item["reused_token_num_top5"][index]["id"]) == str(key):
                    continue
                source_doc = all_data[str(item["reused_token_num_top5"][index]["id"])]
                source_text = source_doc["question"]
                source_text = template.format(question=source_text)
                target_text = template.format(question=question)
                
                sampling_params_only_one = SamplingParams(temperature=0, max_tokens=1)
                source_kvcache,source_token_ids = KVShareNewPipeline.get_kvcache_by_full_compute(
                    pipeline.model,sampling_params_only_one,source_text)
                
                diff_report = KVShareNewPipeline.find_texts_differences(source_token_ids,target_token_ids)
                modified_kvcache,reused_map_indices,unused_map_indices = KVShareNewPipeline.apply_changes2kvcache(
                    source_token_ids, target_token_ids, source_kvcache, diff_report)

                high_reused_token_output,_,_ = KVShareNewPipeline.partial_compute(
                    pipeline.model, sampling_params, target_text,
                    reused_map_indices, unused_map_indices, modified_kvcache)
                
                # item["output"] = high_sim_output
                is_correct_high_reused_token = is_correct(high_reused_token_output,answer)
                profile_reused_token_num_top5_docs.append({
                    "id":item["reused_token_num_top5"][index]["id"],
                    "output":high_reused_token_output,
                    "is_correct":is_correct_high_reused_token,
                    "reused_token_num":item["reused_token_num_top5"][index]["reused_token_num"]
                })
            item["reused_token_num_top5"] = profile_reused_token_num_top5_docs
            save_data.append(item)
        except Exception as e:
            print(e)
            continue
        
    json.dump(save_data,open(output_path,"w"),indent=4,ensure_ascii=False)
    
    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    generate_output_data(input_path="examples/dataset/data/gsm8k/gsm8k_dataset_similar_docs_top5.json",output_path="examples/dataset/data/gsm8k/gsm8k_dataset_similar_docs_top5_output.json")