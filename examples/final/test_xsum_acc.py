import json
from libs.pipeline import KVShareNewPipeline
from libs.edit import KVEditor
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

def split_data_by_windows_size(input_path: str, output_path: str):
    with open(input_path, "r") as f:
        data = json.load(f)
    similar_docs = data["similar_docs"]
    all_data = data["all_documents"]
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    save_data = []
    windows_size = [25]
    global_id = 0
    # similar_docs = random.sample(similar_docs,min(len(similar_docs),100))
    for doc_item in tqdm(similar_docs,desc="Processing"):
        doc_item["id"] = global_id
        try:
            target_doc_tokens = tokenizer.encode(doc_item["document"])
            if len(target_doc_tokens) > 4096:
                continue
            term_items = []
            for reused_item in doc_item["similar_docs"]:
                reused_doc = all_data[str(reused_item["id"])]
                if reused_item["similarity"] > 0.9995:
                    continue
                reused_doc_tokens = tokenizer.encode(reused_doc["document"])
                reused_item["reused_token_num"] = {}
                for window_size in windows_size:
                    diff_report = KVEditor.find_text_differences(target_doc_tokens,reused_doc_tokens,window_size=window_size)
                    if len(diff_report["moves"]) ==0:
                        reused_item["reused_token_num"][window_size] = 0
                    else:
                        reused_item["reused_token_num"][window_size] = sum([move["to_position"][1]-move["to_position"][0]+1 for move in diff_report["moves"]])
                if reused_item["reused_token_num"][25] > 0:
                    term_items.append(reused_item)
        
            if len(term_items) == 0:
                continue
            else:
                simi_top1 = sorted(term_items,key=lambda x:x["similarity"],reverse=True)[0]
                reused_top1_w25 = sorted(term_items,key=lambda x:x["reused_token_num"][25],reverse=True)[0]
                doc_item["simi_top1"] = simi_top1["id"]
                doc_item["reused_items"] = term_items
                save_data.append({
                    "id": doc_item["id"],
                    "document": doc_item["document"],
                    "summary": doc_item["summary"],
                    "simi_top1": simi_top1["id"],
                    "reused_top1_w25": reused_top1_w25
                })
            global_id += 1
        except:
            continue
    data["similar_docs"] = save_data   
    print(f"处理后样本数量: {len(data['similar_docs'])}")
    json.dump(data, open(output_path, "w"), indent=4, ensure_ascii=False)
    
    
qwen_template="""<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant. <|im_end|>\n
<|im_start|>user\nSummarize and condense the following text into a short single sentence.\n{text}\n<|im_end|>\n<|im_start|>assistant\n"""
llama3_template_text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>Summarize and condense the following text into a short single sentence.\n{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

XSUM_KVCACHE_DIR="examples/pipeline/kvcache/xsum"
os.makedirs(XSUM_KVCACHE_DIR,exist_ok=True)

def generate_output_data(input_path: str, output_path: str, model_name = "Qwen/Qwen2.5-7B-Instruct", batch_size=4,window_size=3):
    device = "cuda:0"
    pipeline = KVShareNewPipeline(model_name, device)
    
    with open(input_path, "r") as f:
        data = json.load(f)
    save_data = []
    
    all_data = data["all_documents"]
    similar_pairs = data["similar_docs"]
    similar_pairs = [pair for pair in similar_pairs if pair["reused_top1_w25"]["similarity"] >= 0.59]
    

    # if os.path.exists(output_path):
    #     profile_data = json.load(open(output_path,"r"))["similar_docs"]
    #     profiled_id = [pair["id"] for pair in similar_pairs if profile_data]
    #     similar_pairs = [pair for pair in similar_pairs if pair["id"] not in profiled_id]
    # else:
    profile_data = []
    
    print(f"处理后样本数量: {len(similar_pairs)}")
    similar_pairs = random.sample(similar_pairs, min(len(similar_pairs),2000))
    
    
    
    
    save_data = []

    rouge = evaluate.load('rouge')
    tokenizer = pipeline.model.get_tokenizer()
    
    # 逐个处理数据，不再使用批量处理
    for item in tqdm(similar_pairs, desc="Processing items"):
        try:
            # if item["similarity"] > 0.95:
            #     continue
            # 准备prompt
            question = item["document"]
            answer = item["summary"]
            
            # 添加目标文本
            target_prompt = template.format(text=question)
            source_prompt = template.format(text=all_data[str(item["reused_top1_w25"]["id"])]["document"])
            if source_prompt == "":
                continue
            # 编码token
            target_token_ids = tokenizer.encode(target_prompt)
            
            source_cache_path = os.path.join(XSUM_KVCACHE_DIR,f"opus_kvcache_id-{item['reused_top1_w25']['id']}.pt")
            # 获取kv cache
            if os.path.exists(source_cache_path):
                source_key_values = torch.load(source_cache_path)
                source_token_ids = [tokenizer.encode(source_prompt)]
            else:
                source_key_values, source_outputs = KVShareNewPipeline.get_kvcache_by_full_compute(
                    pipeline.model,
                    SamplingParams(temperature=0, max_tokens=1),
                    [source_prompt]
                )
                torch.save(source_key_values, source_cache_path)
            
                source_token_ids = source_outputs[0].prompt_token_ids
            
            for window_size in [6,12,24]:
                # 单个样本的kvedit
                target_kvcache, reused_map_indices, unreused_map_indices, sample_selected_token_indices = KVEditor.batch_kvedit(
                    [target_token_ids],
                    [source_token_ids],
                    source_key_values,
                    window_size=window_size
                )
                
                # 单个样本的partial compute
                partial_outputs = KVShareNewPipeline.partial_compute(
                    pipeline.model,
                    SamplingParams(temperature=0, max_tokens=512),
                    [target_prompt],
                    reused_map_indices,
                    unreused_map_indices,
                    sample_selected_token_indices,
                    target_kvcache
                )

                try:
                    partial_output = partial_outputs[0].outputs[0].text
                    item["reused_top1_w25"][f"output_w{window_size}"] = partial_output
                    item["reused_top1_w25"][f"rouge_w{window_size}"] = rouge.compute(predictions=[partial_output], references=[answer])
                except Exception as e:
                    print(f"处理item时出错: {str(e)}")
                    continue
            save_data.append(item)   
        except Exception as e:
            print(f"处理样本时出错: {str(e)}")
            continue
            
    data["similar_docs"] = save_data+profile_data
    json.dump(data, open(output_path, "w"), indent=4, ensure_ascii=False)
    
    
    
    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["VLLM_USE_MODELSCOPE"]="True"
    # input_path = "examples/dataset/data/xsum/all-MiniLM-L6-v2_train_similar_docs_topk50.json"
    # output_path = "examples/dataset/data/xsum/xsum_dataset_similar_docs_top50_250403_windows.json"
    # split_data_by_windows_size(input_path,output_path)
    
    template = qwen_template
    model_name = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"
    # template = llama3_template_text
    input_path =  "examples/dataset/data/xsum/xsum_dataset_similar_docs_top50_250403_windows.json"
    output_path = "examples/dataset/data/xsum/xsum_dataset_similar_docs_top50_250403_windows_outputs.json"
    generate_output_data(input_path,output_path,model_name)