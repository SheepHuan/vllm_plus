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
    
if __name__ == "__main__":
    input_path = "examples/dataset/data/xsum/all-MiniLM-L6-v2_train_similar_docs_topk50.json"
    output_path = "examples/dataset/data/xsum/xsum_dataset_similar_docs_top50_250403_windows.json"
    split_data_by_windows_size(input_path,output_path)