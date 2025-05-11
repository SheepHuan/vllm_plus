import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datasets
from transformers import AutoTokenizer
import random
import spacy
from lm_eval.tasks import TaskManager

def load_cot_data(save_path,max_num=512):
    dataset = datasets.load_dataset("/root/.cache/huggingface/hub/datasets--ucinlp--drop/snapshots/95cda593fae71b60b5b19f82de3fcf3298c1239c", split="train",trust_remote_code=True)
    data = []
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    for item in dataset:   
        pass 
        passage = item["passage"]
        question = item["question"]
        answer = item["answers_spans"]
        tokens = tokenizer.encode(passage)
        if len(answer["spans"]) > 1:
            continue
        data.append(
            {
                "answer": answer,
                "target_doc": "\t "+passage +" \n "+question+" Think step by step, then write a line of the form \"Answer: $ANSWER\" at the end of your response. \t",
                "candidate_doc": "\t "+passage +" \n "+question+" \t",
            }
        )
    data = random.sample(data,max_num)
    json.dump(data,open(save_path,"w"),indent=4,ensure_ascii=False)

def random_split_chunks(text: str, chunk_size: int, num_permutations: int = 10):
    """
    生成所有可能的排列序列的函数

    参数：
    text: 原始文本字符串
    chunk_size: 初始分块的token数（N）
    num_permutations: 需要生成的排列序列数量

    返回：
    包含多个排列序列的列表以及每个排列序列对应的打乱顺序
    """
    nlp = spacy.load('en_core_web_sm')  # 加载英文模型
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    
    sentences = [" \n\n\n\n\n\n\n\n\n\n\n\n\n\n 请将下面的内容逐字逐句的翻译成中文 \n\n\n\n\n\n\n\n\n\n\n\n\n\n " + item + " \n\n\n\n\n\n\n\n\n\n\n\n\n\n " for item in sentences]
    return sentences,[]

def chunk_data(data):
    new_data = []
    for item in data:
        # 生成多个打乱序列
        text:str = item["candidate_doc"]
        
        setnece,order = random_split_chunks(text,32)
        item["candidates"] = setnece
        new_data.append(item)
    return new_data


if __name__ == "__main__":
    os.makedirs("examples/dataset/data/drop",exist_ok=True)
    save_path = "examples/dataset/data/drop/sim_drop_benchmark_dataset.json"
    chunk_save_path = "examples/dataset/data/drop/sim_drop_benchmark_dataset_chunk.json"
    # load_cot_data(save_path=save_path,max_num=128)

    
    data = json.load(open(save_path,"r"))
    data = chunk_data(data)
    json.dump(data,open(chunk_save_path,"w"),indent=4,ensure_ascii=False)