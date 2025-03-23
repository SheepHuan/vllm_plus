import datasets
import transformers
from ppl import PerplexityMetric
from tqdm import tqdm
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import random


def group_samsum(input_path,save_path):
    raw_data = json.load(open(input_path,"r"))
    # 选择50个前缀
    # 选择50个后缀
    
    raw_data = random.sample(raw_data,100)
    prefix_data = raw_data[:50]
    suffix_data = raw_data[50:100]
    # mix_data = raw_data[100:]
    # 组合2500个内容
    new_data = []
    tag = ["summary","dialogue"]
    for prefix in prefix_data:
           
        for suffix in suffix_data:
            item = {}
            item[tag[0]] = prefix[tag[0]] + "\n" + suffix[tag[0]]
            item[tag[1]] = prefix[tag[1]] + "\n" + suffix[tag[1]]
            
            # replace \r\n to \n
            item[tag[0]] = item[tag[0]].replace("\r\n", "\n")
            item[tag[1]] = item[tag[1]].replace("\r\n", "\n")
            new_data.append(item)
    json.dump(new_data,open(save_path,"w"),indent=4)
    
    
    
    
    
if __name__ == "__main__":
    
    group_samsum("examples/dataset/data/samsum/train.json","examples/dataset/data/samsum_group.json")