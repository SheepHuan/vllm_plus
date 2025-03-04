import torch
import time
from sentence_transformers import SentenceTransformer
import tokenizers
from typing import List
import json
import numpy as np
from tqdm import tqdm

"""

huggingface-cli download --resume-download Alibaba-NLP/gte-Qwen2-7B-instruct 
huggingface-cli download --resume-download intfloat/multilingual-e5-large-instruct

huggingface-cli download --resume-download Alibaba-NLP/gte-Qwen2-1.5B-instruct
"""



model_name = [
    "Linq-AI-Research/Linq-Embed-Mistral",
    "Alibaba-NLP/gte-Qwen2-7B-instruct",
    "intfloat/multilingual-e5-large-instruct",
    "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
]

class CustomEmbedder:
    def __init__(self, model_name,device: str = "cuda:0"):
        self.model = SentenceTransformer(model_name).to(device).to(torch.bfloat16)

    def embed_texts(self, texts: List[str]):
        # start_time = time.time()
        embeddings = self.model.encode(texts)
        # end_time = time.time()
        # print(f"Time taken to embed {len(texts)} texts: {end_time - start_time} seconds")
        return embeddings
    
    def tokenize_texts(self, texts: List[str]):
        token_ids = self.model.tokenize(texts)
        return token_ids

    def query_embed(self, query: List[str], texts: List[str]):
        query_token_ids = self.tokenize_texts(query)
        texts_token_ids = self.tokenize_texts(texts)
        print("query_token_ids: ",query_token_ids['input_ids'].shape)
        print("texts_token_ids: ",texts_token_ids['input_ids'].shape)
        query_embed = self.embed_texts(query)
        texts_embed = self.embed_texts(texts)
        scores = (query_embed @ texts_embed.T) * 100
        return scores
    
    def profile_embed_texts(self, texts: str):
        # tokens = self.tokenize_texts([texts])
        # length = tokens['input_ids'].shape[1]
        start_time = time.time()
        self.embed_texts([texts])
        end_time = time.time()
        
        

        return end_time - start_time


def profile_dataset(embedder: CustomEmbedder,model_name: str, json_path: str, save_path: str):
    import random
    data = json.load(open(json_path))
    data = random.sample(data, 10000)
    speeds = []
    new_data = []
    for item in tqdm(data, desc="Processing items"):
        
        tokens = embedder.tokenize_texts([item['text']])
        length = tokens['input_ids'].shape[1]
        if length < 100:
            continue
        latency = embedder.profile_embed_texts(item['text'])
        speed = length / latency
        item[f'{model_name}_speed'] = speed
        speeds.append(speed)
        new_data.append(item)
    speed = np.array(speeds)
    print(f"Average speed: {speed.mean()} tokens/s")
    print(f"Median speed: {np.median(speed)} tokens/s")
    print(f"Max speed: {speed.max()} tokens/s")
    print(f"Min speed: {speed.min()} tokens/s")
    print(f"Std speed: {speed.std()} tokens/s")
    json.dump(new_data, open(save_path, "w"), indent=4)
    return speed

if __name__ == "__main__":
    index = 3
    embedder = CustomEmbedder(model_name[index])
    profile_dataset(embedder,model_name[index], "examples/dataset/data/sharegpt90k_ppl.json", f"examples/dataset/data/sharegpt90k_{model_name[index].split('/')[-1]}_speed.json")
