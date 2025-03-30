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
    "sentence-transformers/all-MiniLM-L6-v2"
]

class CustomEmbedder:
    def __init__(self, model_name,device: str = "cuda:1"):
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

    def query_embed(self, source: List[str], targets: List[str]):
        # query_token_ids = self.tokenize_texts(source)
        # texts_token_ids = self.tokenize_texts(targets)
        # print("query_token_ids: ",query_token_ids['input_ids'].shape)
        # print("texts_token_ids: ",texts_token_ids['input_ids'].shape)
        query_embed = self.embed_texts(source)
        texts_embed = self.embed_texts(targets)
        similarity = np.dot(query_embed[0], texts_embed[0]) / (
                np.linalg.norm(query_embed[0]) * np.linalg.norm(texts_embed[0])
            )
        # scores = (query_embed @ texts_embed.T) * 100
        return similarity
    
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


def delta(embedder: CustomEmbedder):
    text1 = "I like to eat apple!"
    text2 = "I love to eat banana!"
    text3 = "I don't like to eat banana!"
    
    out1 = embedder.query_embed([text1], [text2])
    out2 = embedder.query_embed([text1], [text3])
    out3 = embedder.query_embed([text2], [text3])
    print(out1)
    print(out2)
    print(out3)
    
    
if __name__ == "__main__":
    index = 3
    embedder = CustomEmbedder(model_name[index])
    delta(embedder)
    
    # profile_dataset(embedder,model_name[index], "examples/dataset/data/sharegpt90k_ppl.json", f"examples/dataset/data/sharegpt90k_{model_name[index].split('/')[-1]}_speed.json")
