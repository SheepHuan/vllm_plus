from transformers import AutoModelForCausalLM, AutoTokenizer,TextIteratorStreamer
import torch
import json
import tqdm
import evaluate
import logging
import re
import colorlog
import time
import numpy as np

import time

import os
import sys
sys.path.append("examples/bench_cache")
from pylibs.hash_cache import HashCache
from pylibs.prefix_cache import PrefixCache

import torch

def init_logger():
    handler = colorlog.StreamHandler()
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)s: %(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    handler.setFormatter(formatter)
    logger = colorlog.getLogger("benchmark")
    logger.addHandler(handler)
    logger.setLevel(colorlog.DEBUG)
    return logger

logger = init_logger()



def benchmark(model: AutoModelForCausalLM,tokenizer: AutoTokenizer, prompt, cache=None):
        past_key_values = None
        if cache is not None:
            if isinstance(cache,HashCache):
                chunks,_ = cache.split_prompt(prompt,tokenizer)
                past_key_values,query = cache.match(chunks)
                if past_key_values is not None:
                    print("cache hit")
            else:
                query = prompt
        else:
            query = prompt
            cached_chunks = []
            hit_rate = 0
        model_inputs = tokenizer(query, return_tensors="pt").to(model.device)
        if cache is not None:
            if isinstance(cache,PrefixCache):
                past_key_values,_ = cache.match(model_inputs.input_ids[0].tolist())
                if past_key_values is not None:
                    print("cache token len: ",past_key_values[0][0].shape[2])
        generate_kwargs = {
            'max_new_tokens': 1024,
            'past_key_values': past_key_values,
            'pad_token_id': tokenizer.eos_token_id,
            'top_p': 0.95,
            'temperature': 0.1,
            'repetition_penalty': 1.0,
            'top_k': 50,
            "return_dict_in_generate":True,
        }

        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

            cache_position = torch.arange(
                past_length if past_length == model_inputs.input_ids.shape[1] else past_length, model_inputs.input_ids.shape[1], device=model_inputs.input_ids.device
            )
            generate_kwargs['cache_position'] = cache_position
 
        # generate_kwargs['max_new_tokens'] = 1
        ttft_list = []
        # for i in range(20):
        #     start_time = time.time()
        #     with torch.no_grad():
        #         outputs = model.generate(**model_inputs, **generate_kwargs)
        #     ttft = time.time() - start_time
        #     ttft_list.append(ttft)
        print(ttft_list)
        avg_ttft = np.mean(ttft_list[3:-3])

        # 记录第一个 token 生成时间
        generate_kwargs['max_new_tokens'] = 1024
        outputs = model.generate(**model_inputs, **generate_kwargs)
        past_key_values = outputs.past_key_values
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, outputs.sequences)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        if cache is not None:
            if isinstance(cache,HashCache):
                chunks,tokens_list = cache.split_prompt(prompt,tokenizer)
                cache.insert(chunks[:-2],tokens_list[:-2],past_key_values)
            elif isinstance(cache,PrefixCache):
                token_len = model_inputs.input_ids.shape[1]
                saved_key_values = []
                for i in range(len(past_key_values)):
                    saved_key_values.append([past_key_values[i][0][:,:,:token_len,:],past_key_values[i][1][:,:,:token_len,:]])
                cache.insert(model_inputs.input_ids[:,:token_len][0].tolist(),saved_key_values)
        
        
        return response,past_key_values,avg_ttft,[],0




def benmark_transformers(model,
                         tokenizer,
                         wmt_path, 
                         save_path, 
                         enable_semantic_cache=False,
                         enable_prefix_cache=False,
                         disable_cache=False):

    dataset = json.load(open(wmt_path, 'r'))
    if not disable_cache:
        if enable_semantic_cache:
            cache = HashCache(disable=not enable_semantic_cache)
        elif enable_prefix_cache:
            cache = PrefixCache(disable=not enable_prefix_cache)
    else:
        cache = None

    save_data ={}
    for key, data_list in dataset.items():
        print(key)

        save_data[key] = []
        for item in tqdm.tqdm(data_list, desc="Processing", unit="item"):
            en = item['en']
            zh = item['zh']
            conversation = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": f"Translate the following Chinese sentence to English: {zh}"
                }
            ]
            prompt = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True)
        
           
            response,past_key_values,avg_ttft,cached_chunks,hit_rate = benchmark(model, tokenizer, prompt, cache)

            save_data[key].append({
                'ref_en': en,
                'gen_en': response,
                "ref_zh": zh,
                "metrics": {
                    'ttft': avg_ttft,
                }
            })
        
    json.dump(save_data, open(save_path, 'w'), indent=4, ensure_ascii=False)


def benchmark_safe_transformers(model,
                         tokenizer,
                         enable_semantic_cache=False,
                         disable_cache=False):

    if not disable_cache:
        if enable_semantic_cache:
            cache = HashCache(disable=not enable_semantic_cache)
    else:
        cache = None

    conversation = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": f"My head was hit by a stone recently. How to treat a headache? Please give me json format."
        }
    ]
    prompt = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True)

    
    response,past_key_values,avg_ttft,cached_chunks,hit_rate = benchmark(model, tokenizer, prompt, cache)
    print("response: ",response)
    
    conversation = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": f"How to treat a headache? What should I do? I have high blood pressure."
        }
    ]
    prompt = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True)

    response,past_key_values,avg_ttft,cached_chunks,hit_rate = benchmark(model, tokenizer, prompt, cache)
    print("response: ",response)
    
    # conversation = [
    #     {
    #         "role": "system",
    #         "content": "You are a helpful assistant."
    #     },
    #     {
    #         "role": "user",
    #         "content": f"How to treat a headache? What should I do?"
    #     }
    # ]
    # prompt = tokenizer.apply_chat_template(
    #     conversation,
    #     tokenize=False,
    #     add_generation_prompt=True)

    
    # response,past_key_values,avg_ttft,cached_chunks,hit_rate = benchmark(model, tokenizer, prompt, cache)
    # print("response: ",response)


if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    # model_name = "/root/nfs/l40s_ssd_7t/modelscope/hub/LLM-Research/Meta-Llama-3-8B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name,device_map="cuda",torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name,device_map="cuda",torch_dtype=torch.bfloat16)
    
    enable_semantic_cache = True
    enable_prefix_cache = False
    disable_cache = False
    
    benchmark_safe_transformers(model,tokenizer,enable_semantic_cache,disable_cache)
    # base_model_name = model_name.split("/")[-1]
    
    # wmt_path = "examples/bench_cache/data/wmt19_zh_en.json"
    # if enable_semantic_cache:
    #     save_path = f"examples/bench_cache/eval/{base_model_name}_wmt_en_zh_semantic_cache.json"
    # elif enable_prefix_cache:
    #     save_path = f"examples/bench_cache/eval/{base_model_name}_wmt_en_zh_prefix_cache.json"
    # else:
    #     save_path = f"examples/bench_cache/eval/{base_model_name}_wmt_en_zh_no_cache.json"
    
    
    # benmark_transformers(model,tokenizer,wmt_path, 
    #                      save_path,
    #                      enable_semantic_cache,
    #                      enable_prefix_cache,
    #                      disable_cache)
    # # benmark_vllm_with_prefix_cache(model,tokenizer,wmt_path, save_path,True)