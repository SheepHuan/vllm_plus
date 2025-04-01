import torch
import json

import json
from vllm import LLM,SamplingParams
from tqdm import tqdm
import os
from libs.pipeline import KVShareNewPipeline
from libs.edit import KVEditor
from transformers import AutoTokenizer
import random

template_text = "<|im_start|>You are Qwen, created by Alibaba. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{user_text}\n<|im_end|>\n<|im_start|>assistant\n"

def batch_demo(batch_source_prompt,batch_target_prompt,model_name="Qwen/Qwen2.5-1.5B-Instruct"):
    pipeline = KVShareNewPipeline(model_name)
    batch_target_prompt = [template_text.format(user_text=prompt) for prompt in batch_target_prompt]  
    batch_source_prompt = [template_text.format(user_text=prompt) for prompt in batch_source_prompt]  
   
    full_compute_target_outputs = KVShareNewPipeline.batch_full_compute(pipeline.model,SamplingParams(temperature=0.0,max_tokens=512),batch_target_prompt)

    batch_target_token_ids = [output.prompt_token_ids for output in full_compute_target_outputs]
    
    
    batch_source_key_values,batch_source_outputs = KVShareNewPipeline.get_kvcache_by_full_compute(pipeline.model,SamplingParams(temperature=0.0,max_tokens=1),batch_source_prompt)
    
    batch_source_token_ids = [source_output.prompt_token_ids for source_output in batch_source_outputs]

    target_kvcache,reused_map_indices,unreused_map_indices,sample_selected_token_indices = KVEditor.batch_kvedit(batch_target_token_ids,batch_source_token_ids,batch_source_key_values)
    
    
    partial_batch_target_outputs= KVShareNewPipeline.partial_compute(pipeline.model,SamplingParams(temperature=0.0,max_tokens=512),batch_target_prompt,reused_map_indices,unreused_map_indices,sample_selected_token_indices,target_kvcache)
    
    for full_compute_output,partial_compute_output in zip(full_compute_target_outputs,partial_batch_target_outputs):
        partial_ttft = partial_compute_output.metrics.first_token_time-partial_compute_output.metrics.first_scheduled_time
        partial_output = partial_compute_output.outputs[0].text
        
        # full_compute_output = full_compute_output.outputs[0].text
        full_compute_ttft = full_compute_output.metrics.first_token_time-full_compute_output.metrics.first_scheduled_time
        
        # print(partial_compute_output)
        # print(full_compute_output)
        # print(partial_ttft)
        # print(full_compute_ttft)
        print(partial_output)
        print(full_compute_output.outputs[0].text)
        



if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    source_prompt = ["I come from China. My name is Huan.","I come from Japan. My name is Wu."]
    target_prompt = ["I come from Japan. My name is Wu.","I come from Japan. My name is Wu."]
    batch_demo(source_prompt,target_prompt)