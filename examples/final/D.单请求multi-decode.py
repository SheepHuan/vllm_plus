from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from typing import List
import evaluate
import random
from tqdm import tqdm
import sys
import math
import re
from libs.edit2 import KVEditor

from libs.pipeline import KVShareNewPipeline
from vllm import LLM,SamplingParams
import vllm.utils


def full_compute(pipeline:KVShareNewPipeline,
                 batch_candidate_docs:List[str],
                 batch_target_docs:List[str],
                enbale_has_token_error:bool=False,
                enbale_las_token_error:bool=False):
    template_text = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nPlease tell the answer of  following question: {prompt}\n<|im_end|>\n<|im_start|>assistant\n"
 
    # rouge = evaluate.load("rouge")
    # data = json.load(open("/root/code/kvshare/bench_data/benchmark_gsm8k.json","r"))
    # data = data[:48]
    normal_score = []
    batch_size = 16 # 设置批处理大小
    sampling_params = SamplingParams(temperature=0.0,max_tokens=512)
    tokenizer = pipeline.model.get_tokenizer()

    batch_candidate_prompts = [template_text.format(prompt=doc) for doc in batch_candidate_docs]
    batch_target_prompts = [template_text.format(prompt=doc) for doc in batch_target_docs]

    # with vllm.utils.cprofile_context("vllm_profile/full_compute.prof"):
    gt_outputs = pipeline.full_compute(
        pipeline.model,
        sampling_params,
        batch_target_prompts,
    )

    
    batch_candidate_kvcache,batch_candidate_outputs,_ = pipeline.get_kvcache_by_full_compute(
        pipeline.model,
        sampling_params,
        batch_candidate_prompts,
    )
    pass
    max_request_id = max([int(output.request_id) for output in batch_candidate_outputs])+1
    

    next_batch_request_ids = [max_request_id+ii for ii in range(len(batch_target_prompts))]
    
    batch_candidate_token_ids = [tokenizer.encode(prompt) for prompt in batch_candidate_prompts]
    batch_target_token_ids = [tokenizer.encode(prompt) for prompt in batch_target_prompts]
    
    # batch_candidate_kvcache = torch.stack(batch_candidate_kvcache,dim=0)
    batch_target_kvcache,batch_reused_map_indices,batch_unreused_map_indices= KVEditor.batch_kvedit_v2(
            batch_target_token_ids,
            batch_candidate_token_ids,
            batch_candidate_kvcache,
            tokenizer=None,
            window_size=5)
    # pipeline.model.start_profile()
    # with vllm.utils.cprofile_context("vllm_profile/partial_compute.prof"):
    batch_pc_outputs = pipeline.partial_compute(
        pipeline.model,
        sampling_params,
        batch_target_prompts,
        batch_target_kvcache,
        batch_reused_map_indices,
        batch_unreused_map_indices,
        next_batch_request_ids,
        enable_kvshare=True,
        enable_cacheblend=False,
        enable_only_compute_unreused=True,
        has_additional_value_error = enbale_has_token_error,
        las_additional_value_error = enbale_las_token_error,
        enable_compute_as=True
    )
    # pipeline.model.stop_profile()
    for gt_output,pc_output in zip(gt_outputs,batch_pc_outputs):
        print("==================="*10)
        print("******partial_compute_output******")
        print(f"time for first token:{(pc_output.metrics.first_token_time - gt_output.metrics.first_scheduled_time)*1000}ms")
        print(f"time for last token:{(pc_output.metrics.last_token_time - gt_output.metrics.first_scheduled_time)*1000}ms")
        print(pc_output.outputs[0].text)
        print("******full_compute_output******")
        print(f"time for first token:{(gt_output.metrics.first_token_time - gt_output.metrics.first_scheduled_time)*1000}ms")
        print(f"time for last token:{(gt_output.metrics.last_token_time - gt_output.metrics.first_scheduled_time)*1000}ms")
        print(gt_output.outputs[0].text)
        print("==================="*10)
if __name__ == "__main__":
    import os
    os.environ["VLLM_TORCH_PROFILER_DIR"] = "/root/code/vllm_plus/vllm_profile"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["VLLM_USE_MODELSCOPE"] = "True"
    # os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
    # model_name = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    # model_name = "LLM-Research/Meta-Llama-3.1-8B-Instruct"
    pipeline = KVShareNewPipeline(model_name,device="cuda:0")
    
    batch_target_docs = [
        "我来自中国安徽，我的家乡是安徽芜湖，你能帮我生成一段关于我的家乡的介绍吗？",
        "你叫什么名字，你都有那些功能？你帮我解决什么问题？",
        "中国上海和美国纽约，哪个城市更权威？"
    ]
    batch_candidate_docs = [
        "我来自中国湖南，我的家乡是湖南长沙，你能帮我生成一段关于我的家乡的介绍吗？",
        "你现在给你取名字叫做“华为小艺”，你告诉我，你都有那些功能？你帮我解决什么问题？",
        "中国北京和美国纽约，哪个城市更权威？"
    ]
    
    full_compute(pipeline,
                 batch_candidate_docs,
                 batch_target_docs,
                 enbale_has_token_error=False,
                 enbale_las_token_error=False)
    # model_name = "/root/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
    # model:CustomQwen2ForCausalLM = CustomQwen2ForCausalLM.from_pretrained(model_name,device_map="cuda",torch_dtype=torch.bfloat16).eval()
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # template_text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>Translate the following text from Chinese to English:\n{prompt}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
