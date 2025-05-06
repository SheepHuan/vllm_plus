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
            window_size=2)
    # 打印未复用的token
    for i in range(len(batch_target_token_ids)):
        # print(f"target_token_ids: {tokenizer.decode(batch_target_token_ids[i])}")
        print(f"unreused_token_ids: {tokenizer.decode([batch_target_token_ids[i][j] for j in batch_unreused_map_indices[i] if j != -1])}")
        print(f"reused_token_ids: {tokenizer.decode([batch_target_token_ids[i][j] for j in batch_reused_map_indices[i] if j != -1])}")
    
    sampling_params = SamplingParams(temperature=0.0,max_tokens=512)
    batch_pc_outputs = pipeline.partial_compute(
        pipeline.model,
        sampling_params,
        batch_target_prompts,
        batch_target_kvcache,
        batch_reused_map_indices,
        batch_unreused_map_indices,
        next_batch_request_ids,
        enable_kvshare=False,
        enable_cacheblend=True,
        enable_only_compute_unreused=False,
        has_additional_value_error = enbale_has_token_error,
        las_additional_value_error = enbale_las_token_error,
        enable_compute_as=True,
        cacheblend_recomp_ratio=0.20,
        has_top_ratio=0.20,
        enable_kvshare_decode=True
    )
    # # pipeline.model.stop_profile()
    for gt_output,pc_output in zip(gt_outputs,batch_pc_outputs):
        print("==================="*10)
        print("******partial_compute_output******")
        print(f"time for first token:{(pc_output.metrics.first_token_time - gt_output.metrics.first_scheduled_time)*1000}ms")
        print(f"time for last token:{(pc_output.metrics.last_token_time - gt_output.metrics.first_scheduled_time)*1000}ms")
        print(pc_output.outputs[0].text)

        print("==================="*10)
        print("******full_compute_output******")
        print(f"time for first token:{(gt_output.metrics.first_token_time - gt_output.metrics.first_scheduled_time)*1000}ms")
        print(f"time for last token:{(gt_output.metrics.last_token_time - gt_output.metrics.first_scheduled_time)*1000}ms")
        print(f"gt_output: {gt_output.outputs[0].text}")
        
if __name__ == "__main__":
    import os
    os.environ["VLLM_TORCH_PROFILER_DIR"] = "/root/code/vllm_plus/vllm_profile"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["VLLM_USE_MODELSCOPE"] = "True"
    # os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
    # model_name = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"
    # model_name = "Qwen/Qwen2.5-7B-Instruct"
    model_name = "LLM-Research/Meta-Llama-3.1-8B-Instruct"
    pipeline = KVShareNewPipeline(model_name,device="cuda:0")
    
    # batch_target_docs = [
    #     "Joe and Adam built a garden wall with three courses of bricks. They realized the wall was too low and added 2 more courses. If each course of the wall had 400 bricks, and they took out half of the bricks in the last course to allow easy checkup of the garden, calculate the total number of bricks the wall has."
    # ]
    # batch_candidate_docs = [
    #    [ 
    #     "\n Joe and Adam built a garden wall with three courses of bricks. They realized the wall was too low and added 1 more courses. ",
    #     "\n If each course of the wall had 5 bricks, and they took out half of the bricks in the last course to allow easy checkup of the garden, calculate the total number of bricks the wall has.",
    #     ]
    # ]
    
  
    target_doc = ["As one of the most successful shows on U.S. television history, American Idol has a strong impact not just on television, but also in the wider world of entertainment. It helped create a number of highly successful recording artists, such as Kelly Clarkson, Daughtry and Carrie Underwood, as well as others of varying notability.\n\n\"Already Gone\" is a song performed by American pop singer-songwriter Kelly Clarkson from her fourth studio album, \"All I Ever Wanted\". It is co-written by Clarkson and Ryan Tedder, who also produced it. The song was released as the album's third single in August 2009. What show helped launch the career of the performer who wrote the lyrics to Already Gone?"
    ]
    candidates =[[
            "\nAs one of the most successful shows on U.S. television history, American Idol has a strong impact not just on television, but also in the wider world of entertainment.\n",
            "\nIt helped create a number of highly successful recording artists, such as Kelly Clarkson, Daughtry and Carrie Underwood, as well as others of varying notability.\n",
            "\"Already Gone\" is a song performed by American pop singer-songwriter Kelly Clarkson from her fourth studio album, \"All I Ever Wanted\". It is co-written by Clarkson and Ryan Tedder, who also produced it. ",
            "\nThe song was released as the album's third single in August 2229. \n",
        ]
    ]
    
    full_compute(pipeline,
                 candidates,
                 target_doc,
                 enbale_has_token_error=False,
                 enbale_las_token_error=False)
