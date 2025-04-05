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

template_text = "<|im_start|>You are Qwen, created by Alibaba. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n将下面的文本翻译成英文:\n{user_text}\n<|im_end|>\n<|im_start|>assistant\n"

def batch_demo(batch_source_prompt,batch_target_prompt,model_name="Qwen/Qwen2.5-1.5B-Instruct"):
    pipeline = KVShareNewPipeline(model_name,max_model_len=512)
    tokenizer = pipeline.model.get_tokenizer()
    batch_target_prompt = [template_text.format(user_text=prompt) for prompt in batch_target_prompt]  
    batch_source_prompt = [template_text.format(user_text=prompt) for prompt in batch_source_prompt]  
   
    batch_target_token_ids = [tokenizer.encode(prompt) for prompt in batch_target_prompt]

    batch_source_key_values,batch_source_outputs,batch_source_req_ids = KVShareNewPipeline.get_kvcache_by_full_compute(pipeline.model,SamplingParams(temperature=0.0,max_tokens=512),batch_source_prompt)
    guess_target_req_ids = [batch_source_req_ids[-1]+1+idx for idx in range(len(batch_target_prompt))]
    
    batch_source_token_ids = [source_output.prompt_token_ids for source_output in batch_source_outputs]
    target_kvcache,reused_map_indices,unreused_map_indices,sample_selected_token_indices,batch_target_slice_list = KVEditor.batch_kvedit(batch_target_token_ids,batch_source_token_ids,batch_source_key_values,window_size=15)
    
    
    batch_target_key_values,batch_target_outputs,batch_target_req_ids = KVShareNewPipeline.partial_compute(pipeline.model,
                                                                                                           SamplingParams(temperature=0.0,max_tokens=512),
                                                                                                           batch_target_prompt,
                                                                                                           target_kvcache,reused_map_indices,
                                                                                                           unreused_map_indices,
                                                                                                           sample_selected_token_indices,
                                                                                                           batch_target_slice_list,
                                                                                                           guess_target_req_ids,
                                                                                                           )
    # batch_source_token_ids = [source_output.prompt_token_ids for source_output in batch_source_outputs]

    # target_kvcache,reused_map_indices,unreused_map_indices,sample_selected_token_indices = KVEditor.batch_kvedit(batch_target_token_ids,batch_source_token_ids,batch_source_key_values,window_size=15)
    
    
    # partial_batch_target_outputs= KVShareNewPipeline.partial_compute(pipeline.model,SamplingParams(temperature=0.0,max_tokens=512),batch_target_prompt,reused_map_indices,unreused_map_indices,sample_selected_token_indices,target_kvcache)
    
    # for full_compute_output,partial_compute_output in zip(full_compute_target_outputs,partial_batch_target_outputs):
    #     partial_ttft = partial_compute_output.metrics.first_token_time-partial_compute_output.metrics.first_scheduled_time
    #     partial_output = partial_compute_output.outputs[0].text
        
    #     # full_compute_output = full_compute_output.outputs[0].text
    #     full_compute_ttft = full_compute_output.metrics.first_token_time-full_compute_output.metrics.first_scheduled_time
        
    #     # print(partial_compute_output)
    #     # print(full_compute_output)
    #     # print(partial_ttft)
    #     # print(full_compute_ttft)
    #     print(partial_output)
    #     print(full_compute_output.outputs[0].text)
        



if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["VLLM_USE_MODELSCOPE"]="True"
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    
    source_prompt = [
                     "在6月9日第二次会议上，世界卫生组织总干事，，，，联合国人权事务高级专员、欧洲经济委员会执行秘书以及世界银行、联合国难民事务高级专员办事处和国际农业发展基金的代表作了发言。",
                     "難道他們沒有在大地上旅行而觀察前人的結局是怎樣的嗎？。。。前人比他們努力更大，前人於地方的墾植和建設，勝過他們。他們族中的使者曾給他們帶來許多明証，故真主未虧枉他們，但他們虧枉了自己，",
                     "众人啊！使者确已昭示你们从你们的主降示的真理，故你们当确信他，这对于你们是有益的。如果你们不信道，（那末，真主是无需求你们的）， 因为天地万物，确是真主的。真主是全知的，是至睿的。",
                     ]
    target_prompt = ["将这段文本翻译成英文，11. 在6月9日第二次会议上，世界卫生组织总干事、联合国人权事务高级专员、欧洲经济委员会执行秘书和世界银行、联合国难民事务高级专员办事处和国际农业发展基金的代表发了言。",
                     "将这段文本翻译成英文，難道他們沒有在大地上旅行而觀察前人的結局是怎樣的嗎？前人比他們努力更大，前人於地方的墾植和建設，勝過他們。他們族中的使者曾給他們帶來許多明証，故真主未虧枉他們，但他們虧枉了自己，",
                     "将这段文本翻译成英文，众人啊！使者确已昭示你们从你们的主降示的真理，故你们当确信他，这对于你们是有益的。如果你们不信道，（那末，真主是无需求你们的）， 因为天地万物，确是真主的。真主是全知的，是至睿的。"]
    batch_demo(source_prompt,target_prompt,model_name)