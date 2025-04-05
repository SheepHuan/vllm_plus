from .edit import KVEditor
from vllm import LLM
from vllm.entrypoints.llm import SamplingParams,RequestOutput
from typing import List
import torch
from transformers import AutoTokenizer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["MKL_THREADING_LAYER"] = "GNU"  # 强制使用 GNU 线程层
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"  # 可选：强制使用 Intel 线程
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"


class KVShareNewPipeline:
    def __init__(self, model_name:str, device:str="cuda:0",max_model_len=32768):
        self.model_name = model_name
        self.device = device
        self.model = LLM(model=model_name,
                        device=device,
                        dtype="bfloat16",
                        max_model_len=max_model_len,
                        gpu_memory_utilization=0.9,
                        multi_step_stream_outputs=True,
                        enforce_eager=True,
                        disable_async_output_proc=True,
                        trust_remote_code=True
                    )
        tokenizer = AutoTokenizer.from_pretrained(model_name,local_files_only=True)
    
    @staticmethod
    def get_kvcache_by_full_compute(model:LLM,sampling_params:SamplingParams, prompt:List[str],device:str="cuda:0"):
        model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["check"] = False
        model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata['collect'] = True
        model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["use_additional_indices"] = False
        model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["additional_map_indices"] = None
        model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["old_kv_map_indices"] = None
        model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["imp_indices"] = None
        num_layer = len(model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers)
        for j in range(num_layer):
            model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[j].self_attn.hack_kv = []
        
        output:List[RequestOutput] = model.generate(prompt, sampling_params,use_tqdm=False)
        
        # 清空缓存
        model.llm_engine.model_executor.driver_worker.model_runner._hack_kv_tables = dict()
        torch.cuda.empty_cache()
        
        
        
        llm_layers = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers
    
        

        hack_kv_tables = model.llm_engine.model_executor.driver_worker.model_runner._hack_kv_tables
        keys = sorted(hack_kv_tables.keys())
        batch_kvcache = []
        for key in keys:
            past_key_values = []
            hack_kv = hack_kv_tables[key]
            for j in range(num_layer):
                temp_key_cache = hack_kv[j][0].clone().to(device)
                temp_value_cache = hack_kv[j][1].clone().to(device)
                past_key_values.append(torch.stack([temp_key_cache,temp_value_cache],dim=0))    
            past_key_values = torch.stack(past_key_values,dim=0)
            batch_kvcache.append(past_key_values)
        batch_kvcache = torch.concat(batch_kvcache,dim=2)
        return batch_kvcache,output,keys
        
    @staticmethod
    def find_texts_differences(source_token_ids:List[int],target_token_ids:List[int]):
        return KVEditor.find_text_differences(source_token_ids,target_token_ids)

    @staticmethod
    def apply_changes2kvcache(target_token_ids:List[int],source_kvcache:torch.Tensor,diff_report):
        return KVEditor.apply_change(target_token_ids,source_kvcache,diff_report)

    @staticmethod
    def full_compute(llm_model,sampling_params:SamplingParams,prompt:str) -> List[RequestOutput]:
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["check"] = False
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata['collect'] = False
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["use_additional_indices"] = False
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["additional_map_indices"] = None
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["old_kv_map_indices"] = None
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["imp_indices"] = None

      
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["check"] = False
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata['collect'] = False
        output = llm_model.generate(prompt,sampling_params,use_tqdm=False)
        ttft_time = output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time
        return output
    
    
    @staticmethod
    def batch_full_compute(llm_model,sampling_params:SamplingParams,prompt:List[str]) -> List[RequestOutput]:
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["check"] = False
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata['collect'] = False
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["use_additional_indices"] = False
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["additional_map_indices"] = None
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["old_kv_map_indices"] = None
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["imp_indices"] = None
        # sampling_params = SamplingParams(temperature=0, max_tokens=1)
      
      
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["check"] = False
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata['collect'] = False
        outputs = llm_model.generate(prompt,sampling_params,use_tqdm=False)
        return outputs
    
    
    @staticmethod
    def partial_compute(llm_model:LLM,sampling_params:SamplingParams, 
                        batch_target_prompt,
                        target_kvcache,
                        reused_map_indices,
                        unreused_map_indices,
                        sample_selected_token_indices,
                        batch_slice_list,
                        guess_target_req_ids,
                        device="cuda:0") -> List[RequestOutput]:
        # additional_map_indices = torch.tensor(unused_map_indices).to(device).to(torch.int64)
        # old_kv_map_indices = torch.tensor(reused_map_indices).to(device).to(torch.int64)

        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["check"] = True
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata['collect'] = False
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["recomp_ratio"] = 0.0
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["use_additional_indices"] = True
        # llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["additional_map_indices"] = additional_map_indices
        # llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["old_kv_map_indices"] = old_kv_map_indices
        # llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["selected_token_indices"] = selected_token_indices
        # llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.old_kvs = reused_kvcache
        llm_model.llm_engine.model_executor.driver_worker.model_runner._kvshare_preill_metadata.batch_reused_map_indices = reused_map_indices
        llm_model.llm_engine.model_executor.driver_worker.model_runner._kvshare_preill_metadata.batch_unreused_map_indices = unreused_map_indices
        llm_model.llm_engine.model_executor.driver_worker.model_runner._kvshare_preill_metadata.batch_sample_selected_token_indices = sample_selected_token_indices
        llm_model.llm_engine.model_executor.driver_worker.model_runner._kvshare_preill_metadata.batch_target_slice_list = batch_slice_list
        llm_model.llm_engine.model_executor.driver_worker.model_runner._kvshare_preill_metadata.batch_kvcache = target_kvcache
        llm_model.llm_engine.model_executor.driver_worker.model_runner._kvshare_preill_metadata.batch_current_request_ids = guess_target_req_ids
        outputs = llm_model.generate(batch_target_prompt,sampling_params,use_tqdm=False)
        
        
        llm_model.llm_engine.model_executor.driver_worker.model_runner._kvshare_preill_metadata.batch_reused_map_indices = [] 
        llm_model.llm_engine.model_executor.driver_worker.model_runner._kvshare_preill_metadata.batch_unreused_map_indices = []
        llm_model.llm_engine.model_executor.driver_worker.model_runner._kvshare_preill_metadata.batch_sample_selected_token_indices =[]
        llm_model.llm_engine.model_executor.driver_worker.model_runner._kvshare_preill_metadata.batch_prompt_lengths = []
        llm_model.llm_engine.model_executor.driver_worker.model_runner._kvshare_preill_metadata.batch_kvcache = []
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.old_kvs = None
        torch.cuda.empty_cache()
        return outputs
    
    
