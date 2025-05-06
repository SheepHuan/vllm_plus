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
    def __init__(self, model_name:str, device:str="cuda:0",max_model_len=8192):
        self.model_name = model_name
        self.device = device
        self.model = LLM(model=model_name,
                        device=device,
                        dtype=torch.float16,
                        max_model_len=max_model_len,
                        gpu_memory_utilization=0.8,
                        multi_step_stream_outputs=True,
                        enforce_eager=True,
                        disable_async_output_proc=True,
                        trust_remote_code=True,
                        enable_chunked_prefill=False
                    )
        # tokenizer = AutoTokenizer.from_pretrained(model_name,local_files_only=True)
    
    @staticmethod
    def get_kvcache_by_full_compute(model:LLM,sampling_params:SamplingParams, prompt:List[str],device:str="cuda:0"):
        model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["check"] = False
        model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata['collect'] = True
        model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["use_additional_indices"] = False
        model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["additional_map_indices"] = None
        model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["old_kv_map_indices"] = None
        model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["imp_indices"] = None
        model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["enable_kvshare"] = False
        model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["enable_cacheblend"] = False
        model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["enable_only_compute_unreused"] = False
        model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["has_additional_value_error"] = False
        model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["las_additional_value_error"] = False
        model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["enable_compute_as"] = False  
        model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["prefill_atten_bias"] = None
        model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["selected_token_indices"] = None
        model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["batch_seq_start_loc"] = None
        model.llm_engine.model_executor.driver_worker.model_runner._kvshare_metadata.is_partial_compute = False
        num_layer = len(model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers)
        for j in range(num_layer):
            model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[j].self_attn.hack_kv = []
            model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[j].self_attn.hack_forward_attn = []
            model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[j].self_attn.hack_cross_attn = None
            model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[j].self_attn.hack_attn = None
        
        # 清空缓存
        model.llm_engine.model_executor.driver_worker.model_runner._hack_kv_tables = dict()
        model.llm_engine.model_executor.driver_worker.model_runner._hack_forward_attn_table = dict()
        model.llm_engine.model_executor.driver_worker.model_runner._hack_cross_attn_table = dict()
        model.llm_engine.model_executor.driver_worker.model_runner._hack_attn_table = dict()
        torch.cuda.empty_cache()
        
        output:List[RequestOutput] = model.generate(prompt, sampling_params,use_tqdm=False)
        
        llm_layers = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers
        hack_kv_tables = model.llm_engine.model_executor.driver_worker.model_runner._hack_kv_tables

        keys = sorted(hack_kv_tables.keys())
        batch_kvcache = []
        for key in keys:
            past_key_values = []
            hack_kv = hack_kv_tables[key]
            for j in range(num_layer):
                temp_key_cache = hack_kv[j][0]
                temp_value_cache = hack_kv[j][1]
                past_key_values.append(torch.stack([temp_key_cache,temp_value_cache],dim=0))    
            past_key_values = torch.stack(past_key_values,dim=0)
            batch_kvcache.append(past_key_values)
        # batch_kvcache = torch.concat(batch_kvcache,dim=2)    
        return batch_kvcache,output,keys
        
    @staticmethod
    def find_texts_differences(source_token_ids:List[int],target_token_ids:List[int]):
        return KVEditor.find_text_differences(source_token_ids,target_token_ids)

    @staticmethod
    def apply_changes2kvcache(target_token_ids:List[int],source_kvcache:torch.Tensor,diff_report):
        return KVEditor.apply_change(target_token_ids,source_kvcache,diff_report)
    
    
    @staticmethod
    def full_compute(llm_model,sampling_params:SamplingParams,prompt:List[str]) -> List[RequestOutput]:
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["prefill_atten_bias"] = None
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["check"] = False
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata['collect'] = False
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["use_additional_indices"] = False
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["additional_map_indices"] = None
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["old_kv_map_indices"] = None
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["imp_indices"] = None
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["enable_kvshare"] = False
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["enable_cacheblend"] = False
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["enable_only_compute_unreused"] = False
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["has_additional_value_error"] = False
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["las_additional_value_error"] = False
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["enable_compute_as"] = False    
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["selected_token_indices"] = None
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["batch_seq_start_loc"] = None 
        llm_model.llm_engine.model_executor.driver_worker.model_runner._kvshare_metadata.is_partial_compute = False
        # sampling_params = SamplingParams(temperature=0, max_tokens=1)
      
      
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["check"] = False
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata['collect'] = False
        outputs = llm_model.generate(prompt,sampling_params,use_tqdm=False)
        return outputs
    
    
    @staticmethod
    def partial_compute(llm_model:LLM,
                        sampling_params:SamplingParams, 
                        batch_target_prompt,
                        target_kvcache,
                        reused_map_indices,
                        unreused_map_indices,
                        next_batch_request_ids,
                        enable_kvshare=False,
                        enable_cacheblend=False,
                        enable_only_compute_unreused=False,
                        enable_compute_as=False,
                        enable_epic=False,
                        has_additional_value_error=False,
                        las_additional_value_error=False,
                        enable_kvshare_decode=False,
                        cacheblend_recomp_ratio=0.15,
                        has_top_ratio=0.15,
                        device="cuda:0") -> List[RequestOutput]:
        from vllm.worker.model_runner import SingleRequestKVShareMetadata
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["check"] = True
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata['collect'] = False
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["recomp_ratio"] = cacheblend_recomp_ratio
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["use_additional_indices"] = True
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["enable_kvshare"] = enable_kvshare
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["enable_cacheblend"] = enable_cacheblend
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["enable_only_compute_unreused"] = enable_only_compute_unreused
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["has_additional_value_error"] = has_additional_value_error
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["las_additional_value_error"] = las_additional_value_error
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["enable_compute_as"] = enable_compute_as
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["enable_epic"] = enable_epic
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["enable_kvshare_decode"] = enable_kvshare_decode
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["prefill_atten_bias"] = None
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["selected_token_indices"] = None
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["batch_seq_start_loc"] = None
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["las_top_ratio"] = 1-has_top_ratio
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["has_top_ratio"] = has_top_ratio
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["check_layers"] = [1]
        llm_model.llm_engine.model_executor.driver_worker.model_runner._kvshare_metadata.is_partial_compute = True
        
        for idx,request_id in enumerate(next_batch_request_ids):
            metadata = SingleRequestKVShareMetadata()
            metadata.reused_map_indices = torch.tensor(reused_map_indices[idx],device=device,dtype=torch.long)
            metadata.unreused_map_indices = torch.tensor(unreused_map_indices[idx],device=device,dtype=torch.long)
            metadata.kvcache = target_kvcache[idx]
            metadata.sample_selected_token_indices = torch.tensor([len(unreused_map_indices[idx])-1],device=device,dtype=torch.long)
            llm_model.llm_engine.model_executor.driver_worker.model_runner._kvshare_metadata.batch_kvshare_metadata[request_id] = metadata
            
        
        
        num_layer = len(llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers)
        for j in range(num_layer):
            llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[j].self_attn.hack_kv = []
        
        # 清空缓存
        llm_model.llm_engine.model_executor.driver_worker.model_runner._hack_kv_tables = dict()
        outputs = llm_model.generate(batch_target_prompt,sampling_params,use_tqdm=False)
        
        for idx,request_id in enumerate(next_batch_request_ids):
            llm_model.llm_engine.model_executor.driver_worker.model_runner._kvshare_metadata.batch_kvshare_metadata[request_id].kvcache.to("cpu")
        llm_model.llm_engine.model_executor.driver_worker.model_runner._kvshare_metadata.batch_kvshare_metadata = dict()
        llm_model.llm_engine.model_executor.driver_worker.model_runner._kvshare_metadata.batch_seq_start_loc = None
        
        
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.old_kvs = [[None,None]] * len(llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers)  
        llm_model.llm_engine.model_executor.driver_worker.model_runner._kvshare_metadata.is_partial_compute = False
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.old_kvs = None
        
        torch.cuda.empty_cache()
        
        # updated_indice = llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["imp_indices"]
        return outputs
    
    
