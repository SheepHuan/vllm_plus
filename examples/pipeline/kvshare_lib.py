from vllm import LLM, SamplingParams
import torch
import json
from transformers import AutoTokenizer
from typing import List
from langchain_text_splitters.spacy import SpacyTextSplitter
from sentence_transformers import SentenceTransformer
from log import setup_logger
import os
import hashlib
from pymilvus import MilvusClient
import numpy as np
import time
import copy
import random
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"

def timestamp_long():
    a = str(int(time.time() * 1000))
    a = a + str(random.randint(0,9))
    a = int(a)
    return a

class KVSharePlugin:
    def __init__(self, 
                 llm_model_name, 
                 sentence_model_name,
                 gpu_memory_utilization=0.6,
                 max_model_len=8192,
                 multi_step_stream_outputs=True,
                 disable_async_output_proc=True,
                 chunk_separator="\n\n--\n\n--\n\n",
                 device="cuda:0",
                 milvus_connection_name="kvshare",
                 milvus_database_path="examples/pipeline/data/milvus_demo.db",
                 milvus_dimension=1536,
                 ):
        self.llm = LLM(model=llm_model_name, gpu_memory_utilization=gpu_memory_utilization,max_model_len=max_model_len,
          multi_step_stream_outputs=multi_step_stream_outputs,enforce_eager=True,
          disable_async_output_proc=disable_async_output_proc)
        
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        
        self.sentence_model = SentenceTransformer(sentence_model_name,local_files_only=True).to(device).to(torch.bfloat16)
        
        self.connection_name = milvus_connection_name
        self.client = MilvusClient(milvus_database_path)
        self.collection = self.client.create_collection(
                collection_name=self.connection_name,
                dimension=milvus_dimension,  # The vectors we will use in this demo has 768 dimensions
            )
        
        self.is_need_chunk = True
        self.is_kvcache_hit = False
        self.use_semantic_search = True
        self.save_kvcache = True
        self.enable_show_log = True
        
        self.metrics = {}
        
        self.chunk_separator = chunk_separator
        self.text_splitter = SpacyTextSplitter(
            separator=self.chunk_separator,
        )
        
        self.logger = setup_logger("kvshare")
      
    def edit_kvcache(self,old_prompt,new_prompt,old_kvcache):
        old_token_ids = self.tokenizer.encode(old_prompt)
        new_token_ids = self.tokenizer.encode(new_prompt)
        from edit import edit_distance_with_operations
        _,ops = edit_distance_with_operations(old_token_ids,new_token_ids,self.tokenizer)
        new_kvcache: np.ndarray = copy.deepcopy(np.array(old_kvcache))
        num_layer = new_kvcache.shape[0]
        head_dim = new_kvcache.shape[-1]
        # 先转置，将kvcache的维度从[layer,2,seq_len,head_dim]转置为[seq_len,2,layer,head]
        new_kvcache = new_kvcache.transpose((2,1,0,3))
        additional_indices = []
        for op in ops:
            if op[0]=="Delete":
                index = int(op[-1])
                new_kvcache = np.delete(new_kvcache, index, axis=0)
                
            elif op[0]=="Insert":
                char = op[1]
                index = int(op[-1])
                new_kv = np.zeros([2,num_layer,head_dim])
                new_kv[0,0] = char
                new_kvcache = np.insert(new_kvcache, index, new_kv, axis=0)
                additional_indices.append(index)
            elif op[0]=="Replace":
                char = op[2]
                index = int(op[-1])
                new_kv = np.zeros([2,num_layer,head_dim])
                new_kv[0,0] = char
                new_kvcache[index] = new_kv
                additional_indices.append(index)
        new_kvcache = new_kvcache.transpose((2,1,0,3))
        key_value_cache = [[torch.from_numpy(np.array(new_kvcache[layer_idx][cid])).to(self.device).to(torch.bfloat16) for cid in range(2)]  for layer_idx in range(num_layer)]
        return key_value_cache,additional_indices
    
    def find_kvcache(self,chunks,chunk_indices):
        
        metirc = {
            "num_token_hit":0,
            "num_token_miss":0,
        }
        
        num_layer = len(self.llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers)
        past_key_values = []
        chunk_cache_hit_indices = []
        chunk_cache_miss_indices = []
        partial_token_miss_indices = []
        partial_token_hit_indices = []
        
        # qwen2.5-7B
        attention_dim = 512
        
        is_kvcache_hit = False
        for i in range(len(chunks)):
            embeddings = self.sentence_model.encode(chunks[i])  # 批量计算embedding
            res = self.client.search(
                collection_name=self.connection_name,  # target collection
                data=[embeddings],  # query vectors
                limit=1,  # number of returned entities
                output_fields=["text", "key_value_cache","embeddings","hash_key"],  # specifies fields to be returned
            )
            if len(res[0]) != 0:
                if res[0][0]["distance"] >= 0.8:
                    
                    hash_srouce = hashlib.md5(chunks[i].encode()).hexdigest()
                    hash_target = hashlib.md5( res[0][0]["entity"]["text"].encode()).hexdigest()
                    if hash_srouce == hash_target:
                        key_value_cache = res[0][0]["entity"]["key_value_cache"]
                        key_value_cache = [[torch.from_numpy(np.array(key_value_cache[layer_idx][cid])).to(self.device).to(torch.bfloat16) for cid in range(2)]  for layer_idx in range(num_layer)]
                        chunk_cache_hit_indices.append(chunk_indices[i])
                        if self.enable_show_log:
                            self.logger.info(f"kvcache hit: {chunks[i]}")
                        metirc["num_token_hit"] += len(self.tokenizer.encode(chunks[i]))
                    else:
                        
                        key_value_cache,additional_indices = self.edit_kvcache(old_prompt=res[0][0]["entity"]["text"],new_prompt=chunks[i],old_kvcache=res[0][0]["entity"]["key_value_cache"])
                        
                        new_tokens = self.tokenizer.encode(chunks[i])
                        token_missed = [new_tokens[index] for index in additional_indices]
                        token_hit = [new_tokens[index] for index in range(len(new_tokens)) if index not in additional_indices]
                        if self.enable_show_log:
                            self.logger.info(f"kvcache miss: {self.tokenizer.decode(token_missed)}, kvcache hit: {self.tokenizer.decode(token_hit)}")
                        additional_indices = [index + chunk_indices[i][0] for index in additional_indices]
                        
                        partial_token_miss_indices.extend(additional_indices)
                        partial_token_hit_indices.extend(list(set(list(range(chunk_indices[i][0],chunk_indices[i][1])))-set(additional_indices)))
                        metirc["num_token_miss"] += len(token_missed)
                    is_kvcache_hit = True
                    
                else:
                    # cache miss的key value cache就是0，attention中会重新计算
                    key_value_cache = [[torch.zeros(chunk_indices[i][1]-chunk_indices[i][0],attention_dim).to(self.device).to(torch.bfloat16) for _ in range(2)] for _ in range(num_layer)]
                    chunk_cache_miss_indices.append(chunk_indices[i])
                    if self.enable_show_log:
                        self.logger.info(f"kvcache miss: {chunks[i]}, chunk_indices: {chunk_indices[i]}")
                    metirc["num_token_miss"] += len(self.tokenizer.encode(chunks[i]))
                    
                for j in range(num_layer):
                    if i == 0:
                        past_key_values.append([key_value_cache[j][0],key_value_cache[j][1]])
                    else:
                        past_key_values[j][0] = torch.cat((past_key_values[j][0],key_value_cache[j][0]), dim=0)
                        past_key_values[j][1] = torch.cat((past_key_values[j][1],key_value_cache[j][1]), dim=0)
        
        additional_map_indices = []
        old_kv_map_indices = []
        for indice in chunk_cache_hit_indices:
            old_kv_map_indices.extend(list(range(indice[0],indice[1])))
        for indice in chunk_cache_miss_indices:
            additional_map_indices.extend(list(range(indice[0],indice[1])))
        
        old_kv_map_indices.extend(partial_token_hit_indices)
        additional_map_indices.extend(partial_token_miss_indices)
        
            
        self._set_confuse_metadata_raw("use_additional_indices",True)
        self._set_confuse_metadata_raw("additional_map_indices",torch.tensor(additional_map_indices).to(self.device).to(torch.int64))
        self._set_confuse_metadata_raw("old_kv_map_indices",torch.tensor(old_kv_map_indices).to(self.device).to(torch.int64))
        self.llm.llm_engine.model_executor.driver_worker.model_runner.model.model.old_kvs = past_key_values
        return is_kvcache_hit,metirc


    def generate_with_kvshare(self,prompts,max_tokens=1024):
        """
        prompts分块，然后检查是否存在kvcache，如果存在，则使用kvcache，否则使用full prefill，存储下kv cache
        
        """
        system_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        user_prompt_prefix = "<|im_start|>user\n"
        user_prompt_suffix = "<|im_end|>\n<|im_start|>assistant\n"
        metric = None
        
        if self.is_need_chunk:
            # 检索kvcache
            chunks = self.text_split(prompts)
            chunks = [chunk +'\n' for chunk in chunks]
            chunks = [system_prompt,user_prompt_prefix] + chunks + [user_prompt_suffix]
        else:
            chunks = [system_prompt,user_prompt_prefix] + [prompts+'\n'] + [user_prompt_suffix]
        
        chunks_ids = []
        chunk_indices = []
        for chunk in chunks:
            chunk_ids = self.tokenizer.encode(chunk)
            chunks_ids.extend(chunk_ids)
            if len(chunk_indices) !=0:
                chunk_indices.append([chunk_indices[-1][1],chunk_indices[-1][1]+len(chunk_ids)])
            else:
                chunk_indices.append([0,len(chunk_ids)])
            
        query_prompt = self.tokenizer.decode(chunks_ids)
        query_ids_len = len(self.tokenizer.encode(query_prompt))
        
        if self.enable_show_log:
            self.logger.info(f"query_ids_len: {query_ids_len}, chunk_ids_len: {len(chunk_ids)}")
        

        if self.use_semantic_search:
            self.is_kvcache_hit,metirc = self.find_kvcache(chunks=chunks,chunk_indices=chunk_indices)
        else:
            self.is_kvcache_hit = False
              
        if self.is_kvcache_hit:
            self._set_confuse_metadata_raw("recomp_ratio",0.0)
            self._set_confuse_metadata(check=True,collect=True)
        else:
            self._set_confuse_metadata(check=False,collect=True)
            
            
        #
        sampling_params = SamplingParams(temperature=0, max_tokens=max_tokens)
        
        output = self.llm.generate(query_prompt, sampling_params)
        # self.logger.info(f"Cached generation: {output[0].outputs[0].text}")
        if self.enable_show_log:
            self.logger.info(f"Cached generation: {output[0].outputs[0].text}")
            self.logger.info(f"TTFT with cache: {output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time}")
        
        if self.save_kvcache:
            self.save_current_kvcache(chunks,chunk_indices)

        return output[0].outputs[0].text,metric
        
    
        
    def save_current_kvcache(self,chunk_prompts: List[str],chunk_indices: List[List[int]]):
        
        llm_layers = self.llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers
        num_layer = len(llm_layers)

        chunk_past_key_values = [[] for _ in range(len(chunk_prompts))]
      
        for j in range(num_layer):
            past_key_values = llm_layers[j].self_attn.hack_kv
            temp_key_cache = past_key_values[0].clone()
            temp_value_cache = past_key_values[1].clone()

            for i in range(len(chunk_prompts)):
                # 第i个chunk，保存它对应的key和value
                chunk_past_key_values[i].append([
                    temp_key_cache[chunk_indices[i][0]:chunk_indices[i][1]],
                    temp_value_cache[chunk_indices[i][0]:chunk_indices[i][1]]
                ])
        
        idx_start = timestamp_long() * 1000
        for i in range(len(chunk_prompts)):
            hash_key = hashlib.md5(chunk_prompts[i].encode()).hexdigest()
            # 如果存过该文本，那就跳过
            # 添加embeddings
            embeddings = self.sentence_model.encode(chunk_prompts[i])  # 批量计算embedding
            res = self.client.search(
                    collection_name=self.connection_name,
                    data=[embeddings],
                    limit=1,
                    output_fields=["text", "hash_key"],
                )
            if res!=[]:
                if len(res[0]) != 0 and res[0][0]["distance"] >= 0.999:
                    if self.enable_show_log:
                        self.logger.info(f"skip save kvcache: {chunk_prompts[i]}")
                    continue
            
            key_value_cache=np.array([[chunk_past_key_values[i][layer_idx][cid].to('cpu').to(torch.float16).numpy() for cid in range(2)]  for layer_idx in range(num_layer)])
            data = [
                {"id": idx_start+i, "vector": embeddings, "text": chunk_prompts[i], "hash_key": hash_key, "key_value_cache": key_value_cache}
            ]
            self.client.insert(collection_name=self.connection_name, data=data)
 
            
       
    def _set_confuse_metadata(self,check=False,collect=False):
        # cache_fuse_metadata = 
        self.llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["check"] = check
        self.llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata['collect'] = collect
            
    def _set_confuse_metadata_raw(self,key,value):
        self.llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata[key] = value

    
    def generate(self,prompts,max_tokens=1024):
        self._set_confuse_metadata(check=False,collect=False)
        
        sampling_params = SamplingParams(temperature=0, max_tokens=max_tokens)
            
        output = self.llm.generate(prompts, sampling_params)
        if self.enable_show_log:
            self.logger.info(f"Cached generation: {output[0].outputs[0].text}")
            self.logger.info(f"TTFT with cache: {output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time}")

        return output[0].outputs[0].text
        
    
    
    def text_split(self,text):
        docs = self.text_splitter.split_text(text)
        chunks = docs[0].split(self.chunk_separator)
        return chunks

    def generate_with_cacheblend(self,doc_prompts,user_prompt,max_tokens=1024):
        system_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        user_prompt_prefix = "<|im_start|>user\n"
        user_prompt_suffix = "<|im_end|>\n<|im_start|>assistant\n"
        doc_prompts =  [system_prompt,user_prompt_prefix,user_prompt]  + doc_prompts + [user_prompt_suffix]
        
        doc_chunk_ids = [self.tokenizer.encode(doc) for doc in doc_prompts]
 
        sampling_params = SamplingParams(temperature=0, max_tokens=1)

        # Create an tokenizer and LLM.
        cache_fuse_metadata = self.llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata
        cache_fuse_metadata['collect'] = False
        cache_fuse_metadata['check'] = False

        cache_fuse_metadata['collect'] = True
        cache_fuse_metadata["check"] = False
        num_layer = len(self.llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers)
        chunk_past_key_values = []
    
        for i in range(len(doc_chunk_ids)):
            prompts = [self.tokenizer.decode(doc_chunk_ids[i])]
            output = self.llm.generate(prompts, sampling_params)
            
            llm_layers = self.llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers
            for j in range(num_layer):
                past_key_values = llm_layers[j].self_attn.hack_kv
                temp_k = past_key_values[0].clone()
                temp_v = past_key_values[1].clone()
                if i == 0:
                    chunk_past_key_values.append([temp_k, temp_v])
                else:
                    #pdb.set_trace()
                    chunk_past_key_values[j][0] = torch.cat((chunk_past_key_values[j][0],temp_k), dim=0)
                    chunk_past_key_values[j][1] = torch.cat((chunk_past_key_values[j][1],temp_v), dim=0)
            #print(temp_k.shape[0])
            self.llm.llm_engine.model_executor.driver_worker.model_runner.model.model.old_kvs = chunk_past_key_values
        
        input_ids = []
        for i in range(len(doc_chunk_ids)):
            temp_ids = doc_chunk_ids[i]
            input_ids += temp_ids
        input_prompt = self.tokenizer.decode(input_ids)
        sampling_params = SamplingParams(temperature=0, max_tokens=2048)
        cache_fuse_metadata["check"] = True
        cache_fuse_metadata['collect'] = False
        output = self.llm.generate([input_prompt], sampling_params)
        print(f"Cached generation: {output[0].outputs[0].text}")
        print(f"TTFT with cache: {output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time}")
        metric = {
            "ttft":output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time
        }
        return output[0].outputs[0].text,metric

        
if __name__ == "__main__":
    import os
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    
    text1 = "Translate the following text from English to Chinese: The real talent is resolute aspirations. The miracle appear in bad luck. Man does not become a good husband."
    text2 = "Translate the following texts from English to Chinese and explain them: The real talent is resolute aspirations. Manners make human co-existence of golden key. The miracle appear in bad luck. Man does not become a good husband. Manners make human co-existence of golden key."
    kvshare = KVSharePlugin(llm_model_name="Qwen/Qwen2.5-7B-Instruct",
                            sentence_model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
                            gpu_memory_utilization=0.8,
                            max_model_len=8192,
                            multi_step_stream_outputs=True,
                            disable_async_output_proc=True,
                            chunk_separator="\n\n--\n\n--\n\n")
    
    kvshare.use_semantic_search = True
    kvshare.save_kvcache = False
    
    # kvshare.generate_with_kvcache(text2)
    # kvshare.generate_with_cacheblend(text2,text1)
