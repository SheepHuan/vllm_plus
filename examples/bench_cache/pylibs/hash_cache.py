import re
import torch

class HashCache:
    def __init__(self, disable=False):
        self.cache = {}
        self.disable = disable
        
    def hash_seq(self,seq):
        return hash(seq)

        
    def split_prompt_by_semantic(self,text):
        segments = re.split(r'(?<=[。！？])\s*|(?<=[.?!])\s*', text)
        chunks = [segment.strip() for segment in segments if segment.strip()]
        return chunks

    def split_prompt(self,prompt,tokenizer):
        special_tokens = ['<|im_start|>system\n', '<|im_end|>\n', '<|im_start|>user\n', '<|im_start|>assistant\n']
        chunks = self.split([prompt],special_tokens[0])
        for i in range(1,len(special_tokens)):
            chunks = self.split(chunks,special_tokens[i])
        semantic_chunks = []
        for chunk in chunks:
            if chunk not in special_tokens:
                chunk = self.split_prompt_by_semantic(chunk)
                semantic_chunks.extend(chunk)
            else:
                semantic_chunks.append(chunk)
        tokens_list = []
        for idx, chunk in enumerate(semantic_chunks):
            tokens = tokenizer.encode(chunk, add_special_tokens=False)
            tokens_list.append(tokens)
        return semantic_chunks,tokens_list
        
    def split(self,chunks,special_token):
        new_chunks = []
        for chunk in chunks:
            split_pattern = re.compile(f'({re.escape(special_token)})')
            parts = split_pattern.split(chunk)
            new_chunks.extend(part for part in parts if part)  # 过滤掉空字符串
        return new_chunks
    
    def _get_cached_kv(self,_hash, past_key_value=None):
        if past_key_value == None:
            past_key_value = self.cache[_hash]
        else:
            curr_past_key_value = self.cache[_hash]
            num_layers = len(curr_past_key_value)
            for i in range(num_layers):
                past_key_value[i][0] = torch.cat([past_key_value[i][0],curr_past_key_value[i][0]],dim=2)
                past_key_value[i][1] = torch.cat([past_key_value[i][1],curr_past_key_value[i][1]],dim=2)
        return past_key_value
    
    def insert(self,chunks,tokens_list,past_key_values):
        if self.disable:
            return
        num_layers = len(past_key_values)
        
        last_len = 0
        for chunk,tokens in zip(chunks,tokens_list):
            hash_value = self.hash_seq(chunk)
            new_past_key_value = []
            # logger.info(f"cache chunk: {chunk}")
            for i in range(num_layers):
                key = past_key_values[i][0][:,:,last_len:last_len+len(tokens),:]
                value = past_key_values[i][1][:,:,last_len:last_len+len(tokens),:]
                new_past_key_value.append([key,value])
            self.cache[hash_value] = new_past_key_value
            last_len = last_len+len(tokens)
    
    def match(self,chunks):
        """
        return past_key_values,query
        """
        cached_chunks = []
        query = ''.join(chunks)
        if self.disable:
            return None,query
        past_key_values = None
        for idx,chunk in enumerate(chunks[:-2]):
            hash_value = self.hash_seq(chunk)
            # if idx == len(chunks)-1 and chunk == ""
            if hash_value in self.cache:
                print(f"cache hit: {chunk}")
                past_key_values = self._get_cached_kv(hash_value,past_key_values)
                cached_chunks.append(chunk)
            else:
                break
        if past_key_values is not None:
            for i in range(len(past_key_values)):
                past_key_values[i] = tuple(past_key_values[i])
            past_key_values = tuple(past_key_values)
        return past_key_values,query
    
    
    