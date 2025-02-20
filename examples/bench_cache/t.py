from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
global global_cache
global_cache = {}

def similarity(a,b):
    return torch.nn.functional.cosine_similarity(a,b,dim=1)

def generate_query(model,tokenizer,prompt,enable_embedding_cache=False):
    model_inputs = tokenizer(prompt,return_tensors="pt").to(model.device)
    
    if enable_embedding_cache:
        tokens_embeddings = model.model.embed_tokens(model_inputs.input_ids)
        sentence_embedding = torch.mean(tokens_embeddings,dim=1).unsqueeze(0)
        for key in global_cache.keys():
            if similarity(global_cache[key],sentence_embedding) > 0.9:
                query = key
                kv_cache = global_cache[key]
        # 差异化token扩展，让差异的token重新计算，没有差异的不重新计算
        pass
        
    else:
        query = model_inputs.input_ids
    
    generate_kwargs = {
            'max_new_tokens': 1,
            'pad_token_id': tokenizer.eos_token_id,
            'top_p': 0.95,
            'temperature': 0.1,
            'repetition_penalty': 1.0,
            'top_k': 50,
            "return_dict_in_generate":True,
            "output_attentions":True,
        }
    outputs = model.generate(**model_inputs, **generate_kwargs)
    query = tokenizer.decode(outputs[0],skip_special_tokens=True)
    return query

