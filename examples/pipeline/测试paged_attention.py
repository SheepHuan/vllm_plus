from vllm.attention.ops.paged_attn import PagedAttention
import torch

device = "cuda:0"
query_seq_len = 3
kv_seq_len = 1024
kv_cache_dtype = "bfloat16"
num_kv_heads = 16
scale = 1.0
alibi_slopes = None
k_scale = 1.0
v_scale = 1.0

decode_query = torch.randn(1,3,1024,device=device,dtype=torch.bfloat16)
key_cache = torch.randn(1,1024,1024,device=device,dtype=torch.bfloat16)
value_cache = torch.randn(1,1024,1024,device=device,dtype=torch.bfloat16)
block_tables_arg = torch.randint(0,1024,(1,1024),device=device,dtype=torch.long)
seq_lens_arg = torch.randint(0,1024,(1,),device=device,dtype=torch.long)
max_seq_len_arg = 1024


PagedAttention.forward_decode(
                decode_query,
                key_cache,
                value_cache,
                block_tables_arg,
                seq_lens_arg,
                max_seq_len_arg,
                kv_cache_dtype,
                num_kv_heads,
                scale,
                alibi_slopes,
                k_scale,
                v_scale,
            )