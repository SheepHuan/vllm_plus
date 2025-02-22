from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention,Qwen2Model,Qwen2DecoderLayer,Qwen2ForCausalLM,apply_rotary_pos_emb,eager_attention_forward,ALL_ATTENTION_FUNCTIONS,Qwen2Config,KwargsForCausalLM

import torch
import torch.nn as nn
from typing import Callable, List, Optional, Tuple, Union
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
import transformers.utils.logging as logging
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import (AttentionBias,
                                         BlockDiagonalCausalMask,
                                         BlockDiagonalMask,
                                         LowerTriangularMaskWithTensorBias,LowerTriangularMask,LowerTriangularFromBottomRightMask)

import json
from transformers.generation.utils import GenerationConfig,GenerateNonBeamOutput,GenerateEncoderDecoderOutput,GenerateDecoderOnlyOutput,GenerateEncoderDecoderOutput
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.streamers import BaseStreamer
import os
from transformers import AutoTokenizer

logger = logging.get_logger(__name__)
GLOBAL_KV_CACHE: torch.Tensor = None

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def slim_kv(hidden_states:torch.Tensor,n_rep:int):
    """
    repeatkv的反向操作
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states.reshape(batch, num_key_value_heads // n_rep, n_rep, slen, head_dim)
    # 去除重复的n_rep维度
    hidden_states = hidden_states[:,:,0,:,:]
    return hidden_states.reshape(batch, -1, slen, head_dim)

def sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    cache_metadata = None,
    layer_idx = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:

    
    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

   
    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key.shape[-2]]

    # SDPA with memory-efficient backend is bugged with non-contiguous inputs and custom attn_mask for some torch versions
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    if is_causal is None:
        is_causal = causal_mask is None and query.shape[2] > 1

    # if cache_metadata["cache_blend"]["check"] and cache_metadata["is_prefill"] and (layer_idx > cache_metadata["cache_blend"]["check_layers"][-1] or layer_idx in cache_metadata["cache_blend"]["check_layers"]):
    #     attn_output = query
    #     tmp_query = query[:,:,cache_metadata["cache_blend"]["imp_indices"],:]
    #     tmp_a = torch.nn.functional.scaled_dot_product_attention(
    #         tmp_query,
    #         key,
    #         value,
    #         attn_mask=causal_mask,
    #         dropout_p=dropout,
    #         scale=scaling,
    #         is_causal=is_causal,
    #     )
    #     attn_output[:,:,cache_metadata["cache_blend"]["imp_indices"],:] = tmp_a
    #     attn_output = attn_output.transpose(1, 2).contiguous()
    # else:
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=causal_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    
    # attn_output = attn_output.reshape(attn_output.shape[0],-1,self.num_head*self.head_dim).contiguous()

    return attn_output, None

def run_memory_efficient_xformers_forward(
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        dropout: float = 0.0,
        scaling: Optional[float] = None,
        is_causal: Optional[bool] = None,
        **kwargs,
    ):
    # if hasattr(module, "num_key_value_groups"):
    #     key = repeat_kv(key, module.num_key_value_groups)
    #     value = repeat_kv(value, module.num_key_value_groups)

    n, seq_len,_, dim = query.shape
    num_group = module.num_attention_heads//module.num_key_value_heads
    num_head_per_group = module.num_key_value_heads
    
    def repeat_kv2(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """正确实现键值头重复"""
        batch, seq_len, n_kv_heads, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, :, None, :].expand(
            batch, seq_len, n_kv_heads, n_rep, head_dim
        )
        return hidden_states.reshape(batch, seq_len, n_kv_heads * n_rep, head_dim)
    
     # 确保输入维度正确 [batch, seq_len, heads, dim]
    def reshape_for_gqa(tensor, num_heads):
        return tensor.reshape(
            tensor.shape[0], 
            tensor.shape[1], 
            num_heads, 
            -1
        ).contiguous()

    key = repeat_kv2(key, num_group)
    value = repeat_kv2(value,num_group)
    # query = query.reshape(n, seq_len, -1, dim).contiguous()
    # key = repeat_kv2(key, num_group).contiguous()
    # value = repeat_kv2(value, num_group).contiguous()
    # 调整形状为xFormers期望格式
    query = reshape_for_gqa(query, module.num_attention_heads)
    key = reshape_for_gqa(key, module.num_attention_heads)
    value = reshape_for_gqa(value, module.num_attention_heads)

    # scale_factor = 1.0 / (query.size(-1) ** 0.5) if scaling is None else scaling
    # attn_bias = LowerTriangularFromBottomRightMask()
    # attn_bias = None
    # if attention_mask is not None:
    #     if is_causal:
    #         # 融合因果掩码和输入掩码
    #         attn_bias = xops.LowerTriangularMask() 
    #         attn_bias = attn_bias.masked_fill(~attention_mask, -float("inf"))
    #     else:
    #         attn_bias = xops.fmha.BlockDiagonalMask.from_seqlens(
    #             [attention_mask.size(1)] * attention_mask.size(0)
    #         )
    # elif is_causal:
    #     attn_bias = xops.LowerTriangularMask()

    out = xops.memory_efficient_attention(
        query,
        key,
        value,
        attn_bias=LowerTriangularFromBottomRightMask(),
        p=0.0,
        # scale=None,
        scale=None,  # 因为我们已经在上面应用了缩放
    )

    # 恢复原始形状 [batch, heads, seq_len, dim]
    out = out.reshape(n, seq_len, -1).contiguous()
    
    return out

class CustomQwen2Attention(Qwen2Attention):
    def __init__(self,config:Qwen2Config, layer_idx: int):
        super().__init__(config,layer_idx)
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hack_kv = []

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        cache_metadata: dict = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    
  
        if cache_metadata["cache_blend"]["collect"] and cache_metadata["is_prefill"]:
            self.hack_kv = [key_states.clone(),value_states.clone()]

        cos, sin = position_embeddings
        from transformers.models.qwen2.modeling_qwen2 import rotate_half
        def pos_emb_apply(h,cos,sin, position_ids=None, unsqueeze_dim=1):
            cos = cos.unsqueeze(unsqueeze_dim)
            sin = sin.unsqueeze(unsqueeze_dim)
            h_embed = (h * cos) + (rotate_half(h) * sin)
            return h_embed
       
        if  cache_metadata["cache_blend"]["check"] and cache_metadata["is_prefill"] and self.layer_idx > cache_metadata["cache_blend"]["check_layers"][-1]:
            # 重计算
            cos, sin = cache_metadata["cache_blend"]["imp_position_embeddings"]
            query_states = pos_emb_apply(query_states, cos, sin, position_ids=None, unsqueeze_dim=1)
            key_states = pos_emb_apply(key_states, cos, sin, position_ids=None, unsqueeze_dim=1)
        else:
            # cos, sin = position_embeddings
            query_states = pos_emb_apply(query_states, cos, sin, position_ids=None, unsqueeze_dim=1)
            key_states = pos_emb_apply(key_states, cos, sin, position_ids=None, unsqueeze_dim=1)
        
        if past_key_value is not None:
            # check layer之后每一次prefill都需要先更新一下past_key_value
            if cache_metadata["cache_blend"]["check"]:
                if cache_metadata["is_prefill"]:
                    if cache_metadata["cache_blend"]["fake_q"] is None:
                        cache_metadata["cache_blend"]['fake_q'] = torch.rand_like(query_states)
                    cos,sin = cache_metadata["cache_blend"]['position_embeddings_for_key']
                    _,GLOBAL_KV_CACHE[self.layer_idx][0] = apply_rotary_pos_emb(cache_metadata["cache_blend"]['fake_q'], GLOBAL_KV_CACHE[self.layer_idx][0], cos, sin)
                    
                    # # 计算误差
                    if self.layer_idx in cache_metadata["cache_blend"]["check_layers"]:
                        if cache_metadata["use_additional_indices"]:
                            old_indices = cache_metadata["old_kv_map_indices"]
                            temp_diff = torch.sum((value_states[:,:,old_indices,:]-GLOBAL_KV_CACHE[self.layer_idx][1][:,:,old_indices,:])**2, dim=[0,1,3])
                            topk_num = int(len(temp_diff)*cache_metadata["cache_blend"]["recomp_ratio"])
                            top_indices = torch.topk(temp_diff, k=topk_num).indices
                            top_indices = torch.cat([top_indices,cache_metadata["additional_map_indices"]])
                            if key_states.shape[-2]-1 not in top_indices:
                                top_indices = torch.cat([top_indices,torch.tensor([key_states.shape[-2]-1],device=query_states.device)])
                            top_indices,_ = torch.sort(top_indices)
                            query_states = query_states[:,:,top_indices,:]
                            cache_metadata["cache_blend"]["imp_indices"] = top_indices
                        else:
                            temp_diff = torch.sum((value_states-GLOBAL_KV_CACHE[self.layer_idx][1])**2, dim=[0,1,3])
                            topk_num = int(len(temp_diff)*cache_metadata["cache_blend"]["recomp_ratio"])
                            top_indices = torch.topk(temp_diff, k=topk_num).indices
                            if len(temp_diff)-1 not in top_indices:
                                top_indices = torch.cat([top_indices,torch.tensor([len(temp_diff)-1],device=query_states.device)])
                            top_indices,_ = torch.sort(top_indices)
                            query_states = query_states[:,:,top_indices,:]
                            cache_metadata["cache_blend"]["imp_indices"] = top_indices
                        
                        # 更新误差较大的key和value
                        GLOBAL_KV_CACHE[self.layer_idx][0][:,:,top_indices,:] = key_states[:,:,top_indices,:]
                        GLOBAL_KV_CACHE[self.layer_idx][1][:,:,top_indices,:] = value_states[:,:,top_indices,:]
                    elif self.layer_idx > cache_metadata["cache_blend"]["check_layers"][-1]:
                        GLOBAL_KV_CACHE[self.layer_idx][0][:,:,cache_metadata["cache_blend"]["imp_indices"],:] = key_states
                        GLOBAL_KV_CACHE[self.layer_idx][1][:,:,cache_metadata["cache_blend"]["imp_indices"],:] = value_states
                    else:
                        GLOBAL_KV_CACHE[self.layer_idx][0] = key_states
                        GLOBAL_KV_CACHE[self.layer_idx][1]= value_states
                    

                    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                    key_states, value_states = past_key_value.update(GLOBAL_KV_CACHE[self.layer_idx][0], GLOBAL_KV_CACHE[self.layer_idx][1], self.layer_idx, cache_kwargs)
                    pass
                else:
                    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                    key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
                
            else:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            pass


        sliding_window = None
        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window

        # attn_output, attn_weights = sdpa_attention_forward(
        #     self,
        #     query_states,
        #     key_states,
        #     value_states,
        #     attention_mask,
        #     dropout=0.0 if not self.training else self.attention_dropout,
        #     scaling=self.scaling,
        #     sliding_window=sliding_window,  # main diff with Llama
        #     cache_metadata = cache_metadata,
        #     layer_idx = self.layer_idx,
        #     **kwargs,
        # )
        # attn_output = attn_output.reshape(attn_output.shape[0],-1,self.num_head*self.head_dim).contiguous()
     
        
        attn_output = run_memory_efficient_xformers_forward(
            self,
            query_states.transpose(1,2),
            key_states.transpose(1,2),
            value_states.transpose(1,2),
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=0.08838834764831845,
            is_causal=None,
        )
    
        # # reshape to [batch,seq_len,dim]
        
        attn_output = self.o_proj(attn_output)
        return attn_output, None
    
    
class CustomQwen2DecoderLayer(Qwen2DecoderLayer):
    def __init__(self, config:Qwen2Config, layer_idx: int):
        super().__init__(config,layer_idx)
        self.layer_idx = layer_idx
        self.self_attn = CustomQwen2Attention(config=config, layer_idx=layer_idx)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        cache_metadata: dict = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            cache_metadata=cache_metadata,
            **kwargs,
        )
        
        
        if cache_metadata["cache_blend"]["check"] and cache_metadata["is_prefill"] and self.layer_idx in cache_metadata["cache_blend"]["check_layers"]:
            if cache_metadata["cache_blend"]["imp_indices"] is not None:
                # 确保residual的维度与hidden_states匹配
                residual = residual[:, cache_metadata["cache_blend"]["imp_indices"], :]
        
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs
    
    

class CustomQwen2Model(Qwen2Model):
    def __init__(self, config:Qwen2Config):
        super().__init__(config)
        self.layers = nn.ModuleList([CustomQwen2DecoderLayer(config=config, layer_idx=i) for i in range(config.num_hidden_layers)])
    
        
        self.cache_fuse_metadata ={
            "is_prefill": True,
            "use_additional_indices": False,
            "additional_indices": None,
            "cache_blend": {
                "check_layers":[1],
                "check": False,
                "recomp_ratios":[0.1],
                "recomp_ratio": 0.1,
                "original_slot_mapping":None,
                "our_slot_mapping":None,
                "kv_cache_dtype": None,
                "attn_bias": None,
                "imp_indices": None,
                "org_seq_len": None,
                "collect": False
            }
        }
        self.old_kvs = [[None,None]] * len(self.layers)
        
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        
        # Huan: input_ids: [batch,seq_len,dim]
        is_prefill = self.cache_fuse_metadata["is_prefill"] 
        # self.cache_fuse_metadata["is_prefill"] = is_prefill
        if is_prefill:
            if self.cache_fuse_metadata["cache_blend"]["check"]:
                self.cache_fuse_metadata["cache_blend"]["org_seq_len"] = input_ids.shape[0] 
                self.cache_fuse_metadata["cache_blend"]["fake_q"] = None  
                self.cache_fuse_metadata["cache_blend"]["attn_bias"] = None
                self.cache_fuse_metadata["cache_blend"]["imp_indices"] = None
                self.cache_fuse_metadata["cache_blend"]["original_slot_mapping"] = None
                self.cache_fuse_metadata["cache_blend"]["our_slot_mapping"] = None
                self.cache_fuse_metadata["cache_blend"]['position_embeddings_for_key'] = position_embeddings
        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    cache_metadata=self.cache_fuse_metadata,
                    **flash_attn_kwargs,
                )
            if self.cache_fuse_metadata["cache_blend"]["check"] and self.cache_fuse_metadata["is_prefill"] and layer_idx in self.cache_fuse_metadata["cache_blend"]["check_layers"]:
                # 确保imp_indices存在且有效
                if self.cache_fuse_metadata["cache_blend"]["imp_indices"] is not None:
                    # 计算position_ids对应的切片
                    imp_position_ids = position_ids[:, self.cache_fuse_metadata["cache_blend"]["imp_indices"]]
                    # 确保hidden_states的维度正确
                    imp_hidden_states = hidden_states[:, self.cache_fuse_metadata["cache_blend"]["imp_indices"], :]
                    
                    # 计算position embeddings
                    self.cache_fuse_metadata["cache_blend"]["imp_position_embeddings"] = self.rotary_emb(
                        imp_hidden_states, 
                        imp_position_ids
                    )

            hidden_states = layer_outputs[0]
            # if i==len(self.layers)-1:
            #     pass
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

    
        
        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()


   
    
class CustomQwen2ForCausalLM(Qwen2ForCausalLM):
    
    def __init__(self,config):
        super().__init__(config)
        self.model = CustomQwen2Model(config)


    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
    
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )
       
        hidden_states = outputs[0]
        if self.model.cache_fuse_metadata["cache_blend"]["check"]:
            # slice_indices = hidden_states.shape[1] - 1
            self.model.cache_fuse_metadata["cache_blend"]["check"] = False
        
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        """
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        model_forward = self.__call__
        if isinstance(model_kwargs.get("past_key_values"), Cache):
            is_compileable = model_kwargs["past_key_values"].is_compileable and self._supports_static_cache
            is_compileable = is_compileable and not self.generation_config.disable_compile
            if is_compileable and (
                self.device.type == "cuda" or generation_config.compile_config._compile_all_devices
            ):
                os.environ["TOKENIZERS_PARALLELISM"] = "0"
                model_forward = self.get_compiled_call(generation_config.compile_config)

        is_prefill = True
        while self._has_unfinished_sequences(
            this_peer_finished, synced_gpus, device=input_ids.device, cur_len=cur_len, max_length=max_length
        ):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            if is_prefill:
                self.model.cache_fuse_metadata["is_prefill"] = True
                outputs = self(**model_inputs, return_dict=True)
                is_prefill = False
                self.model.cache_fuse_metadata["is_prefill"] = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True)
            # if outputs.past_key_values is not None:
            #     print(outputs.past_key_values.key_cache[0].shape)
            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue
          
            
            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].clone().float()
            next_token_logits = next_token_logits.to(input_ids.device)

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids
        
        
        
def task_qa(model,tokenizer,doc_prompts,system_prompt,question_prompt):
    generarte_args = {
        "max_new_tokens": 20,
        "do_sample": True,
        "top_p": 0.9,
        "top_k": 0,
        "temperature": 0.9,
        "repetition_penalty": 1.0,
        "length_penalty": 1.0,
        "return_dict_in_generate":True,
    }
    
    doc_prompts = ["<|im_start|>\n"+doc_prompt+"\n<|im_end|>\n" for doc_prompt in doc_prompts]
    
    all_prompts = system_prompt + doc_prompts + question_prompt

    num_layer = len(model.model.layers)
    chunk_past_key_values = []
    all_prompts_ids = []
    
    model.model.cache_fuse_metadata["cache_blend"]["check"] = False
    model.model.cache_fuse_metadata['cache_blend']['collect'] = False
    
    # inputs = tokenizer(' '.join(all_prompts), return_tensors="pt").to(model.device)
    # outputs = model.generate(**inputs,**generarte_args)
    # print(tokenizer.decode(outputs.sequences[0],skip_special_tokens=False))
    
    # return 
    global GLOBAL_KV_CACHE

    for i in range(len(all_prompts)):
        inputs = tokenizer(all_prompts[i], return_tensors="pt").to(model.model.device)
        all_prompts_ids.extend(inputs['input_ids'][0].tolist())
        model.model.cache_fuse_metadata["cache_blend"]["collect"] = True
        model.model.cache_fuse_metadata["cache_blend"]["check"] = False 

        outputs = model.generate(**inputs,**generarte_args)
        print(tokenizer.decode(outputs.sequences[0],skip_special_tokens=True))
        llm_layers = model.model.layers
        for j in range(num_layer):
            past_key_values = llm_layers[j].self_attn.hack_kv
            
            temp_k = past_key_values[0].clone()
            temp_v = past_key_values[1].clone()    

            if i == 0:
                chunk_past_key_values.append([temp_k, temp_v])
            else:
                chunk_past_key_values[j][0] = torch.cat((chunk_past_key_values[j][0],temp_k), dim=2)
                chunk_past_key_values[j][1] = torch.cat((chunk_past_key_values[j][1],temp_v), dim=2)
    # global GLOBAL_KV_CACHE
    GLOBAL_KV_CACHE = chunk_past_key_values   
    generarte_args["max_new_tokens"] = 1024

    inputs = {
        "input_ids":torch.tensor(all_prompts_ids).unsqueeze(0).to(model.model.device).to(torch.int64),
        "attention_mask":torch.ones_like(torch.tensor(all_prompts_ids)).unsqueeze(0).to(model.model.device).to(torch.int64),
    }
    
    model.model.cache_fuse_metadata["cache_blend"]["check"] = True
    model.model.cache_fuse_metadata['cache_blend']['collect'] = False
    
    outputs = model.generate(**inputs,**generarte_args)
    print(tokenizer.decode(outputs.sequences[0],skip_special_tokens=True))


def task_translate(model,tokenizer,doc_prompts,system_prompt,question_prompt):
    generarte_args = {
        "max_new_tokens":1,
        "do_sample": False,
        "top_p": 0.9,
        "top_k": 0,
        "temperature": 0.1,
        "repetition_penalty": 1.0,
        "length_penalty": 1.0,
        "return_dict_in_generate":True,
    }
    
    doc_prompts = ["<|im_start|>\n"+doc_prompt+"\n<|im_end|>\n" for doc_prompt in doc_prompts]
    
    all_prompts = system_prompt + doc_prompts + question_prompt

    num_layer = len(model.model.layers)
    chunk_past_key_values = []
    all_prompts_ids = []

    for i in range(len(all_prompts)):
        inputs = tokenizer(all_prompts[i], return_tensors="pt").to(model.model.device)
        all_prompts_ids.extend(inputs['input_ids'][0].tolist())
        model.model.cache_fuse_metadata["cache_blend"]["collect"] = True
        model.model.cache_fuse_metadata["cache_blend"]["check"] = False 

        outputs = model.generate(**inputs,**generarte_args)
        # print(tokenizer.decode(outputs.sequences[0],skip_special_tokens=True))
        llm_layers = model.model.layers
        for j in range(num_layer):
            past_key_values = llm_layers[j].self_attn.hack_kv
            
            temp_k = past_key_values[0].clone()
            temp_v = past_key_values[1].clone()    

            if i == 0:
                chunk_past_key_values.append([temp_k, temp_v])
            else:
                chunk_past_key_values[j][0] = torch.cat((chunk_past_key_values[j][0],temp_k), dim=2)
                chunk_past_key_values[j][1] = torch.cat((chunk_past_key_values[j][1],temp_v), dim=2)
        # print(i,chunk_past_key_values[0][0].shape)
        # print(all_prompts[i],"\nlength:",len(inputs['input_ids'].tolist())
    global GLOBAL_KV_CACHE
    GLOBAL_KV_CACHE = chunk_past_key_values
    generarte_args["max_new_tokens"] = 1024

    inputs = {
        "input_ids":torch.tensor(all_prompts_ids).unsqueeze(0).to(model.model.device).to(torch.int64),
        "attention_mask":torch.ones_like(torch.tensor(all_prompts_ids)).unsqueeze(0).to(model.model.device).to(torch.int64),
    }
    
    model.model.cache_fuse_metadata["cache_blend"]["check"] = True
    model.model.cache_fuse_metadata['cache_blend']['collect'] = False
    
    outputs = model.generate(**inputs,**generarte_args)
    print(tokenizer.decode(outputs.sequences[0],skip_special_tokens=False))
    
def task_translate_2(model:CustomQwen2ForCausalLM,tokenizer:AutoTokenizer,doc_prompts,additional_doc_prompts,system_prompt,question_prompt):
    generarte_args = {
        "max_new_tokens":1,
        "do_sample": False,
        "top_p": 0.9,
        "top_k": 0,
        "temperature": 0.1,
        "repetition_penalty": 1.0,
        "length_penalty": 1.0,
        "return_dict_in_generate":True,
    }
    
    doc_prompts = ["<|im_start|>\n"+doc_prompt+"\n<|im_end|>\n" for doc_prompt in doc_prompts]
    additional_doc_prompts = ["<|im_start|>\n"+additional_doc_prompt+"\n<|im_end|>\n" for additional_doc_prompt in additional_doc_prompts]
    all_prompts = system_prompt + doc_prompts + question_prompt

    num_layer = len(model.model.layers)
    chunk_past_key_values = []
    all_prompts_ids = []
    each_chunk_indices = []

    for i in range(len(all_prompts)):
        inputs = tokenizer(all_prompts[i], return_tensors="pt").to(model.model.device)
        all_prompts_ids.append(inputs['input_ids'][0].tolist())
        each_chunk_indices.append(list(range(len(inputs['input_ids'][0].tolist()))))
        model.model.cache_fuse_metadata["cache_blend"]["collect"] = True
        model.model.cache_fuse_metadata["cache_blend"]["check"] = False 
        print(i,len(inputs['input_ids'][0].tolist()))
        outputs = model.generate(**inputs,**generarte_args)
        # print(tokenizer.decode(outputs.sequences[0],skip_special_tokens=True))
        llm_layers = model.model.layers
        for j in range(num_layer):
            past_key_values = llm_layers[j].self_attn.hack_kv
            
            temp_k = past_key_values[0].clone()
            temp_v = past_key_values[1].clone()    

            if i == 0:
                chunk_past_key_values.append([temp_k, temp_v])
            else:
                chunk_past_key_values[j][0] = torch.cat((chunk_past_key_values[j][0],temp_k), dim=2)
                chunk_past_key_values[j][1] = torch.cat((chunk_past_key_values[j][1],temp_v), dim=2)
    
    model.model.cache_fuse_metadata["use_additional_indices"] = True
    model.model.cache_fuse_metadata["cache_blend"]["check"] = True
    model.model.cache_fuse_metadata['cache_blend']['collect'] = False
    
    if  model.model.cache_fuse_metadata["use_additional_indices"]:
        old_chunk_end_pos = len(system_prompt + doc_prompts)
        for i,additional_doc_prompt in enumerate(additional_doc_prompts):
            inputs = tokenizer(additional_doc_prompt, return_tensors="pt").to(model.model.device)
            all_prompts_ids.insert(old_chunk_end_pos+i,inputs['input_ids'][0].tolist())
            each_chunk_indices.insert(old_chunk_end_pos+i,list(range(len(inputs['input_ids'][0].tolist()))))
    
    global GLOBAL_KV_CACHE
    GLOBAL_KV_CACHE = chunk_past_key_values
    generarte_args["max_new_tokens"] = 1024
    import itertools
    input_ids = list(itertools.chain(*all_prompts_ids))
    
    inputs = {
        "input_ids":torch.tensor(input_ids).unsqueeze(0).to(model.model.device).to(torch.int64),
        "attention_mask":torch.ones_like(torch.tensor(input_ids)).unsqueeze(0).to(model.model.device).to(torch.int64),
    }
    

    
    if model.model.cache_fuse_metadata["cache_blend"]["check"] and model.model.cache_fuse_metadata["use_additional_indices"]:
        # 处理indices
        new_map_indices = []
        old_chunks_last_index = 0
        additional_chunks_last_index = 0
        for i,chunk_index in enumerate(each_chunk_indices):
            if i == old_chunk_end_pos:
                old_chunks_last_index = new_map_indices[-1]
            if i == old_chunk_end_pos + len(additional_doc_prompts):
                additional_chunks_last_index = new_map_indices[-1]
            if i == 0:
                new_map_indices.extend(chunk_index)
            else:
                new_map_indices.extend([index+new_map_indices[-1]+1 for index in chunk_index])
           
        additional_map_indices = new_map_indices[old_chunks_last_index+1:additional_chunks_last_index+1]
        model.model.cache_fuse_metadata["additional_map_indices"] = torch.tensor(additional_map_indices).to(model.model.device).to(torch.int64)
        
        old_kv_map_indices = new_map_indices[:old_chunks_last_index+1]
        old_kv_map_indices.extend(new_map_indices[additional_chunks_last_index+1:])
        model.model.cache_fuse_metadata["old_kv_map_indices"] = torch.tensor(old_kv_map_indices).to(model.model.device).to(torch.int64)
        
        for layer_idx in range(len(model.model.layers)):
            old_k = GLOBAL_KV_CACHE[layer_idx][0]
            old_v = GLOBAL_KV_CACHE[layer_idx][1]
            new_k_shape =list( old_k.shape)
            new_k_shape[2] = len(new_map_indices)
            new_k = torch.zeros(new_k_shape).to(old_k.device).to(old_k.dtype)
            new_v = torch.zeros_like(new_k)
            
            new_k[:,:,model.model.cache_fuse_metadata["old_kv_map_indices"],:] = old_k
            new_v[:,:,model.model.cache_fuse_metadata["old_kv_map_indices"],:] = old_v
            GLOBAL_KV_CACHE[layer_idx][0] = new_k
            GLOBAL_KV_CACHE[layer_idx][1] = new_v
    outputs = model.generate(**inputs,**generarte_args)
    print(tokenizer.decode(outputs.sequences[0],skip_special_tokens=False))
    
    

if __name__=="__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = CustomQwen2ForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct",device_map="cuda",torch_dtype=torch.bfloat16).eval()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct",device_map="cuda")
    
    data = json.load(open("/root/code/vllm_plus/examples/bench_cache/data/2.json","r"))
    chunk_num = data['chunk_num']
    question_prompt = data['question']

    doc_prompts = [data["context"][f'{i}'] for i in range(chunk_num-2)]
    additional_doc_prompts = [data["context"][f'{i}'] for i in range(chunk_num-2,chunk_num)]
    system_prompt = ["<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"]
    
    question_prompt = ["<|im_start|>user\n" + question_prompt + "<|im_end|>\n<|im_start|>assistant\n"]
    
    # task_translate(model,tokenizer,doc_prompts,system_prompt,question_prompt)
    task_translate_2(model,tokenizer,doc_prompts,additional_doc_prompts,system_prompt,question_prompt)
    

   
    # data = json.load(open("/root/code/vllm_plus/examples/bench_cache/data/2.json","r"))
    # chunk_num = data['chunk_num']
    # question_prompt = ["<|im_start|>user\n" + "translate the chinese text to english" + "<|im_end|>\n<|im_start|>assistant\n"]

    # doc_prompts = [data[f'{i}'] for i in range(chunk_num)]
    # system_prompt = ["<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"]

    # task_translate(model,tokenizer,doc_prompts,system_prompt,question_prompt)
    
    