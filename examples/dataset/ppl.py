import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from evaluate import logging

class PerplexityMetric:
    def __init__(self, model_id="gpt2", device=None):
        """
        初始化困惑度计算器
        Args:
            model_id: 使用的模型ID
            device: 计算设备 ('cuda', 'cpu', 'gpu')
        """
        self.device = device
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        
    def _load_model(self):
        """加载模型和分词器"""
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, 
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16
        )
        self.model = self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        if self.tokenizer.pad_token is None:
            existing_special_tokens = list(self.tokenizer.special_tokens_map_extended.values())
            assert len(existing_special_tokens) > 0, "Model must have at least one special token to use for padding."
            self.tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})
            
    def compute(self, texts, max_length=None):
        """
        计算文本的困惑度
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            add_start_token: 是否添加开始标记
            max_length: 最大序列长度
        Returns:
            dict: 包含每个文本的困惑度和平均困惑度
        """
        if self.model is None:
            self._load_model()
            
     

        encodings = self.tokenizer(
            texts,
            add_special_tokens=False,
            padding=True,
            truncation=True if max_length else False,
            max_length=max_length,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.device)

        ppls = []
        loss_fct = CrossEntropyLoss(reduction="none")

        labels = encodings["input_ids"]
        attn_masks = encodings["attention_mask"]

        with torch.no_grad():
            out_logits = self.model(**encodings).logits

   
        # 分开计算每个文本的困惑度
        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask = attn_masks[..., 1:].contiguous().to(torch.int32)

        e = loss_fct(shift_logits.transpose(1, 2), shift_labels)
        x = (e * shift_attention_mask).sum(1) / shift_attention_mask.sum(1)
        perplexity = torch.exp(x)
        ppls = perplexity.tolist()
        
        torch.cuda.empty_cache()  # 添加这一行以释放显存
        return ppls
        # return {
        #     "perplexities": ppls,
        #     "mean_perplexity": np.mean(ppls)
        # }
        
    def __del__(self):
        """清理模型"""
        if self.model is not None:
            del self.model
            self.model = None 