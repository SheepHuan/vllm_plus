# 在PyTorch代码中添加显存重置
import torch
from ctypes import CDLL
libc = CDLL("libc.so.6")

def clean_gpu_context():
    torch.cuda.empty_cache()
    libc.malloc_trim(0)  # 释放glibc的内存池