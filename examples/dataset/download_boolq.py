# download_boolq.py
from datasets import load_dataset

# 配置
cache_dir = "examples/bench_cache/data/boolq"
split = "train"  # or "validation"

# 下载数据集
dataset = load_dataset(
    "google/boolq",
    split=split,
    cache_dir=cache_dir,
    streaming=False
)

# 验证加载
print(f"数据集大小: {len(dataset)}")
print("示例数据:", dataset[0])