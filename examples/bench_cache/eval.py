import json
import numpy as np
# import evaluate
from evaluate import load

if __name__ == "__main__":
    model_name = "Qwen2.5-7B-Instruct"
    prefix_cache_data = json.load(open(f"examples/bench_cache/eval/{model_name}_wmt_en_zh_prefix_cache.json", "r"))
    semantic_cache_data = json.load(open(f"examples/bench_cache/eval/{model_name}_wmt_en_zh_semantic_cache.json", "r"))
    no_cache_data = json.load(open(f"examples/bench_cache/eval/{model_name}_wmt_en_zh_no_cache.json", "r"))

    key = "example1"
    speed_up1 = []
    speed_up2 = []
    meteor1 = []
    meteor2 = []
    f11=[]
    f12=[]
    f13=[]
    bertscore = load("bertscore")
    for item1, item2, item3 in zip( semantic_cache_data[key],prefix_cache_data[key],no_cache_data[key]):
        ttft1 = item1["metrics"]["ttft"]
        ttft2 = item2["metrics"]["ttft"]
        ttft3 = item3["metrics"]["ttft"]
        # if ttft1/ttft2 < 1.0:
        #     continue
        speed_up1.append(ttft3/ttft1)
        speed_up2.append(ttft2/ttft2)
        
        results = bertscore.compute(predictions=[item1["gen_en"]], references=[item1["ref_en"]], lang="en")
        f11.append(results["f1"])
        
        results = bertscore.compute(predictions=[item2["gen_en"]], references=[item2["ref_en"]], lang="en")
        f12.append(results["f1"])

        results = bertscore.compute(predictions=[item3["gen_en"]], references=[item3["ref_en"]], lang="en")
        f13.append(results["f1"])
        # meteor1.append(item1["metrics"]["meteor"])
        # meteor2.append(item2["metrics"]["meteor"])
    
    print("f11：",np.mean(f11))
    print("f12：",np.mean(f12))
    print("f13：",np.mean(f13))
    print("speed_up1, avg:",np.mean(speed_up1)," min:",np.min(speed_up1)," max:",np.max(speed_up1))
    print("speed_up2, avg:",np.mean(speed_up2)," min:",np.min(speed_up2)," max:",np.max(speed_up2))
