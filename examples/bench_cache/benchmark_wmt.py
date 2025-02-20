from vllm import LLM, SamplingParams
import torch
import json
from datasets import load_dataset
import evaluate
import tqdm
from typing import List, Dict
import colorlog



model_name = "Qwen/Qwen2.5-7B-Instruct"
dataset_name = "wmt/wmt19"


ttft_vllm = []

def init_logger():
    handler = colorlog.StreamHandler()
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)s: %(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    handler.setFormatter(formatter)
    logger = colorlog.getLogger("benchmark")
    logger.addHandler(handler)
    logger.setLevel(colorlog.DEBUG)
    return logger

logger = init_logger()


def benchmark(llm: LLM, conversation: List[Dict[str, str]],sampling_params: SamplingParams):
    outputs = llm.chat(conversation,
                   sampling_params=sampling_params,
                   use_tqdm=False)
    return outputs

def benmark_vllm_without_cache(wmt_path, save_path, enable_prefix_caching=False):
    llm = LLM(model=model_name, trust_remote_code=True, enable_prefix_caching=enable_prefix_caching)
    sampling_params = SamplingParams(temperature=0.1, top_p=0.95,max_tokens=2048)
    dataset = json.load(open(wmt_path, 'r'))
    bleu = evaluate.load("bleu")
    meteor = evaluate.load('meteor')
    save_data ={}
    logger.info(f"Start benchmark {model_name}, enable_prefix_caching: {enable_prefix_caching}")
    for key, data_list in dataset.items():
        print(key)
        save_data[key] = []
        for item in tqdm.tqdm(data_list, desc="Processing", unit="item"):
            en = item['en']
            zh = item['zh']
            conversation = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant"
                },
                {
                    "role": "user",
                    "content": f"Translate the following Chinese sentence to English: {zh}"
                }
            ]
            bleu_results_list = []
            meteor_results_list = []
            ttft_vllm_list = []
            gen_en_list = []
            for _ in range(10):
                outputs = benchmark(llm, conversation, sampling_params)
                pred_text = outputs[0].outputs[0].text
                bleu_results = bleu.compute(predictions=[pred_text], references=[en])
                bleu_results_list.append(bleu_results)
                meteor_results = meteor.compute(predictions=[pred_text], references=[en])
                meteor_results_list.append(meteor_results)
                ttft_vllm_list.append(outputs[0].metrics.first_token_time-outputs[0].metrics.first_scheduled_time)
                gen_en_list.append(pred_text)
            avg_bleu = sum([result['bleu'] for result in bleu_results_list]) / len(bleu_results_list)
            avg_meteor = sum([result['meteor'] for result in meteor_results_list]) / len(meteor_results_list)
            avg_ttft_vllm = sum(ttft_vllm_list[1:-1]) / len(ttft_vllm_list[1:-1])
            logger.info(f'{ttft_vllm_list}')
            logger.info(f"{key}, avg_bleu: {avg_bleu}, avg_meteor: {avg_meteor}, avg_ttft_vllm: {avg_ttft_vllm}")
            save_data[key].append({
                'ref_en': en,
                'gen_en': gen_en_list,
                "ref_zh": zh,
                "metrics": {
                    'avg_bleu': avg_bleu,
                    'avg_meteor': avg_meteor,
                    'avg_ttft_vllm': avg_ttft_vllm
                }
            })
    json.dump(save_data, open(save_path, 'w'), indent=4, ensure_ascii=False)


wmt_path = 'examples/benchmark/data/wmt19_zh_en.json'
save_path = 'examples/benchmark/data/benchmark_wmt19_vllm_cache.json'
benmark_vllm_without_cache(wmt_path, save_path,True)

