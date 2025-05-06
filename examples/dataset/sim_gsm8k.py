import json
import random
import datasets
from transformers import AutoTokenizer
from openai import OpenAI
import multiprocessing


def load_gsm8k_data(max_num=128,save_path=None):
    dataset = datasets.load_dataset(
        "openai/gsm8k",
        'main',
        split="test"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    # dataset = json.load(open("examples/dataset/data/samsum/train.json")) + json.load(open("examples/dataset/data/samsum/test.json"))
    data = []
    # dataset = random.sample(dataset,max_num)
    for item in dataset:
        tokens = tokenizer.encode(item["question"])
        # if len(tokens) <= 32 or len(tokens) >= 1024:
        #     continue
        data.append(
            {
                "answer": item["answer"],
                "target_doc": item["question"],
                "candidate_doc": item["question"]
            }
        )
    data = random.sample(data,max_num)
    print(len(data))
    json.dump(data,open(save_path,"w"),indent=4,ensure_ascii=False)


from transformers import AutoTokenizer
import random

import spacy

def random_split_chunks(text: str, chunk_size: int) -> list:
    """
    双层分块函数：先按固定token数分块，再随机拆分成两个子块
    
    参数：
    text: 原始文本字符串
    tokenizer: transformers的分词器对象
    chunk_size: 初始分块的token数（N）
    
    返回：
    List[str] 分块后的文本列表
    """
    # 第一阶段：按N个token分块\
    nlp = spacy.load('en_core_web_sm')  # 加载中文模型


    # # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    # # primary_chunks = [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]
    return sentences
    # return [text]


def gsm8k_candidate_generate(text):
    # 创建新的客户端实例，避免多进程共享问题
    local_client = OpenAI(
        api_key="sk-PMl5s5V78VDlTQoRhledqZ41fJIWJKTgjprIkYZrg7TxdvWK",
        base_url="https://www.dmxapi.cn/v1",
    )
    chat_completion = local_client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": f"""" \nText: {text}\n Please modify the characters related to the array in the input text. You can change them to any numbers. For example, change 50 to 77, and 18:00 to 23:39. Just give me the modified text.""",
        }
    ],
    model="gpt-4.1-mini",    #  替换成你先想用的模型全称， 模型全称可以在DMXAPI 模型价格页面找到并复制。
    )
    return chat_completion.choices[0].message.content

def process_item(item):
    input_text = item["target_doc"]
    output = gsm8k_candidate_generate(input_text)
    item["candidate_doc"] = output
    return item

def chunk_data(data,chunk_size=96):
    for item in data:
        chunk_tokens = random_split_chunks(item["candidate_doc"],chunk_size)
        item["candidates"] = ["\n "+chunk for chunk in chunk_tokens]
    return data


if __name__ == "__main__":  
    save_path = "examples/dataset/data/gsm8k/sim_gsm8k_benchmark_dataset.json"
    gpt_save_path ="examples/dataset/data/gsm8k/benchmark_gsm8k_gpt.json"
    gpt_save_chunk_path ="examples/dataset/data/gsm8k/sim_gsm8k_benchmark_dataset_chunk.json"
    # load_gsm8k_data(save_path=save_path,max_num=128)
    
    
    # data = json.load(open(save_path,"r"))
    # num_processes = min(64, len(data))
    # with multiprocessing.Pool(processes=num_processes) as pool:
    #     gpt_data = pool.map(process_item, data)
    
    # json.dump(gpt_data, open(gpt_save_path, "w"), indent=4, ensure_ascii=False)

    
    data = json.load(open(gpt_save_path,"r",encoding="utf-8"))
    data = chunk_data(data,chunk_size=32)
    json.dump(data,open(gpt_save_chunk_path,"w"),indent=4,ensure_ascii=False)
