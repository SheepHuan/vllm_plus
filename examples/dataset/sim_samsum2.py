import json
import uuid
import random
import datasets
from transformers import AutoTokenizer
from tqdm import tqdm


def load_samsum(max_num=256):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    dataset = json.load(open("examples/dataset/data/samsum/train.json")) + json.load(open("examples/dataset/data/samsum/test.json"))
    data = []
    # dataset = random.sample(dataset,max_num)
    for item in dataset:
        tokens = tokenizer.encode(item["dialogue"])
        if len(tokens) <= 256 or len(tokens) >= 512:
            continue
        data.append(
            {
                "answer": item["summary"],
                "target_doc": item["dialogue"],
                "candidate_doc": item["dialogue"]
            }
        )
    data = random.sample(data,max_num)
    print(len(data))
    json.dump(data,open("examples/dataset/data/samsum/sim_samsum_benchmark_dataset.json","w"),indent=4,ensure_ascii=False)
    
    
import json
from openai import OpenAI
import multiprocessing
from functools import partial

client = OpenAI(
    api_key="sk-PMl5s5V78VDlTQoRhledqZ41fJIWJKTgjprIkYZrg7TxdvWK",  # 替换成你的 DMXapi 令牌key
    base_url="https://www.dmxapi.cn/v1",  # 需要改成DMXAPI的中转 https://www.dmxapi.com/v1 ，这是已经改好的。
)


def samsum_candidate_generate(text):
    # 创建新的客户端实例，避免多进程共享问题
    local_client = OpenAI(
        api_key="sk-PMl5s5V78VDlTQoRhledqZ41fJIWJKTgjprIkYZrg7TxdvWK",
        base_url="https://www.dmxapi.cn/v1",
    )
    chat_completion = local_client.chat.completions.create(
    messages=[
        {
            "role": "user",
             "content": f""""
            1. Please modify and replace the names of the main characters in the text I input. For example, change Lisa to LiasABC, change Peter to PeterDFS. You need change all the names in text.
            2. Expand and modify the numbers that appear in the text. For instance, change 1 to 11, 13 to 133, and 4 to 554. 
            3. Also modify the time that appears in the text. For example, change 7:00 to 18:00. 
            You can use your own discretion when modifying the numbers and time. 
            \nText: {text} \n Just give me the modified text.
            """,
        }
    ],
    model="gpt-4.1-mini",    #  替换成你先想用的模型全称， 模型全称可以在DMXAPI 模型价格页面找到并复制。
    )
    return chat_completion.choices[0].message.content

def process_item(item):
    input_text = item["target_doc"]
    output = samsum_candidate_generate(input_text)
    item["candidate_doc"] = output
    return item




from transformers import AutoTokenizer
import random

import spacy

def random_split_chunks(text: str, chunk_size: int) -> list:
    """
    双层分块函数：先按固定token数分块，再随机拆分成两个子块
    
    参数：
    text: 原始文本字符串
    chunk_size: 初始分块的token数（N）
    
    返回：
    List[str] 分块后的文本列表
    """
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    tokens = tokenizer.encode(text)
    
    # 按chunk_size分块
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunks.append(tokens[i:i+chunk_size])
    
    # 处理最后一个chunk
    if len(chunks) > 1 and len(chunks[-1]) < chunk_size:
        # 如果最后一个chunk不足chunk_size且存在前一个chunk，则合并到前一个chunk
        chunks[-2].extend(chunks.pop())
    
    # 将每个chunk的tokens解码为字符串
    return [tokenizer.decode(chunk) for chunk in chunks]
    
    


def chunk_data(data,chunk_size=96):
    for item in data:
        chunk_tokens = random_split_chunks(item["candidate_doc"],chunk_size) + [ "Please summarize the main content of the following text. The summary should be concise and clear, and key information should be retained. "]
        item["candidates"] = [" \n \n \n \n \n \n \n \n \n "+chunk+" \n \n \n \n \n \n \n \n \n " for chunk in chunk_tokens]
        item["target_doc"] = item["target_doc"] + " \n " +  "Please summarize the main content of the following text. The summary should be concise and clear, and key information should be retained. "
    return data

if __name__ == "__main__":
    load_samsum()
    # data = json.load(open("examples/dataset/data/samsum/sim_samsum_benchmark_dataset.json","r",encoding="utf-8"))
    # with multiprocessing.Pool(processes=16) as pool:
    #     data = list(tqdm(pool.imap(process_item, data), total=len(data)))
    # json.dump(data,open("examples/dataset/data/samsum/sim_samsum_benchmark_dataset_gpt.json","w"),indent=4,ensure_ascii=False)

    data = json.load(open("examples/dataset/data/samsum/sim_samsum_benchmark_dataset.json","r",encoding="utf-8"))
    data = chunk_data(data,chunk_size=64)
    json.dump(data,open("examples/dataset/data/samsum/sim_samsum_benchmark_dataset_chunk.json","w"),indent=4,ensure_ascii=False)
