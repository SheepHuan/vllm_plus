import json
import uuid
import random
import datasets
from transformers import AutoTokenizer
from tqdm import tqdm


def load_samsum(max_num=256):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    dataset = json.load(open("examples/dataset/data/samsum/train.json"))
    data = []
    # dataset = random.sample(dataset,max_num)
    for item in dataset:
        tokens = tokenizer.encode(item["dialogue"])
        if len(tokens) <= 256 or len(tokens) >= 2048:
            continue
        data.append(
            {
                "answer": item["summary"],
                "target_doc": item["dialogue"] + " \n " + "Please summarize the main content of the following text. The summary should be concise and clear, and key information should be retained. ",
                "candidate_doc": item["dialogue"] + " \n " + "Please summarize the main content of the following text. The summary should be concise and clear, and key information should be retained. "
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
    
    # 按chunk_size分块
    nlp = spacy.load('en_core_web_sm')  # 加载英文模型
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    
    # 检查每个句子的token长度并合并
    merged_sentences = []
    current_chunk = ""
    
    for i, sent in enumerate(sentences):
        current_sent_tokens = tokenizer.encode(sent)
        
        if len(current_sent_tokens) < 10:
            # 如果当前句子token数小于10，尝试与下一个句子合并
            if i < len(sentences) - 1:
                next_sent = sentences[i + 1]
                combined_tokens = tokenizer.encode(sent + " " + next_sent)
                if len(combined_tokens) > 10:
                    current_chunk = sent + " " + next_sent
                    sentences[i + 1] = current_chunk  # 更新下一个句子
                    continue
            # 如果没有下一个句子或合并后仍小于10，与当前chunk合并
            if current_chunk:
                current_chunk += " " + sent
            else:
                current_chunk = sent
        else:
            # 如果当前句子token数大于等于10
            if current_chunk:
                merged_sentences.append(current_chunk)
                current_chunk = ""
            merged_sentences.append(sent)
    
    # 处理最后一个chunk
    if current_chunk:
        merged_sentences.append(current_chunk)
    
    # 去重，保持顺序
    seen = set()
    merged_sentences = [x for x in merged_sentences if not (x in seen or seen.add(x))]
    
    # 添加分隔符
    merged_sentences = [" \n\n\n\n\n\n\n\n\n\n\n\n\n\n " + item + " \n\n\n\n\n\n\n\n\n\n\n\n\n\n " for item in merged_sentences]
    return merged_sentences
    
    


def chunk_data(data,chunk_size=96):
    for item in data:
        chunk_tokens = random_split_chunks(item["candidate_doc"],chunk_size) + [ "Please summarize the main content of the above text. The summary should be concise and clear, and key information should be retained. "]
        item["candidates"] = [" \n\n\n\n\n\n\n\n\n "+chunk+" \n\n\n\n\n\n\n\n\n " for chunk in chunk_tokens]
        item["target_doc"] = item["target_doc"] + " \n " +  "Please summarize the main content of the above text. The summary should be concise and clear, and key information should be retained. "
    return data

if __name__ == "__main__":
    # load_samsum()
    # data = json.load(open("examples/dataset/data/samsum/sim_samsum_benchmark_dataset.json","r",encoding="utf-8"))
    # with multiprocessing.Pool(processes=16) as pool:
    #     data = list(tqdm(pool.imap(process_item, data), total=len(data)))
    # json.dump(data,open("examples/dataset/data/samsum/sim_samsum_benchmark_dataset_gpt.json","w"),indent=4,ensure_ascii=False)

    data = json.load(open("examples/dataset/data/samsum/sim_samsum_benchmark_dataset.json","r",encoding="utf-8"))
    data = chunk_data(data,chunk_size=64)
    json.dump(data,open("examples/dataset/data/samsum/sim_samsum_benchmark_dataset_chunk.json","w"),indent=4,ensure_ascii=False)
