import json
import datasets
import random
from openai import OpenAI
import multiprocessing

def load_musique_data(save_path):
    dataset = datasets.load_dataset("bdsaglam/musique","answerable", split="train")
    data  = []

    for item in dataset:
        chunks = [p["paragraph_text"] for p in item["paragraphs"]]
        question = item["question"]
        answer = item["answer"]
        
        target_doc = "\n".join(chunks)
        data.append({
            "target_doc": target_doc,
            "answer": answer,
            "candidate_doc": target_doc,
            "question": question
        })
    data = random.sample(data,128)
    json.dump(data,open(save_path,"w"),indent=4,ensure_ascii=False)
    return data

def sim_content(text,question,answer):
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
I will input a text, a question, and the answer to the question. I want to simplify the input text by retaining only the content related to the question and the answer. \nText: {text} \n Question: {question} \n Answer: {answer} \n Just give me the modified text.
            """,
        }
    ],
    model="gpt-4.1-mini",    #  替换成你先想用的模型全称， 模型全称可以在DMXAPI 模型价格页面找到并复制。
    )
    return chat_completion.choices[0].message.content

def check_content(text,question,answer):
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
Please modify and replace some people's names, names of building locations, and some important times in the text. For example, replace "Paris" with "Singapore", "1920" with "2004", and "Beethoven" with "Trump".\nText: {text} \n Just give me the modified text.
            """,
        }
    ],
    model="gpt-4.1-mini",    #  替换成你先想用的模型全称， 模型全称可以在DMXAPI 模型价格页面找到并复制。
    )
    return chat_completion.choices[0].message.content

def process_item(item):
    input_text = item["target_doc"]
    answer = item["answer"]
    question = item["question"]
    # text,question = input_text.split("\n====================\n")
    output = sim_content(input_text,question,answer)
    item["target_doc"] = output
    return item

def change_item(item):
    input_text = item["target_doc"]
    answer = item["answer"]
    question = item["question"]
    output = check_content(input_text,question,answer)
    item["candidate_doc"] = output
    return item

import spacy

def random_split_chunks(text: str) -> list:
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

    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    # # primary_chunks = [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]
    return sentences
    # return [text]


import os
os.makedirs("examples/dataset/data/musique",exist_ok=True)
if __name__ == "__main__":
    save_path = "examples/dataset/data/musique/musique_benchmark.json"
    gpt_save_path ="examples/dataset/data/musique/musique_benchmark_gpt.json"
    gpt_save_chunk_path ="examples/dataset/data/musique/musique_benchmark_gpt_chunk.json"
    
    # load_musique_data(save_path)
    # data = json.load(open(save_path,"r"))
    # num_processes = min(64, len(data))
    
    # with multiprocessing.Pool(processes=num_processes) as pool:
    #     data = pool.map(change_item, data)
    # json.dump(data,open(gpt_save_path,"w"),indent=4,ensure_ascii=False)


    # data = json.load(open(gpt_save_path,"r"))
    # num_processes = min(64, len(data))
    # with multiprocessing.Pool(processes=num_processes) as pool:
    #     data = pool.map(change_item, data)
    # json.dump(data,open(gpt_save_path,"w"),indent=4,ensure_ascii=False)


    data = json.load(open(gpt_save_path,"r"))
    new_data = []
    for item in data:
        chunks = random_split_chunks(item["target_doc"])
        # for chunk in chunks:
        item["candidates"] = chunks + [item["question"]]
        item["candidates"] = ["\n "+ chunk for chunk in item["candidates"] ]
        new_data.append(item)
        item["target_doc"] = item["target_doc"]+"\n"+item["question"]
    json.dump(new_data,open(gpt_save_chunk_path,"w"),indent=4,ensure_ascii=False)








