import json
import datasets
import random
from openai import OpenAI
import multiprocessing
from transformers import AutoTokenizer

def load_musique_data(save_path):
    dataset = datasets.load_dataset("bdsaglam/musique","answerable", split="train")
    data  = []

    for item in dataset:
        chunks = [p["paragraph_text"] for p in item["paragraphs"]]
        question = item["question"]
        answer = item["answer"]
        
        target_doc = "\n".join(chunks)
        target_doc = "Message: \n " + target_doc
        target_doc = "Please answer the question based on the following message. Use the format \"|Answer: result|\" to return the key answer.\n Example1: |Answer: Dobsonia| , Example2: |Answer: off Midtown Manhattan|, Example3: |Answer: 190,884| \n" + target_doc
        data.append({
            "target_doc": target_doc + " \n Question: " + question,
            "answer": answer,
            "candidate_doc": target_doc + " \n Question: " + question,
            # "question": question
        })
    data = random.sample(data,128)
    json.dump(data,open(save_path,"w"),indent=4,ensure_ascii=False)
    return data

# def sim_content(text,question,answer):
#     # 创建新的客户端实例，避免多进程共享问题
#     local_client = OpenAI(
#         api_key="sk-PMl5s5V78VDlTQoRhledqZ41fJIWJKTgjprIkYZrg7TxdvWK",
#         base_url="https://www.dmxapi.cn/v1",
#     )
#     chat_completion = local_client.chat.completions.create(
#     messages=[
#         {
#             "role": "user",
#              "content": f""""
# I will input a text, a question, and the answer to the question. I want to simplify the input text by retaining only the content related to the question and the answer. \nText: {text} \n Question: {question} \n Answer: {answer} \n Just give me the modified text.
#             """,
#         }
#     ],
#     model="gpt-4.1-mini",    #  替换成你先想用的模型全称， 模型全称可以在DMXAPI 模型价格页面找到并复制。
#     )
#     return chat_completion.choices[0].message.content

# def check_content(text,question,answer):
#     # 创建新的客户端实例，避免多进程共享问题
#     local_client = OpenAI(
#         api_key="sk-PMl5s5V78VDlTQoRhledqZ41fJIWJKTgjprIkYZrg7TxdvWK",
#         base_url="https://www.dmxapi.cn/v1",
#     )
#     chat_completion = local_client.chat.completions.create(
#     messages=[
#         {
#             "role": "user",
#              "content": f""""
# Please modify and replace some people's names, names of building locations, and some important times in the text. For example, replace "Paris" with "Singapore", "1920" with "2004", and "Beethoven" with "Trump".\nText: {text} \n Just give me the modified text.
#             """,
#         }
#     ],
#     model="gpt-4.1-mini",    #  替换成你先想用的模型全称， 模型全称可以在DMXAPI 模型价格页面找到并复制。
#     )
#     return chat_completion.choices[0].message.content

# def process_item(item):
#     input_text = item["target_doc"]
#     answer = item["answer"]
#     question = item["question"]
#     # text,question = input_text.split("\n====================\n")
#     output = sim_content(input_text,question,answer)
#     item["target_doc"] = output
#     return item

# def change_item(item):
#     input_text = item["target_doc"]
#     answer = item["answer"]
#     question = item["question"]
#     output = check_content(input_text,question,answer)
#     item["candidate_doc"] = output
#     return item

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


import os
os.makedirs("examples/dataset/data/musique",exist_ok=True)
if __name__ == "__main__":
    save_path = "examples/dataset/data/musique/musique_benchmark.json"
    gpt_save_path ="examples/dataset/data/musique/musique_benchmark_gpt.json"
    gpt_save_chunk_path ="examples/dataset/data/musique/musique_benchmark_gpt_chunk.json"
    
    load_musique_data(save_path)
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


    # data = json.load(open(save_path,"r"))
    # new_data = []
    # for item in data:
    #     chunks = random_split_chunks(item["candidate_doc"],96)
    
    #     item["candidates"] = chunks
    #     # item["candidates"] = ["\n "+ chunk for chunk in item["candidates"] ]
    #     new_data.append(item)
    #     # item["target_doc"] = item["target_doc"]+"\n"+item["question"]
    # json.dump(new_data,open(gpt_save_chunk_path,"w"),indent=4,ensure_ascii=False)

    # data = json.load(open(gpt_save_chunk_path,"r"))
    # for item in data:
    #     item["target_doc"] = "Please answer the question based on the following text. \n" + item["target_doc"]
    # json.dump(data,open(gpt_save_chunk_path,"w"),indent=4,ensure_ascii=False)






