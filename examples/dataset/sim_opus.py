import json
import uuid
import random
# input_path = "examples/dataset/data/opus/zero_score_samples.json"
# output_path = "examples/dataset/data/opus/benchmark_opus.json"

# data = json.load(open(input_path, "r", encoding="utf-8"))


# all_docs = data["all_translations"]
# all_paris = data["similar_pairs"]

# class Context:
#     def __init__(self, docs):
#         self.docs = docs
#         self.uid = str(uuid.uuid4())
        
#     def get_context(self):
#         zh = "\n".join([doc["zh"] for doc in self.docs])
#         en = "\n".join([doc["en"] for doc in self.docs])
#         return zh,en
#         # "\n".join(self.docs)
        
#     def get_sentence(self):
#         return random.choice(self.docs)


# def generate_context(docs,max_num=512):
#     docs = random.sample(docs,max_num)
#     contexts = []
#     for i in range(0,len(docs),4):
#         contexts.append(Context(docs[i:i+4]))
#     return contexts

# def get_data(docs,max_num=128):
#     contexts = generate_context(docs)
#     new_data = []
    
    
#     for _ in range(max_num):
#         sub_contexts = random.sample(contexts,3)
#         sub_sentences = []
#         for c in sub_contexts:
#             sub_sentences.append(c.get_sentence())
        
#         new_context = Context(sub_sentences)
#         new_data.append([new_context,[c.uid for c in sub_contexts]])
    
#     save_data = {
#         "candidates":{},
#         "targets":[]
#     }
#     prompt = "Please translate the following text from English to Chinese. {text}"
#     # 先保存
#     for data in contexts:
#         zh,en = data.get_context()
#         save_data["candidates"][data.uid] = {
#             "uid":data.uid,
#             "input":zh,
#             "output":en
#         }
    
#     for data in new_data:
#         zh,en = data[0].get_context()
#         save_data["targets"].append({
#             "uid":data[0].uid,
#             "candidates":data[1],
#             "input":zh,
#             "output":en
#         })
    
#     return save_data
    
## 从all_docs中随机不重复选择1000条数据

## 每5个句子顺序成一个上下文，生成200个上下文。

## 从200个上下文中随机不重复抽取3个上下文，分别拿出一句话，组成一个新的上下文。


# if __name__ == "__main__":
#     input_path = "examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_top50_250403_windows.json"
#     data = json.load(open(input_path, "r", encoding="utf-8"))
#     all_docs = data["all_translations"]
#     all_paris = data["similar_pairs"]
#     # keys,values= zip(*all_paris)
#     new_data = get_data(list(all_docs.values()))
#     json.dump(new_data,open("examples/dataset/data/opus/benchmark_syc_opus_dataset.json","w",encoding="utf-8"),indent=4,ensure_ascii=False)

import datasets

def load_opus():
    config = "en-zh"
    dataset_path = "Helsinki-NLP/opus-100"
    # validation_dataset = list(datasets.load_dataset(dataset_path,config,split="validation"))
    test_dataset = list(datasets.load_dataset(dataset_path,config,split="test"))
    # train_dataset = list(datasets.load_dataset(dataset_path,config,split="train"))
    data = []
    test_dataset = random.sample(test_dataset,256)
    for item in test_dataset:
        answer = item["translation"]["en"]
        target_doc =item["translation"]["zh"]
        data.append({
            "answer": answer,
            "target_doc": target_doc
        })
    json.dump(data,open("examples/dataset/data/opus/sim_benchmark_opus.json","w"),indent=4,ensure_ascii=False)

load_opus()