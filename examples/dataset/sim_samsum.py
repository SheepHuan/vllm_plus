import json
import uuid
import random


class Context:
    def __init__(self, docs):
        self.docs = docs
        self.uid = str(uuid.uuid4())
        
    def get_context(self):
        document = "\n".join([doc["document"] for doc in self.docs])
        summary = "\n".join([doc["summary"] for doc in self.docs])
        return document,summary
        # "\n".join(self.docs)
        
    def get_sentence(self):
        return random.choice(self.docs)


def generate_context(docs,max_num=256):
    docs = random.sample(docs,max_num)
    contexts = []
    for i in range(0,len(docs),1):
        contexts.append(Context(docs[i:i+1]))
    return contexts

def get_data(docs,max_num=128):
    contexts = generate_context(docs)
    new_data = []
    
    
    for _ in range(max_num):
        sub_contexts = random.sample(contexts,1)
        sub_sentences = []
        for c in sub_contexts:
            sub_sentences.append(c.get_sentence())
        
        new_context = Context(sub_sentences)
        new_data.append([new_context,[c.uid for c in sub_contexts]])
    
    save_data = {
        "candidates":{},
        "targets":[]
    }
    prompt = "\nPlease summarize the above information in the shortest text possible. "
    # 先保存
    for data in contexts:
        input,output = data.get_context()
        save_data["candidates"][data.uid] = {
            "uid":data.uid,
            "input":input,
            # "output":output
        }
    
    for data in new_data:
        input,output = data[0].get_context()
        save_data["targets"].append({
            "uid":data[0].uid,
            "candidates":data[1],
            "input":input,
            "output":output
        })
    
    return save_data
    
## 从all_docs中随机不重复选择1000条数据

## 每5个句子顺序成一个上下文，生成200个上下文。

## 从200个上下文中随机不重复抽取3个上下文，分别拿出一句话，组成一个新的上下文。


if __name__ == "__main__":
    input_path = "examples/dataset/data/samsum/all-mini-l6-v2_samsum_similar_docs_topk50.json"
    data = json.load(open(input_path, "r", encoding="utf-8"))
    all_docs = data["all_documents"]
    all_paris = data["similar_docs"]
    # keys,values= zip(*all_paris)
    new_data = get_data(list(all_docs.values()))
    json.dump(new_data,open("examples/dataset/data/samsum/benchmark_samsum_dataset.json","w",encoding="utf-8"),indent=4,ensure_ascii=False)
