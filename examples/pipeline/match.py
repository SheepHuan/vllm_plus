from pymilvus import MilvusClient
from pymilvus import IndexType
from sentence_transformers import SentenceTransformer
from typing import List
import time
import random
import hashlib
# from datasketch import MinHashLSH

def timestamp_long():
    a = str(int(time.time() * 1000))
    a = a + str(random.randint(0,9))
    a = int(a)
    return a

class KVDataBase:
    def __init__(self,sentence_model_name,db_path,connection_name):
        self.db_path = db_path
       
        self.connection_name = connection_name
        
        
        self.sentence_model = SentenceTransformer(sentence_model_name,device="cuda:0")
        self.model_dimension = self.sentence_model.get_sentence_embedding_dimension()
        self.client = MilvusClient(db_path)
        self.collection = self.client.create_collection(
            collection_name=self.connection_name,
            dimension=self.model_dimension,  # The vectors we will use in this demo has 768 dimensions
            index_type=IndexType.HNSW,
           )
    def embed(self,text):
        return self.sentence_model.encode(text)
    
    def insert(self,text,metadata={}):
        id = timestamp_long() * 1000
        embeddings = self.embed(text)
        hash_key = hashlib.md5(text.encode()).hexdigest()
        res = self.client.search(
                    collection_name=self.connection_name,
                    data=[embeddings],
                    limit=1,
                    output_fields=["text", "hash_key"],
                )
        if res!=[]:
            if len(res[0]) != 0 and res[0][0]["distance"] >= 0.99 and res[0][0]["entity"]["hash_key"] == hash_key:
                return
        
        data = [{
            "text":text,
            "vector":embeddings,
            "metadata":metadata,
            "id":id,
            "hash_key":hash_key
        }]
        self.client.insert(collection_name=self.connection_name, data=data)
        
        
    def query(self,text:str):
        start_time = time.time()
        embeddings = self.embed(text)
        time2 = time.time()
        print(f"embed time: {time2 - start_time} seconds")
        
        time3 = time.time()
        res = self.client.search(
                collection_name=self.connection_name,  # target collection
                data=[embeddings],  # query vectors
                limit=100,  # number of returned entities
                output_fields=["text","embeddings"],  # specifies fields to be returned
            )
        end_time = time.time()
        print(f"search time: {end_time - time3} seconds")
        return res



if __name__ == "__main__":
    texts = [
        "我在年青时候也曾经做过许多梦，后来大半忘却了，但自己也并不以为可惜。所谓回忆者，虽说可以使人欢欣，有时也不免使人寂寞，使精神的丝缕还牵着己逝的寂寞的时光，又有什么意味呢，而我偏苦于不能全忘却，这不能全忘的一部分，到现在便成了《呐喊》的来由。",
        "我有四年多，曾经常常，——几乎是每天，出入于质铺和药店里，年纪可是忘却了，总之是药店的柜台正和我一样高，质铺的是比我高一倍，我从一倍高的柜台外送上衣服或首饰去，在侮蔑里接了钱，再到一样高的柜台上给我久病的父亲去买药。",   
        "回家之后，又须忙别的事了，因为开方的医生是最有名的，以此所用的药引也奇特：冬天的芦根，经霜三年的甘蔗，蟋蟀要原对的，结子的平地木，……多不是容易办到的东西。然而我的父亲终于日重一日的亡故了。"
        "有谁从小康人家而坠入困顿的么，我以为在这途路中，大概可以看见世人的真面目；我要到 Ｎ进Ｋ学堂去了，仿佛是想走异路，逃异地，去寻求别样的人们。",
        "我的母亲没有法，办了八元的川资，说是由我的自便；然而伊哭了，这正是情理中的事，因为那时读书应试是正路，所谓学洋务，社会上便以为是一种走投无路的人，只得将灵魂卖给鬼子，要加倍的奚落而且排斥的，而况伊又看不见自己的儿子了。",
       "然而我也顾不得这些事，终于到 Ｎ去进了Ｋ学堂了，在这学堂里，我才知道世上还有所谓格致，算学，地理，历史，绘图和体操。"
       "生理学并不教，但我们却看到些木版的《全体新论》和《化学卫生论》之类了。"
       "我还记得先前的医生的议论和方药，和现在所知道的比较起来，便渐渐的悟得中医不过是一种有意的或无意的骗子，同时又很起了对于被骗的病人和他的家族的同情；而且从译出的历史上，又知道了日本维新是大半发端于西方医学的事实。"
    ]
    
    
    model_name = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
    db_path = "examples/pipeline/data/milvus_demo.db"
    connection_name = "text_match1"
    kv_database = KVDataBase(model_name,db_path,connection_name)
    
    for text in texts:
        kv_database.insert(text=text)
        
    res = kv_database.query("\n".join(texts))
    if len(res) != 0:
        res = res[0]
        if len(res) != 0:
            for i in range(len(res)):
                print(res[i]["distance"],res[i]["entity"]["text"])
                print("-"*100)