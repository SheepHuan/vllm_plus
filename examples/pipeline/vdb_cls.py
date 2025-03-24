from pymilvus import MilvusClient
from typing import List


class VectorDB:
    def __init__(self,
                 collection_name:str,
                 dimension:int,
                 database_path:str,
                 ):
        options = [
            ('grpc.max_receive_message_length', 1500 * 1024 * 1024),  # 500MB
            ('grpc.max_send_message_length', 1500 * 1024 * 1024)     # 500MB
        ]

        
        self.client = MilvusClient(database_path,   channel_options=options)
        self.collection_name = collection_name
        self.create_collection(collection_name,dimension)
        self.client.load_collection(collection_name)
        
    def insert(self,data:List[dict]):
        self.client.upsert(collection_name=self.collection_name,data=data)
    
    def search_by_id(self,ids:List[int]):
        return self.client.get(collection_name=self.collection_name,ids=ids)
    
    def delete_collection(self):
        self.client.release_collection(collection_name=self.collection_name)
        
    
    def create_collection(self,collection_name:str,dimension:int):
        self.client.create_collection(collection_name=collection_name,dimension=dimension)
    
    