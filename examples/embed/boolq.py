from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# 加载数据
dataset = load_dataset("boolq", split="train")
questions = [ex["question"] for ex in dataset]

# 生成嵌入
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embeddings = model.encode(questions, show_progress_bar=True)

# 计算相似度矩阵
similarity_matrix = cosine_similarity(embeddings)
np.fill_diagonal(similarity_matrix, -1)

# 提取相似对
threshold = 0.8
top_k = 5
similar_pairs = []
for i in range(len(questions)):
    sorted_indices = np.argsort(similarity_matrix[i])[::-1]
    for j in sorted_indices[:top_k]:
        if similarity_matrix[i][j] > threshold:
            similar_pairs.append({
                "text1": questions[i],
                "text2": questions[j],
                "similarity": similarity_matrix[i][j]
            })

# 清洗结果
df = pd.DataFrame(similar_pairs)
df = df[df['text1'] != df['text2']]
df = df.sort_values(by='similarity', ascending=False).drop_duplicates(subset=['similarity'])
df.to_csv("boolq_similar_pairs.csv", index=False)
print(df.head(10))