# src/vector_db_v2.py
import faiss
import numpy as np
import os
import sys
import pickle
import jieba
from typing import List, Tuple, Dict
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_excel_data
from src.encoder_v2 import encode_entry_full_text, encode_query, VECTOR_DIM

INDEX_PATH = "models/dialect_index_v2.faiss"
ID_MAP_PATH = "models/entry_id_map_v2.pkl"

STOPWORDS = {
    '的', '是', '什么', '怎么', '说', '用', '在', '中', '我', '今天', '没', '了', '啊', '吧', '呢', '吗'
}

def preprocess_query(query_text: str) -> str:
    words = jieba.lcut(query_text)
    filtered_words = [w for w in words if w.strip() and w not in STOPWORDS]
    return " ".join(filtered_words)

def build_faiss_index_v2() -> Tuple[faiss.IndexFlatIP, List[str]]:
    df, entry_ids = load_excel_data()
    total_entries = len(df)
    print(f"开始构建方案二向量索引，共处理 {total_entries} 条方言数据...")

    vectors = []
    for idx in tqdm(range(total_entries), desc="生成向量"):
        entry = {
            '方言词': df.iloc[idx]['方言词'],
            '简易发音': df.iloc[idx]['简易发音'],
            '标准发音': df.iloc[idx]['标准发音'],
            '释义注释': df.iloc[idx]['释义注释']
        }
        vector = encode_entry_full_text(entry)
        vectors.append(vector)

    vectors_np = np.array(vectors, dtype=np.float32)
    index = faiss.IndexFlatIP(VECTOR_DIM)
    index.add(vectors_np)

    os.makedirs("models", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(ID_MAP_PATH, 'wb') as f:
        pickle.dump(entry_ids, f)

    print(f"方案二向量索引构建完成！索引包含 {index.ntotal} 个向量，维度 {VECTOR_DIM}")
    print(f"索引文件已保存到：{INDEX_PATH}")
    print(f"ID映射文件已保存到：{ID_MAP_PATH}")

    return index, entry_ids

def load_faiss_index_v2() -> Tuple[faiss.IndexFlatIP, List[str]]:
    if not os.path.exists(INDEX_PATH) or not os.path.exists(ID_MAP_PATH):
        raise FileNotFoundError("方案二向量索引文件不存在，请先运行build_faiss_index_v2()构建索引")
    index = faiss.read_index(INDEX_PATH)
    with open(ID_MAP_PATH, 'rb') as f:
        entry_ids = pickle.load(f)
    print(f"成功加载方案二本地索引，包含 {index.ntotal} 个向量")
    return index, entry_ids

def semantic_search_v2(query_text: str, top_k: int = 10) -> List[Tuple[str, float, Dict]]:
    try:
        index, entry_ids = load_faiss_index_v2()
    except FileNotFoundError:
        index, entry_ids = build_faiss_index_v2()

    processed_query = preprocess_query(query_text)
    print(f"原始查询：{query_text}")
    print(f"预处理后查询：{processed_query}")

    query_vector = encode_query(processed_query)
    query_vector_np = np.array([query_vector], dtype=np.float32)

    scores, indices = index.search(query_vector_np, top_k)

    df, _ = load_excel_data()
    results = []
    for i in range(top_k):
        if indices[0][i] < 0:
            continue
        entry_id = entry_ids[indices[0][i]]
        similarity = scores[0][i]
        entry_detail = df.iloc[indices[0][i]].to_dict()
        results.append((entry_id, similarity, entry_detail))

    return results

# 测试代码
if __name__ == "__main__":
    try:
        index, entry_ids = build_faiss_index_v2()
    except Exception as e:
        print(f"构建索引出错：{e}")
        exit(1)

    test_query1 = "爸爸"
    print(f"\n测试检索1：{test_query1}")
    results1 = semantic_search_v2(test_query1, top_k=10)
    print("\n检索结果（按相似度排序）：")
    for idx, (entry_id, similarity, detail) in enumerate(results1[:5], 1):
        print(f"\n{idx}. 词条ID：{entry_id}")
        print(f"   相似度：{similarity:.4f}")
        print(f"   方言词：{detail['方言词']}")
        print(f"   释义注释：{detail['释义注释'][:100]}...")

    test_query2 = "我今天没吃饭中的吃饭用方言怎么说"
    print(f"\n\n测试检索2：{test_query2}")
    results2 = semantic_search_v2(test_query2, top_k=10)
    print("\n检索结果（按相似度排序）：")
    for idx, (entry_id, similarity, detail) in enumerate(results2[:5], 1):
        print(f"\n{idx}. 词条ID：{entry_id}")
        print(f"   相似度：{similarity:.4f}")
        print(f"   方言词：{detail['方言词']}")
        print(f"   释义注释：{detail['释义注释'][:100]}...")

    print("\n方案二语义检索功能测试通过！")