#构建向量库并执行检索
import faiss
import numpy as np
import os
import sys
import pickle
from typing import List, Tuple, Dict
from tqdm import tqdm

# 把项目根目录加入Python路径，解决导入问题
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_excel_data
from src.encoder import encode_entry_with_weights, encode_query, VECTOR_DIM
from src.query_rewriter import rewrite_query

# ====================== 配置区域 ======================
# FAISS索引保存路径
INDEX_PATH = "models/dialect_index_augmented.faiss"
# 词条ID映射保存路径
ID_MAP_PATH = "models/entry_id_map_augmented.pkl"
# ====================== 配置结束 ======================

def build_faiss_index() -> Tuple[faiss.IndexFlatIP, List[str]]:
    """
    构建FAISS向量索引
    :return: FAISS索引对象，词条ID列表
    """
    # 1. 加载增强版Excel数据
    df, entry_ids = load_excel_data()
    
    # 2. 生成所有词条的向量
    vectors = []
    print("开始生成向量...")
    for i in tqdm(range(len(df)), desc="生成向量"):
        row = df.iloc[i]
        entry = {
            "方言词": row["方言词"],
            "检索增强文本": row["检索增强文本"]
        }
        vec = encode_entry_with_weights(entry)
        vectors.append(vec)
    
    # 3. 转换为FAISS支持的numpy数组（float32）
    vectors_np = np.array(vectors, dtype=np.float32)
    
    # 4. 构建FAISS索引（IndexFlatIP：内积索引，归一化后等价于余弦相似度）
    index = faiss.IndexFlatIP(VECTOR_DIM)
    index.add(vectors_np)
    
    # 5. 保存索引和ID映射到本地
    os.makedirs("models", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(ID_MAP_PATH, "wb") as f:
        pickle.dump(entry_ids, f)
    
    print(f"索引构建完成！共 {index.ntotal} 条向量")
    print(f"索引文件：{INDEX_PATH}")
    print(f"ID映射：{ID_MAP_PATH}")
    return index, entry_ids

def load_faiss_index() -> Tuple[faiss.IndexFlatIP, List[str]]:
    """
    加载本地已保存的FAISS索引
    :return: FAISS索引对象，词条ID列表
    """
    # 如果索引不存在，先构建
    if not os.path.exists(INDEX_PATH) or not os.path.exists(ID_MAP_PATH):
        return build_faiss_index()
    
    # 加载索引
    index = faiss.read_index(INDEX_PATH)
    # 加载ID映射
    with open(ID_MAP_PATH, "rb") as f:
        entry_ids = pickle.load(f)
    
    print(f"成功加载本地索引，共 {index.ntotal} 条向量")
    return index, entry_ids

def semantic_search(query_text: str, top_k: int = 10) -> List[Tuple[str, float, Dict]]:
    """
    语义检索核心函数
    :param query_text: 用户原始查询
    :param top_k: 返回最相似的top_k条结果
    :return: 检索结果列表（词条ID, 相似度, 词条详情）
    """
    # 1. 加载索引
    index, entry_ids = load_faiss_index()
    
    # 2. 用大模型重写查询（理解意图+扩展同义词）
    processed_query = rewrite_query(query_text)
    
    # 3. 生成查询向量
    query_vector = encode_query(processed_query)
    query_vector_np = np.array([query_vector], dtype=np.float32)
    
    # 4. 执行检索
    scores, indices = index.search(query_vector_np, top_k)
    
    # 5. 加载原始数据，组装结果
    df, _ = load_excel_data()
    results = []
    for i in range(top_k):
        idx = indices[0][i]
        if idx < 0:
            continue
        entry_id = entry_ids[idx]
        similarity = float(scores[0][i])
        entry_detail = df.iloc[idx].to_dict()
        results.append((entry_id, similarity, entry_detail))
    
    return results

# 测试代码
if __name__ == "__main__":
    # 先构建索引
    build_faiss_index()
    # 测试检索
    print("\n测试检索：爸爸")
    results = semantic_search("爸爸", top_k=5)
    for i, (eid, score, detail) in enumerate(results, 1):
        print(f"{i}. 相似度：{score:.3f} | 方言词：{detail['方言词']} | 释义：{detail['释义注释'][:50]}...")