# src/vector_db.py
import faiss
import numpy as np
import os
import sys
import pickle
from typing import List, Tuple, Dict
from tqdm import tqdm  # 进度条，方便查看向量生成进度

# 解决导入路径问题：将项目根目录加入Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入本地模块
from src.data_loader import load_excel_data
from src.encoder import encode_entry_with_weights, encode_query, VECTOR_DIM

# 向量索引保存路径
INDEX_PATH = "models/dialect_index.faiss"
# 词条ID映射文件路径
ID_MAP_PATH = "models/entry_id_map.pkl"

def build_faiss_index() -> Tuple[faiss.IndexFlatIP, List[str]]:
    """
    构建FAISS向量索引（内积索引，归一化后等价于余弦相似度）
    :return: FAISS索引对象，词条ID列表（和索引向量一一对应）
    """
    # 1. 加载方言数据
    df, entry_ids = load_excel_data()
    total_entries = len(df)
    print(f"开始构建向量索引，共处理 {total_entries} 条方言数据...")

    # 2. 生成所有词条的融合向量
    vectors = []
    for idx in tqdm(range(total_entries), desc="生成向量"):
        # 提取单条词条的字段数据
        entry = {
            '方言词': df.iloc[idx]['方言词'],
            '简易发音': df.iloc[idx]['简易发音'],
            '标准发音': df.iloc[idx]['标准发音'],
            '释义注释': df.iloc[idx]['释义注释']
        }
        # 生成加权融合向量
        vector = encode_entry_with_weights(entry)
        vectors.append(vector)

    # 3. 转换为FAISS支持的numpy数组（float32类型）
    vectors_np = np.array(vectors, dtype=np.float32)

    # 4. 构建FAISS索引（IndexFlatIP：内积索引，适配归一化向量的余弦相似度）
    index = faiss.IndexFlatIP(VECTOR_DIM)
    index.add(vectors_np)

    # 5. 保存索引和ID映射到本地
    os.makedirs("models", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(ID_MAP_PATH, 'wb') as f:
        pickle.dump(entry_ids, f)

    print(f"向量索引构建完成！索引包含 {index.ntotal} 个向量，维度 {VECTOR_DIM}")
    print(f"索引文件已保存到：{INDEX_PATH}")
    print(f"ID映射文件已保存到：{ID_MAP_PATH}")

    return index, entry_ids

def load_faiss_index() -> Tuple[faiss.IndexFlatIP, List[str]]:
    """
    加载本地已保存的FAISS索引和ID映射
    :return: FAISS索引对象，词条ID列表
    """
    if not os.path.exists(INDEX_PATH) or not os.path.exists(ID_MAP_PATH):
        raise FileNotFoundError("向量索引文件不存在，请先运行build_faiss_index()构建索引")

    # 加载索引
    index = faiss.read_index(INDEX_PATH)
    # 加载ID映射
    with open(ID_MAP_PATH, 'rb') as f:
        entry_ids = pickle.load(f)

    print(f"成功加载本地索引，包含 {index.ntotal} 个向量")
    return index, entry_ids

def semantic_search(query_text: str, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
    """
    语义检索核心函数
    :param query_text: 用户查询文本（如"阿公是什么意思"）
    :param top_k: 返回最相似的top_k条结果
    :return: 检索结果列表，每个元素包含（entry_id, 相似度, 词条详情）
    """
    # 1. 加载索引（如果不存在则先构建）
    try:
        index, entry_ids = load_faiss_index()
    except FileNotFoundError:
        index, entry_ids = build_faiss_index()

    # 2. 生成查询向量
    query_vector = encode_query(query_text)
    query_vector_np = np.array([query_vector], dtype=np.float32)

    # 3. 执行检索（返回相似度得分和索引）
    scores, indices = index.search(query_vector_np, top_k)

    # 4. 加载原始数据，组装结果
    df, _ = load_excel_data()
    results = []
    for i in range(top_k):
        if indices[0][i] < 0:  # 无结果时跳过
            continue
        # 获取词条ID
        entry_id = entry_ids[indices[0][i]]
        # 获取相似度（内积得分，归一化后范围0-1，越高越相似）
        similarity = scores[0][i]
        # 获取词条详情
        entry_detail = df.iloc[indices[0][i]].to_dict()
        results.append((entry_id, similarity, entry_detail))

    return results

# 测试代码：验证向量索引构建和语义检索
if __name__ == "__main__":
    # 第一步：构建索引（首次运行耗时约1-2分钟，5820条数据）
    try:
        index, entry_ids = build_faiss_index()
    except Exception as e:
        print(f"构建索引出错：{e}")
        exit(1)

    # 第二步：测试语义检索
    test_query = "阿公是什么意思"
    print(f"\n测试检索：{test_query}")
    results = semantic_search(test_query, top_k=3)

    # 打印检索结果
    print("\n检索结果（按相似度排序）：")
    for idx, (entry_id, similarity, detail) in enumerate(results, 1):
        print(f"\n{idx}. 词条ID：{entry_id}")
        print(f"   相似度：{similarity:.4f}")
        print(f"   方言词：{detail['方言词']}")
        print(f"   简易发音：{detail['简易发音']}")
        print(f"   标准发音：{detail['标准发音']}")
        print(f"   释义注释：{detail['释义注释'][:100]}...")  # 只显示前100字

    print("\n语义检索功能测试通过！")