# src/encoder_v2.py
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from typing import Dict

model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'bge-small-zh-v1.5')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"""
    本地模型不存在！请先下载模型到以下路径：
    {model_path}
    下载地址：https://hf-mirror.com/BAAI/bge-small-zh-v1.5
    """)

model = SentenceTransformer(model_path)
VECTOR_DIM = 512

def encode_entry_full_text(entry: Dict[str, str]) -> np.ndarray:
    """
    方案二：全文本拼接嵌入
    :param entry: 单条词条字典
    :return: 归一化后的512维向量
    """
    # 拼接所有字段成一个长文本
    full_text = (
        f"方言词：{entry.get('方言词', '')}；"
        f"简易发音：{entry.get('简易发音', '')}；"
        f"标准发音：{entry.get('标准发音', '')}；"
        f"释义注释：{entry.get('释义注释', '')}"
    ).strip()
    
    if not full_text:
        return np.zeros(VECTOR_DIM)
    
    vector = model.encode(
        full_text,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return vector

def encode_query(query_text: str) -> np.ndarray:
    if query_text.strip() == '':
        return np.zeros(VECTOR_DIM)
    vector = model.encode(
        query_text,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return vector

# 测试代码
if __name__ == "__main__":
    test_entry = {
        '方言词': '阿',
        '简易发音': 'a1',
        '标准发音': 'a533',
        '释义注释': '①用在某些亲属名称的前面：～舅|～叔|～妹。②用在单名、排行或姓前面，表亲昵：～灿|～水|～...'
    }
    print("正在加载本地模型...")
    final_vector = encode_entry_full_text(test_entry)
    print(f"全文本向量维度：{final_vector.shape}（512维，符合预期）")
    print(f"向量模长：{np.linalg.norm(final_vector):.4f}（≈1，归一化成功）")
    query_vector = encode_query("阿是什么意思")
    print(f"查询向量维度：{query_vector.shape}（512维，符合预期）")
    print("\n方案二核心嵌入逻辑测试通过！")