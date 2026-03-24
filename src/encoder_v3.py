#生成向量
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from typing import Dict

# ====================== 配置区域 ======================
# 本地模型路径（bge-small-zh-v1.5）
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "bge-small-zh-v1.5")
# 模型输出向量维度（bge-small是512维）
VECTOR_DIM = 512
# ====================== 配置结束 ======================

# 全局加载模型，避免重复加载
_model = None

def load_model():
    """加载模型，全局单例"""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"模型不存在：{MODEL_PATH}")
        print("正在加载嵌入模型...")
        _model = SentenceTransformer(MODEL_PATH)
    return _model

def encode_single_field(text: str) -> np.ndarray:
    """
    对单个文本生成归一化向量
    :param text: 输入文本
    :return: 512维归一化向量
    """
    model = load_model()
    
    # 空文本返回全0向量
    if text.strip() == "" or text == "nan":
        return np.zeros(VECTOR_DIM)
    
    # 生成向量并归一化（保证余弦相似度计算准确）
    vector = model.encode(
        text,
        normalize_embeddings=True,  # 生成时直接归一化
        show_progress_bar=False     # 关闭进度条
    )
    return vector

def encode_entry_with_weights(entry: Dict[str, str]) -> np.ndarray:
    """
    对单条方言词条生成向量（只用增强文本）
    :param entry: 词条字典（必须包含"检索增强文本"）
    :return: 512维归一化向量
    """
    # 直接用增强文本生成向量
    augmented_text = entry.get("检索增强文本", "")
    vector = encode_single_field(augmented_text)
    
    # 再次归一化（保证模长为1）
    norm = np.linalg.norm(vector)
    if norm > 1e-6:
        vector = vector / norm
    return vector

def encode_query(query_text: str) -> np.ndarray:
    """
    对用户查询生成向量
    :param query_text: 用户查询
    :return: 512维归一化向量
    """
    return encode_single_field(query_text)

# 测试代码
if __name__ == "__main__":
    test_entry = {
        "检索增强文本": "郎罢 爸爸 父亲 老爸 爹 爸爸用方言怎么说"
    }
    vec = encode_entry_with_weights(test_entry)
    print("向量维度:", vec.shape)
    print("向量模长:", np.linalg.norm(vec))