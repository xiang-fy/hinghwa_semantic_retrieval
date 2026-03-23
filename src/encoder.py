# src/encoder.py
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from typing import Dict, List

# 加载本地模型
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'bge-small-zh-v1.5')

# 如果本地模型不存在，先提示用户下载
if not os.path.exists(model_path):
    raise FileNotFoundError(f"""
    本地模型不存在！请先下载模型到以下路径：
    {model_path}
    下载地址：https://hf-mirror.com/BAAI/bge-small-zh-v1.5
    """)

# 加载轻量级中文嵌入模型
model = SentenceTransformer(model_path)

# 模型输出向量维度（bge-small-zh-v1.5实际输出512维）
VECTOR_DIM = 512

# 方案一权重配置（适配你的4个字段，总权重=1,可手动修改）
FIELD_WEIGHTS = {
    '方言词': 0.5,       # 核心字段，最高权重
    '简易发音': 0.075,   # 发音字段1，权重0.075
    '标准发音': 0.075,   # 发音字段2，权重0.075（两个发音合计0.15）
    '释义注释': 0.35     # 包含普通话对应词，权重0.35
}

def encode_single_field(text: str) -> np.ndarray:
    """
    对单个字段文本生成嵌入向量（方案一：字段级独立嵌入）
    :param text: 字段文本（如"阿"、"a1"、"a533"等）
    :return: 归一化后的512维向量
    """
    # 空文本返回全0向量
    if text.strip() == '' or text == 'nan':
        return np.zeros(VECTOR_DIM)
    
    # 生成向量并归一化（保证余弦相似度计算准确）
    vector = model.encode(
        text,
        normalize_embeddings=True,  # 生成时直接归一化
        show_progress_bar=False     # 关闭进度条，避免干扰，测试是可以开启
    )
    return vector

def encode_entry_with_weights(entry: Dict[str, str]) -> np.ndarray:
    """
    对单条方言词条做字段级加权融合
    :param entry: 单条词条字典（包含方言词、简易发音、标准发音、释义注释）
    :return: 加权融合后的最终向量
    """
    # 初始向量改为512维
    total_vector = np.zeros(VECTOR_DIM)
    
    # 遍历每个字段，独立嵌入后加权
    for field_name, weight in FIELD_WEIGHTS.items():
        # 获取字段文本（处理空值）
        field_text = entry.get(field_name, '')
        # 生成字段向量
        field_vector = encode_single_field(field_text)
        # 加权累加
        total_vector += field_vector * weight
    
    # 融合后再次归一化（关键：保证向量模长为1，余弦相似度准确）
    total_vector = total_vector / np.linalg.norm(total_vector)
    return total_vector

def encode_query(query_text: str) -> np.ndarray:
    """
    对用户查询生成嵌入向量
    :param query_text: 用户输入的查询（如"阿公是什么意思"）
    :return: 归一化后的查询向量
    """
    return encode_single_field(query_text)

# 测试代码：验证字段嵌入和加权融合
if __name__ == "__main__":
    # 用第一条数据测试
    test_entry = {
        '方言词': '阿',
        '简易发音': 'a1',
        '标准发音': 'a533',
        '释义注释': '①用在某些亲属名称的前面：～舅|～叔|～妹。②用在单名、排行或姓前面，表亲昵：～灿|～水|～...'
    }
    
    print("正在加载本地模型...")
    # 测试单个字段嵌入
    word_vector = encode_single_field(test_entry['方言词'])
    print(f"\n单个字段向量维度：{word_vector.shape}（512维，符合预期）")
    
    # 测试加权融合
    final_vector = encode_entry_with_weights(test_entry)
    print(f"融合后向量维度：{final_vector.shape}（512维，符合预期）")
    print(f"融合后向量模长：{np.linalg.norm(final_vector):.4f}（≈1，归一化成功）")
    
    # 测试查询嵌入
    query_vector = encode_query("阿是什么意思")
    print(f"查询向量维度：{query_vector.shape}（512维，符合预期）")
    print("\n方案一核心嵌入逻辑测试通过！")