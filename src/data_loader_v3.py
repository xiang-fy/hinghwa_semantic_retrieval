#加载增强版Excel数据
import pandas as pd
import os

def load_excel_data(excel_path: str = "data/dialect_dict_augmented.xlsx"):
    """
    读取增强版Excel数据
    :param excel_path: 增强版Excel路径
    :return: 清洗后的DataFrame，词条ID列表
    """
    # 检查文件是否存在
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"增强版Excel不存在：{excel_path}，请先运行generate_augmented_data.py生成")
    
    # 读取Excel
    df = pd.read_excel(excel_path, engine="openpyxl")
    
    # 基础清洗：去除空行、去除首尾空格
    df = df.dropna(how="all").copy()
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    
    # 生成唯一词条ID
    df["entry_id"] = [f"entry_{i:04d}" for i in range(len(df))]
    
    print(f"增强版Excel加载成功！共 {len(df)} 条词条")
    return df, df["entry_id"].tolist()