# src/data_loader.py
import pandas as pd
import os

def load_excel_data(excel_path: str = "data/dialect_dict.xlsx"):
    """
    读取兴化方言词典Excel数据，完成基础清洗
    :param excel_path: Excel文件路径
    :return: 清洗后的DataFrame，以及词条ID列表
    """
    # 检查文件是否存在
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel文件不存在：{excel_path}，请将文件放入data/文件夹下")
    
    print(" 正在读取Excel数据...")
    
    # 读取Excel（.xlsx格式用openpyxl引擎）
    # 明确指定我们的4列表头
    df = pd.read_excel(
        excel_path,
        engine='openpyxl',
        usecols=['方言词', '简易发音', '标准发音', '释义注释']  
    )
    
    # 基础清洗：去除空行、去除首尾空格
    df = df.dropna(how='all').copy()
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    
    # 生成唯一词条ID
    df['entry_id'] = [f"entry_{i:04d}" for i in range(len(df))]
    
    print(f" Excel数据加载成功！共加载 {len(df)} 条方言词条")
    print(f" 识别到的字段：{df.columns.tolist()}")
    
    return df, df['entry_id'].tolist()

if __name__ == "__main__":
    # 测试读取
    try:
        df, _ = load_excel_data()
        print("\n 前3条数据预览：")
        print(df.head(3))
        print("\n 数据加载测试通过！")
    except Exception as e:
        print(f" 错误：{e}")