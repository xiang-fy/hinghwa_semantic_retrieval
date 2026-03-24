#离线生成增强后的execl
import pandas as pd
import time
from tqdm import tqdm  # 进度条库
import dashscope
from http import HTTPStatus

# ====================== 配置区域 ======================
# 请替换为你的阿里云API Key（去阿里云百炼申请免费额度）
dashscope.api_key = "YOUR_API_KEY"
# 原始Excel路径
INPUT_EXCEL = "data/dialect_dict.xlsx"
# 生成的增强Excel路径
OUTPUT_EXCEL = "data/dialect_dict_augmented.xlsx"
# ====================== 配置结束 ======================

def generate_augmented_text(dialect_word: str, definition: str) -> str:
    """
    用大模型给单个方言词条生成增强文本
    :param dialect_word: 方言词（如“郎罢”）
    :param definition: 释义（如“父亲，爸爸”）
    :return: 增强文本（如“郎罢 爸爸 父亲 老爸 爹 爸爸用方言怎么说”）
    """
    # 精心设计的Prompt，约束大模型只输出我们需要的内容
    prompt = f"""
你是兴化方言专家，请为以下兴化方言词条生成检索增强文本。
要求：
1. 生成3-5个该词条的普通话同义词/近义词；
2. 生成2-3个用户可能用普通话提问的方式；
3. 所有内容用空格分隔，不要标点、不要换行、不要解释；
4. 必须包含原方言词和原释义。

示例：
方言词：郎罢
释义：父亲，爸爸
输出：郎罢 父亲 爸爸 老爸 爹 父亲用方言怎么说 爸爸用方言怎么说 父亲，爸爸

现在处理：
方言词：{dialect_word}
释义：{definition}
输出：
    """

    # 调用通义千问API（qwen-turbo免费额度足够）
    try:
        response = dashscope.Generation.call(
            model=dashscope.Generation.Models.qwen_turbo,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,  # 低温保证输出稳定
            max_tokens=150    # 限制输出长度
        )
        if response.status_code == HTTPStatus.OK:
            return response.output.choices[0].message.content.strip()
        else:
            # 如果API调用失败，返回原始内容保底
            return f"{dialect_word} {definition}"
    except Exception as e:
        print(f"生成出错（词条：{dialect_word}）：{e}")
        return f"{dialect_word} {definition}"

def main():
    """
    主函数：批量生成增强文本并保存
    """
    # 1. 读取原始Excel
    print(f"正在读取原始Excel：{INPUT_EXCEL}")
    df = pd.read_excel(INPUT_EXCEL, engine="openpyxl")
    
    # 2. 批量生成增强文本
    augmented_texts = []
    print(f"开始生成增强文本，共 {len(df)} 条词条...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        dialect_word = str(row["方言词"])
        definition = str(row["释义注释"])
        # 生成增强文本
        augmented = generate_augmented_text(dialect_word, definition)
        augmented_texts.append(augmented)
        # 加小延迟，避免API限流
        time.sleep(0.15)
    
    # 3. 把增强文本加入DataFrame
    df["检索增强文本"] = augmented_texts
    
    # 4. 保存为新的Excel
    df.to_excel(OUTPUT_EXCEL, index=False, engine="openpyxl")
    print(f"增强文本生成完成！保存到：{OUTPUT_EXCEL}")

if __name__ == "__main__":
    main()