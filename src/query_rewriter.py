#重构理解用户意图
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ====================== 配置区域 ======================
# 轻量级查询重写模型（Qwen2-0.5B-Instruct，仅1GB，CPU可跑）
MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
# 国内镜像地址
MODEL_MIRROR = "https://hf-mirror.com"
# ====================== 配置结束 ======================

# 全局加载模型和tokenizer
_model = None
_tokenizer = None

# 精心设计的系统提示词，约束大模型只做查询重写
SYSTEM_PROMPT = """
你是兴化方言查询重写助手。
用户输入的是关于兴化方言的自然语言问题。
你的任务是：
1. 提取用户要查询的核心普通话词汇/短语；
2. 生成3-5个该核心词的同义词/近义词；
3. 所有内容用空格分隔，只输出关键词，不要任何解释、标点、换行。

示例：
用户输入：我今天没吃饭中的“吃饭”用方言怎么说
你输出：吃饭 用餐 进餐
用户输入：我膝盖疼用方言怎么说
你输出：膝盖 膝关节 膝头
用户输入：阿公是什么意思
你输出：阿公 祖父 外祖父 爷爷
"""

def load_rewriter():
    """加载查询重写模型，全局单例"""
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        print("正在加载查询重写模型（首次运行会自动下载，约1GB）...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, mirror=MODEL_MIRROR)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,  # CPU用float32
            device_map="auto",
            mirror=MODEL_MIRROR
        )
        print("查询重写模型加载完成！")
    return _model, _tokenizer

def rewrite_query(user_query: str) -> str:
    """
    对用户查询做重写
    :param user_query: 用户原始自然语言查询
    :return: 重写后的检索关键词
    """
    model, tokenizer = load_rewriter()
    
    # 构建对话格式
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # 生成结果（低温保证稳定）
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=64,
            temperature=0.1,
            do_sample=False
        )
    # 只取生成的部分，去掉输入部分
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    rewritten_query = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    # 打印日志，方便调试
    print(f"原始查询：{user_query}")
    print(f"重写后查询：{rewritten_query}")
    return rewritten_query

# 测试代码
if __name__ == "__main__":
    test_queries = [
        "我今天没吃饭中的“吃饭”用方言怎么说",
        "我膝盖疼用方言怎么说",
        "爸爸"
    ]
    for q in test_queries:
        print("\n" + "="*50)
        rewrite_query(q)