"""拼音意图解析器

功能：使用LLM从自然语言中提取拼音片段
支持解析用户输入中的拼音查询意图
"""
import re
import requests
import os
from dotenv import load_dotenv
from typing import Dict, List, Optional

# 加载环境变量
load_dotenv()

# 拼音模式匹配
PINYIN_PATTERN = re.compile(r'[a-zA-Züv]+[1-6]?', re.IGNORECASE)


def parse_pinyin_intent(user_input: str) -> Dict:
    """
    使用LLM解析用户输入中的拼音查询意图
    
    返回格式：
    {
        "intent": "pinyin" | "text" | "mixed",
        "pinyin_parts": ["拼音1", "拼音2", ...],
        "text_parts": ["文本1", "文本2", ...],
        "confidence": 0.0-1.0
    }
    """
    # 1. 首先使用规则提取拼音片段
    pinyin_parts = extract_pinyin_parts(user_input)
    text_parts = extract_text_parts(user_input)
    
    # 2. 如果没有明显的拼音片段，尝试使用LLM解析
    if not pinyin_parts:
        llm_result = _call_llm_parser(user_input)
        if llm_result:
            pinyin_parts = llm_result.get('pinyin_parts', [])
            text_parts = llm_result.get('text_parts', [])
    
    # 3. 判断意图类型
    intent = _determine_intent(pinyin_parts, text_parts)
    
    return {
        "intent": intent,
        "pinyin_parts": pinyin_parts,
        "text_parts": text_parts,
        "confidence": _calculate_confidence(intent, pinyin_parts, text_parts)
    }


def extract_pinyin_parts(text: str) -> List[str]:
    """从文本中提取拼音片段"""
    matches = PINYIN_PATTERN.findall(text)
    # 过滤掉太短的匹配（至少2个字符）
    return [match.lower() for match in matches if len(match) >= 2]


def extract_text_parts(text: str) -> List[str]:
    """从文本中提取中文文本片段"""
    # 提取中文字符序列
    chinese_pattern = re.compile(r'[\u4e00-\u9fa5]+')
    matches = chinese_pattern.findall(text)
    return matches


def _call_llm_parser(user_input: str) -> Optional[Dict]:
    """调用LLM解析器"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    
    if not api_key:
        return None
    
    url = f"{base_url}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    prompt = f"""
你是一个拼音意图解析助手，请分析用户输入并提取其中的拼音片段。

用户输入：{user_input}

请输出JSON格式：
{{
    "pinyin_parts": ["拼音1", "拼音2", ...],
    "text_parts": ["文本1", "文本2", ...]
}}

注意：
- 拼音片段：只包含字母（可以包含声调数字1-6）
- 文本片段：只包含中文
- 如果没有对应类型的内容，返回空数组
"""
    
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 100
    }
    
    try:
        response = requests.post(url, json=data, headers=headers, timeout=3)
        response.raise_for_status()
        result = response.json()["choices"][0]["message"]["content"].strip()
        return eval(result)
    except Exception as e:
        print(f"LLM解析出错：{e}")
        return None


def _determine_intent(pinyin_parts: List[str], text_parts: List[str]) -> str:
    """判断意图类型"""
    has_pinyin = len(pinyin_parts) > 0
    has_text = len(text_parts) > 0
    
    if has_pinyin and not has_text:
        return "pinyin"
    elif not has_pinyin and has_text:
        return "text"
    else:
        return "mixed"


def _calculate_confidence(intent: str, pinyin_parts: List[str], text_parts: List[str]) -> float:
    """计算置信度"""
    total_parts = len(pinyin_parts) + len(text_parts)
    
    if total_parts == 0:
        return 0.0
    
    if intent == "pinyin":
        return len(pinyin_parts) / total_parts
    elif intent == "text":
        return len(text_parts) / total_parts
    else:
        # 混合模式下，如果拼音占比超过50%，置信度更高
        pinyin_ratio = len(pinyin_parts) / total_parts
        return 0.7 + pinyin_ratio * 0.3


if __name__ == "__main__":
    test_cases = [
        "langba是什么意思",
        "用莆田话怎么说爸爸",
        "tiau对应的方言词",
        "我想查一下putian方言",
        "普通话语义词"
    ]
    
    for test in test_cases:
        result = parse_pinyin_intent(test)
        print(f"输入: {test}")
        print(f"  结果: {result}")
        print()