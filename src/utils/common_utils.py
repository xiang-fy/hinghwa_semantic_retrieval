import re

_UNIFIED_IPA_SPECIAL = {"ɒ", "ɔ", "ø", "œ", "ŋ", "ɬ", "ʔ", "ʰ", "˥", "˧", "˨", "˩", "¹", "²", "³", "⁴", "⁵", "⁶", "⁷", "⁸"}
_UNIFIED_ALLOWED_ASCII = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ")
def clean_ipa_str(ipa_str):
    # 先打印原始输入
    # print(f"[调试] 原始输入IPA：{repr(ipa_str)}")
    
    # 只做最基础的清洗：去首尾空格、中间多余空格
    ipa_str = ipa_str.strip()
    ipa_str = re.sub(r"\s+", " ", ipa_str)
    
    # 打印清洗后的结果
    # print(f"[调试] 清洗后IPA：{repr(ipa_str)}")
    return ipa_str

def format_match_result(item):
    return {
        "方言词": item.get("方言词", ""),
        "简易发音": item.get("简易发音", ""),
        "标准发音": item.get("标准发音", ""),
        "释义注释": item.get("释义注释", ""),
        "相似度": round(item.get("score", 0.0), 4)
    }