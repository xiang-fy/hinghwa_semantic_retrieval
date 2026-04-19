from src.query_rewriter import parse_query
from src.vector_db import core_search
from src.result_formatter import format_result
# ========== 新架构导入 ==========
from src.matcher import MatcherManager
from src.utils.common_utils import clean_ipa_str
from src.data_loader import FIELD_MAPPING
from typing import List, Dict, Optional, Set
import re

# ========== 日志屏蔽 ==========
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# ========== 缓存功能必需的导入 =====================
import json
import os
import hashlib

# ========== 全局配置 ==========
ENABLE_IPA_MATCH = True
INPUT_RECOGNITION_MODE = "strict"

# ========== 缓存路径配置（自动创建，不用管）==========
CACHE_DIR = "cache"
IPA_CHAR_CACHE = os.path.join(CACHE_DIR, "ipa_chars.json")
DATASET_SIGNATURE = os.path.join(CACHE_DIR, "dataset_signature.txt")

# ========== 全局统一匹配管理器（新架构核心）==========
matcher_manager = MatcherManager()

# ========== 模块化意图提取器（未来扩展用） ==========
class IntentExtractor:
    """
    模块化意图提取器：模糊/拼音/IPA 通用
    用于从混合输入中提取核心内容
    """
    @staticmethod
    def extract_ipa(user_input: str) -> Optional[str]:
        return None

    @staticmethod
    def extract_pinyin(user_input: str) -> Optional[str]:
        return None

# ========== 动态 IPA 识别器（保留，非常好用）==========
class DynamicIPARecognizer:
    def __init__(self, valid_ipa_chars: set):
        self.valid_ipa_chars = valid_ipa_chars
        self.basic_chars = set("abcdefghijklmnopqrstuvwxyz0123456789 ")

    def is_ipa_input(self, user_input: str) -> bool:
        s = user_input.strip()
        if not s:
            return False

        # 包含中文 → 不是 IPA
        if re.search(r"[\u4e00-\u9fa5]", s):
            return False

        # 字符必须合法
        for c in s:
            if c not in self.basic_chars and c not in self.valid_ipa_chars:
                return False

        # 必须带数字（简易发音 / IPA 都有）
        return any(c.isdigit() for c in s)

# ========== 缓存工具类（保留，优化性能）==========
class IPACharCache:
    def __init__(self, all_ipa_list):
        self.all_ipa_list = all_ipa_list

    def _get_dataset_signature(self) -> str:
        try:
            content = "|".join(self.all_ipa_list)
            return hashlib.md5(content.encode("utf-8")).hexdigest()
        except:
            return "unknown"

    def load(self) -> Optional[Set[str]]:
        os.makedirs(CACHE_DIR, exist_ok=True)
        if not os.path.exists(IPA_CHAR_CACHE) or not os.path.exists(DATASET_SIGNATURE):
            return None

        with open(DATASET_SIGNATURE, "r", encoding="utf-8") as f:
            cached_sig = f.read().strip()
        if cached_sig != self._get_dataset_signature():
            return None

        with open(IPA_CHAR_CACHE, "r", encoding="utf-8") as f:
            return set(json.load(f))

    def save(self, chars: Set[str]):
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(IPA_CHAR_CACHE, "w", encoding="utf-8") as f:
            json.dump(list(chars), f, ensure_ascii=False)
        with open(DATASET_SIGNATURE, "w", encoding="utf-8") as f:
            f.write(self._get_dataset_signature())

    def get_chars(self) -> Set[str]:
        cached = self.load()
        if cached is not None:
            return cached

        chars = set()
        for ipa in self.all_ipa_list:
            for c in ipa:
                chars.add(c)
        self.save(chars)
        return chars

# ========== 可扩展融合查询管理器（已完全重构）==========
class ExtensibleFusionQueryManager:
    def __init__(self):
        self.original_enabled = True
        self.ipa_enabled = ENABLE_IPA_MATCH
        self.ipa_recognizer = None

        if self.ipa_enabled:
            try:
                # ==============================================
                # 【关键】使用新架构的 MatcherManager
                # ==============================================
                self.ipa_matcher = matcher_manager

                # 从新 IPA 匹配器获取所有 IPA 列表
                all_ipa = matcher_manager.ipa_matcher.all_ipa_list
                cache = IPACharCache(all_ipa)
                self.valid_ipa_chars = cache.get_chars()
                self.ipa_recognizer = DynamicIPARecognizer(self.valid_ipa_chars)

            except Exception as e:
                print(f"IPA 模块加载失败：{e}")
                self.ipa_enabled = False

    def query(self, user_input: str) -> str:
        if self.ipa_enabled and self.ipa_recognizer.is_ipa_input(user_input):
            return self._ipa_query_path(user_input)
        else:
            return self._original_query_path(user_input)

    def _ipa_query_path(self, user_input: str) -> str:
        """
        新版 IPA 查询：
        自动支持 → 简易发音精准 + 标准IPA精准 + 模糊匹配
        """
        res = self.ipa_matcher.ipa_query(user_input, top_k=5)
        if res:
            return format_result(self._adapt(res))
        return "未匹配到对应 IPA 词条"

    def _original_query_path(self, user_input: str) -> str:
        parsed = parse_query(user_input)
        result = core_search(parsed)
        return format_result(result)

    def _pinyin_query_path(self, user_input: str) -> str:
        """预留拼音接口，未来直接实现"""
        return "拼音匹配功能尚未开放"

    def _adapt(self, res):
        adapted = []
        for item in res:
            adapted.append({
                FIELD_MAPPING["dialect_word"]: item.get("方言词"),
                FIELD_MAPPING["simple_pron"]: item.get("简易发音"),
                FIELD_MAPPING["standard_pron"]: item.get("标准发音"),
                FIELD_MAPPING["definition"]: item.get("释义注释"),
                "相似度": item.get("相似度", 0.0)
            })
        return adapted

# ========== main ==========
def main():
    manager = ExtensibleFusionQueryManager()

    print("="*60)
    print("        莆仙方言精准检索系统")
    print("="*60)
    print("支持查询：")
    print("1. 方言词查询")
    print("2. 普通话 / 释义查询")
    if ENABLE_IPA_MATCH and manager.ipa_enabled:
        print("3. IPA / 简易发音查询（支持精准 + 模糊）")
    print("输入 q/quit 退出\n")

    while True:
        user_input = input("请输入查询：").strip()
        if user_input.lower() in ["q", "quit"]:
            print("再见！")
            break
        if not user_input:
            print("查询不能为空，请重新输入！\n")
            continue

        try:
            formatted_result = manager.query(user_input)
            print("\n" + formatted_result + "\n")
        except Exception as e:
            print(f"出错：{e}\n")

if __name__ == "__main__":
    main()