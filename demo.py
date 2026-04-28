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

# ========== 动态 IPA 识别器==========
class DynamicIPARecognizer:
    def __init__(self, valid_ipa_chars: set, known_ipa_forms: Optional[Set[str]] = None):
        self.valid_ipa_chars = valid_ipa_chars
        self.known_ipa_forms = known_ipa_forms or set()
        self.basic_chars = set("abcdefghijklmnopqrstuvwxyz0123456789 ")
        self.ipa_special_chars = set("ɒɔøœŋɬʔβɛɯɾʃʂʈɖʐʑŋɱɳ")

    def _strip_phonetic_marks(self, text: str) -> str:
        return re.sub(r"[\d\s'’`]+", "", text.strip().lower())

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

        normalized = self._strip_phonetic_marks(s)

        # 已知 IPA 词形优先命中，避免与拼音路由冲突
        if normalized in self.known_ipa_forms:
            return True

        # 含声调数字或明确 IPA 特征符号时，直接走 IPA
        if any(c.isdigit() for c in s):
            return True

        if any(c in self.ipa_special_chars for c in s):
            return True

        return False


class DynamicPinyinRecognizer:
    def __init__(self):
        self.initials = (
            "zh", "ch", "sh",
            "b", "p", "m", "f", "d", "t", "n", "l",
            "g", "k", "h", "j", "q", "x",
            "r", "z", "c", "s", "y", "w",
        )
        self.finals = (
            "iang", "iong", "uang", "ueng",
            "iao", "ian", "ing", "uai", "uan", "uen",
            "ong", "ang", "eng", "ai", "ei", "ao", "ou", "an", "en", "ia", "ie", "iu", "ua", "uo", "ui", "un", "ve", "üe",
            "a", "o", "e", "i", "u", "v", "ü", "er", "ê",
        )
        self._finals_sorted = sorted(set(self.finals), key=len, reverse=True)
        self._initials_sorted = sorted(set(self.initials), key=len, reverse=True)

    def _normalize(self, text: str) -> str:
        return re.sub(r"[\d\s'’`]+", "", text.strip().lower())

    def _is_valid_syllable(self, syllable: str) -> bool:
        if not syllable:
            return False
        for initial in self._initials_sorted:
            if syllable.startswith(initial):
                tail = syllable[len(initial):]
                return tail in self._finals_sorted
        return syllable in self._finals_sorted

    def is_pinyin_input(self, user_input: str) -> bool:
        s = self._normalize(user_input)
        if not s:
            return False

        if re.search(r"[\u4e00-\u9fa5]", s):
            return False
        if re.search(r"[ɒɔøœŋɬʔβɛɯɾʃʂʈɖʐʑ]", s):
            return False
        if not re.fullmatch(r"[a-züv]+", s):
            return False

        memo = {}

        def can_parse(start: int) -> bool:
            if start == len(s):
                return True
            if start in memo:
                return memo[start]

            for end in range(min(len(s), start + 6), start, -1):
                chunk = s[start:end]
                if self._is_valid_syllable(chunk) and can_parse(end):
                    memo[start] = True
                    return True

            memo[start] = False
            return False

        return can_parse(0)

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

# ========== 可扩展融合查询管理器==========
class ExtensibleFusionQueryManager:
    def __init__(self):
        self.original_enabled = True
        self.ipa_enabled = ENABLE_IPA_MATCH
        self.ipa_recognizer = None
        self.pinyin_recognizer = DynamicPinyinRecognizer()
        self.known_ipa_forms = set()

        if self.ipa_enabled:
            try:
                # ==============================================
                # 【关键】使用新架构的 MatcherManager
                # ==============================================
                self.ipa_matcher = matcher_manager

                # 从新 IPA 匹配器获取所有 IPA 列表
                all_ipa = matcher_manager.ipa_matcher.all_ipa_list
                tone_free_ipa = getattr(matcher_manager.ipa_matcher, "all_tone_free_ipa_list", [])
                self.known_ipa_forms = {self._strip_route_marks(item) for item in all_ipa + tone_free_ipa}
                cache = IPACharCache(all_ipa)
                self.valid_ipa_chars = cache.get_chars()
                self.ipa_recognizer = DynamicIPARecognizer(self.valid_ipa_chars, self.known_ipa_forms)

            except Exception as e:
                print(f"IPA 模块加载失败：{e}")
                self.ipa_enabled = False

    @staticmethod
    def _strip_route_marks(text: str) -> str:
        return re.sub(r"[\d\s'’`]+", "", str(text).strip().lower())

    def _route_query(self, user_input: str) -> str:
        if re.search(r"[\u4e00-\u9fa5]", user_input):
            return "original"

        if self.ipa_enabled and self.ipa_recognizer and self.ipa_recognizer.is_ipa_input(user_input):
            return "ipa"

        if self.pinyin_recognizer.is_pinyin_input(user_input):
            return "pinyin"

        return "original"

    def query(self, user_input: str) -> str:
        route = self._route_query(user_input)

        if route == "ipa":
            return self._ipa_query_path(user_input)
        if route == "pinyin":
            return self._pinyin_query_path(user_input)
        return self._original_query_path(user_input)

    def _ipa_query_path(self, user_input: str) -> str:
        """
        新版 IPA 查询：
        自动支持 → 简易发音精准 + 标准IPA精准 + 模糊匹配
        """
        res = self.ipa_matcher.ipa_query(user_input, top_k=3)
        if res:
            return format_result(self._adapt(res))
        return "未匹配到对应 IPA 词条"

    def _original_query_path(self, user_input: str) -> str:
        parsed = parse_query(user_input)
        result = core_search(parsed)
        return format_result(result)

    def _pinyin_query_path(self, user_input: str) -> str:
        """预留拼音接口，当前仅占位，不影响主分流。"""
        return "拼音匹配功能已预留，当前尚未开放"

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
        print("3. IPA查询（精准 + 模糊）")
    # print("4. 拼音查询（路由已预留）")
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