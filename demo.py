from src.query_rewriter import parse_query
from src.vector_db import core_search
from src.result_formatter import format_result
# ========== 新增：IPA匹配相关导入 ==========
from src.matcher.precise_ipa_matcher import PreciseIPAMatcher
from src.utils.common_utils import clean_ipa_str
from src.data_loader import FIELD_MAPPING
from typing import List, Dict, Optional, Set
import re

# 屏蔽 sentence-transformers / transformers 非项目代码本身的冗余日志
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

# ========== 新增：模块化意图提取器（未来扩展用） ==========
class IntentExtractor:
    """
    模块化意图提取器：未来模糊/拼音匹配用，预留接口
    用于从混合输入中提取核心IPA/拼音内容
    """
    @staticmethod
    def extract_ipa(user_input: str) -> Optional[str]:
        """从混合输入中提取核心IPA内容（未来实现）"""
        return None

    @staticmethod
    def extract_pinyin(user_input: str) -> Optional[str]:
        """从混合输入中提取核心拼音内容（未来实现）"""
        return None

# ========== 【动态 IPA 识别器：自适应数据集】==========
class DynamicIPARecognizer:
    def __init__(self, valid_ipa_chars: set):
        self.valid_ipa_chars = valid_ipa_chars  # 从数据集自动提取
        # 基础允许字符（字母、数字、空格）
        self.basic_chars = set("abcdefghijklmnopqrstuvwxyz0123456789 ")

    def is_ipa_input(self, user_input: str) -> bool:
        """
        【真正工程化】
        自动判断是否为 IPA：无中文 + 全部字符都在数据集真实出现过的字符里
        换任何数据集都不用改代码！
        """
        s = user_input.strip()
        if not s:
            return False

        # 1. 包含中文 → 绝对不是 IPA
        if re.search(r"[\u4e00-\u9fa5]", s):
            return False

        # 2. 所有字符必须是：基础字符 或 数据集中真实存在的IPA符号
        for c in s:
            if c not in self.basic_chars and c not in self.valid_ipa_chars:
                return False

        # 3. 必须至少包含一个非字母数字的 IPA 符号
        return any(c in self.valid_ipa_chars for c in s)

# ========== 【缓存工具类：自动加载 / 保存 / 更新】==========
class IPACharCache:
    def __init__(self, ipa_dict_keys, dialect_df):
        self.ipa_dict_keys = ipa_dict_keys
        self.dialect_df = dialect_df

    def _get_dataset_signature(self) -> str:
        """生成数据集唯一指纹，换数据集自动识别"""
        try:
            ipa_series = self.dialect_df["标准发音"].dropna().astype(str)
            content = "|".join(ipa_series.tolist())
            return hashlib.md5(content.encode("utf-8")).hexdigest()
        except:
            return "unknown"

    def load(self) -> Optional[Set[str]]:
        os.makedirs(CACHE_DIR, exist_ok=True)
        if not os.path.exists(IPA_CHAR_CACHE) or not os.path.exists(DATASET_SIGNATURE):
            return None

        # 检查数据集是否变化
        with open(DATASET_SIGNATURE, "r", encoding="utf-8") as f:
            cached_sig = f.read().strip()
        if cached_sig != self._get_dataset_signature():
            return None

        # 加载缓存
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
            # print(f"[缓存] 加载 IPA 字符集，共 {len(cached)} 个符号")
            return cached

        # print("[缓存] 首次运行/数据集更新，重新提取 IPA 字符...")
        chars = set()
        for ipa in self.ipa_dict_keys:
            for c in ipa:
                chars.add(c)
        self.save(chars)
        # print(f"[缓存] 提取完成，已保存 {len(chars)} 个符号")
        return chars

# ========== 可扩展查询管理器 ==========
class ExtensibleFusionQueryManager:
    def __init__(self):
        self.original_enabled = True
        self.ipa_enabled = False
        self.ipa_recognizer = None

        if ENABLE_IPA_MATCH:
            try:
                # 加载 IPA 匹配器
                self.ipa_matcher = PreciseIPAMatcher()

                # ==============================================
                # 核心：带缓存的动态 IPA 字符集
                # ==============================================
                cache = IPACharCache(
                    ipa_dict_keys=self.ipa_matcher.ipa_to_row.keys(),
                    dialect_df=self.ipa_matcher.dialect_df
                )
                self.valid_ipa_chars = cache.get_chars()

                self.ipa_recognizer = DynamicIPARecognizer(self.valid_ipa_chars)
                self.ipa_enabled = True
            except Exception as e:
                print(f"IPA 模块加载失败：{e}")

    def _extract_all_ipa_chars(self) -> set:
        """保留你原有接口，实际已走缓存"""
        return self.valid_ipa_chars

    def query(self, user_input: str) -> str:
        if self.ipa_enabled and self.ipa_recognizer.is_ipa_input(user_input):
            return self._ipa_query_path(user_input)
        else:
            return self._original_query_path(user_input)

    def _ipa_query_path(self, user_input: str) -> str:
        """IPA查询路径：当前精准匹配，未来可扩展模糊匹配"""
        # 未来灵活模式下，可先调用IntentExtractor.extract_ipa提取核心IPA
        # clean = clean_ipa_str(user_input)
        res = self.ipa_matcher.precise_ipa_match(user_input)
        if res:
            return format_result(self._adapt(res))
        return "未匹配到对应 IPA 词条"

    def _original_query_path(self, user_input: str) -> str:
        parsed = parse_query(user_input)
        result = core_search(parsed)
        return format_result(result)

    def _pinyin_query_path(self, user_input: str) -> str:
        """拼音查询路径：预留实现，未来可扩展"""
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
    # 初始化可扩展融合查询管理器
    manager = ExtensibleFusionQueryManager()
    
    # 原有标题保留，仅新增IPA查询说明（仅当开启时显示）
    print("="*60)
    print("        莆仙方言精准检索系统")
    print("="*60)
    print("支持查询：")
    print("1. 方言查方言（如：漉、𢫫裤）")
    print("2. 普通话/释义查方言（如：爸爸、踩水）")
    if ENABLE_IPA_MATCH and manager.ipa_enabled:
        print("3. 标准IPA查方言（如：lɒʔ1）")
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
            continue

if __name__ == "__main__":
    main()