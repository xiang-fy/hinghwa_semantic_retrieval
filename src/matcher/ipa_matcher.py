import re
import Levenshtein
from typing import List, Dict
from .base_matcher import BaseMatcher
from src.data_loader import get_full_df, FIELD_MAPPING

# ====================== 莆仙方言固定混淆集（人工精准定义，无任何自动生成映射） ======================
CONFUSION_MAP = {
    # 声母
    "d": ["t"],
    "t": ["d"],
    "c": ["l", "ɬ"],
    "l": ["c", "ɬ"],
    "s": ["ɬ", "l"],
    "ɬ": ["s", "l", "c"],
    "z": ["l", "t"],
    "g": ["k"],
    "k": ["g"],
    "b": ["p"],
    "p": ["b"],
    "ng": ["ŋ"],
    "n": ["ŋ", "ɬ", "l"],
    "ŋ": ["ng", "n", "l"],
    # 元音
    "o": ["ɔ", "ɒ", "ø"],
    "ɔ": ["o", "ɒ", "a"],
    "ɒ": ["o", "ɔ", "a"],
    "a": ["ɔ", "ɒ", "e"],
    "e": ["a", "ɛ", "ø"],
    "ɛ": ["e", "a"],
    "ø": ["o", "e", "oe"],
    "ou": ["ɔu", "au"],
    "au": ["ɔu", "ou"],
    # 声门塞音符号
    "ʔ": ["?", "h"],
    "?": ["ʔ", "h"],
    "h": ["ʔ", "?"],
}

# 全局声调兼容映射（固定）
TONE_MAP = {
    "1": ["1", "13", "21", "42", "453", "533", "55"],
    "2": ["2", "13", "21", "42", "453", "533", "55"],
    "3": ["3", "13", "21", "42", "453", "533", "55"],
    "4": ["4", "13", "21", "42", "453", "533", "55"],
    "5": ["5", "13", "21", "42", "453", "533", "55"],
    "6": ["6", "13", "21", "42", "453", "533", "55"],
    "7": ["7", "13", "21", "42", "453", "533", "55"],
    "8": ["8", "13", "21", "42", "453", "533", "55"],
}


def clean_ipa_str(ipa_str: str) -> str:
    """
    IPA统一清洗函数，仅清洗无关字符，绝不破坏方言IPA合法字符
    """
    if not ipa_str or not isinstance(ipa_str, str):
        return ""
    # 全角转半角
    full_to_half = str.maketrans({chr(0xFF01 + i): chr(0x21 + i) for i in range(94)})
    ipa_str = ipa_str.translate(full_to_half)
    # 小写、去除空格
    ipa_str = ipa_str.lower().replace(" ", "")
    # 剔除中文，保留所有IPA/简易发音字符
    ipa_str = re.sub(r"[\u4e00-\u9fff]+", "", ipa_str)
    return ipa_str


class IPAMatcher(BaseMatcher):
    def __init__(self, enable_rule_fuzzy: bool = True, enable_edit_fuzzy: bool = True, debug: bool = False):
        self.enable_rule_fuzzy = enable_rule_fuzzy
        self.enable_edit_fuzzy = enable_edit_fuzzy
        self.debug = debug

        # 加载数据集
        self.dialect_df = get_full_df()
        # 双索引：简易发音索引 + 标准IPA索引
        self.simple_ipa_index: Dict[str, List[Dict]] = {}
        self.standard_ipa_index: Dict[str, List[Dict]] = {}
        self.all_ipa_list: List[str] = []

        self._build_index()

    def _build_index(self):
        """构建双索引，完全对应数据集字段"""
        for _, row in self.dialect_df.iterrows():
            row_dict = row.to_dict()
            std_ipa = clean_ipa_str(str(row_dict.get(FIELD_MAPPING["standard_pron"], "")))
            simple_ipa = clean_ipa_str(str(row_dict.get(FIELD_MAPPING["simple_pron"], "")))

            # 标准IPA索引
            if std_ipa:
                if std_ipa not in self.standard_ipa_index:
                    self.standard_ipa_index[std_ipa] = []
                self.standard_ipa_index[std_ipa].append(row_dict)
                self.all_ipa_list.append(std_ipa)

            # 简易发音索引
            if simple_ipa:
                if simple_ipa not in self.simple_ipa_index:
                    self.simple_ipa_index[simple_ipa] = []
                self.simple_ipa_index[simple_ipa].append(row_dict)
                self.all_ipa_list.append(simple_ipa)

    def _precise_match(self, clean_input: str) -> List[Dict]:
        """第一层：精准匹配（简易发音优先 + 标准IPA兜底）"""
        results = []
        # 优先简易发音精准命中
        if clean_input in self.simple_ipa_index:
            for res in self.simple_ipa_index[clean_input]:
                res_copy = res.copy()
                res_copy["相似度"] = 1.0
                res_copy["匹配类型"] = "精准匹配-简易发音"
                results.append(res_copy)
        # 标准IPA精准命中
        if clean_input in self.standard_ipa_index:
            for res in self.standard_ipa_index[clean_input]:
                res_copy = res.copy()
                res_copy["相似度"] = 1.0
                res_copy["匹配类型"] = "精准匹配-标准IPA"
                results.append(res_copy)
        # 结果去重
        seen_word = set()
        unique_res = []
        for item in results:
            word = item[FIELD_MAPPING["dialect_word"]]
            if word not in seen_word:
                seen_word.add(word)
                unique_res.append(item)
        return unique_res

    def _generate_fuzzy_candidates(self, clean_input: str) -> List[str]:
        """第二层：基于固定混淆集生成模糊候选（字符替换+声调容错）"""
        candidates = {clean_input}
        # 1. 字符级替换
        temp_list = list(candidates)
        for cand in temp_list:
            for idx, char in enumerate(cand):
                if char in CONFUSION_MAP:
                    for rep_char in CONFUSION_MAP[char]:
                        new_cand = cand[:idx] + rep_char + cand[idx+1:]
                        candidates.add(new_cand)
        # 2. 音节声调拆分替换
        syllable_re = re.compile(r"([a-zɒɔøŋɬʔβ]+)([0-9]+)")
        temp_list = list(candidates)
        for cand in temp_list:
            syllables = syllable_re.findall(cand)
            if not syllables:
                continue
            tone_combine = [[]]
            for base, tone in syllables:
                new_comb = []
                tone_cand = TONE_MAP.get(tone, [tone])
                for pre in tone_combine:
                    for t in tone_cand:
                        new_comb.append(pre + [(base, t)])
                tone_combine = new_comb
            for comb in tone_combine:
                new_cand = "".join([b + t for b, t in comb])
                candidates.add(new_cand)
        return list(candidates)

    def _rule_fuzzy_match(self, clean_input: str) -> List[Dict]:
        """规则级模糊匹配"""
        if not self.enable_rule_fuzzy:
            return []
        candidates = self._generate_fuzzy_candidates(clean_input)
        results = []
        for cand in candidates:
            # 候选匹配简易发音
            if cand in self.simple_ipa_index:
                for res in self.simple_ipa_index[cand]:
                    cp = res.copy()
                    cp["相似度"] = 0.95
                    cp["匹配类型"] = "规则容错匹配"
                    cp["修正后IPA"] = cand
                    results.append(cp)
            # 候选匹配标准IPA
            if cand in self.standard_ipa_index:
                for res in self.standard_ipa_index[cand]:
                    cp = res.copy()
                    cp["相似度"] = 0.95
                    cp["匹配类型"] = "规则容错匹配"
                    cp["修正后IPA"] = cand
                    results.append(cp)
        # 去重
        seen = set()
        unique = []
        for item in results:
            w = item[FIELD_MAPPING["dialect_word"]]
            if w not in seen:
                seen.add(w)
                unique.append(item)
        return unique[:3]

    def _edit_distance_match(self, clean_input: str) -> List[Dict]:
        """第三层：编辑距离兜底模糊匹配"""
        if not self.enable_edit_fuzzy:
            return []
        in_len = len(clean_input)
        max_dis = 2 if in_len <= 6 else 3
        min_sim = 0.45
        match_list = []
        for ipa in set(self.all_ipa_list):
            dis = Levenshtein.distance(clean_input, ipa)
            if dis > max_dis:
                continue
            sim = 1 - (dis / max(in_len, len(ipa)))
            if sim < min_sim:
                continue
            match_list.append((ipa, round(sim, 3)))
        # 按相似度降序
        match_list.sort(key=lambda x: -x[1])
        results = []
        for ipa, sim in match_list[:3]:
            if ipa in self.simple_ipa_index:
                for res in self.simple_ipa_index[ipa]:
                    cp = res.copy()
                    cp["相似度"] = sim
                    cp["匹配类型"] = "模糊泛化匹配"
                    results.append(cp)
            if ipa in self.standard_ipa_index:
                for res in self.standard_ipa_index[ipa]:
                    cp = res.copy()
                    cp["相似度"] = sim
                    cp["匹配类型"] = "模糊泛化匹配"
                    results.append(cp)
        # 去重
        seen = set()
        unique = []
        for item in results:
            w = item[FIELD_MAPPING["dialect_word"]]
            if w not in seen:
                seen.add(w)
                unique.append(item)
        return unique[:3]

    def match(self, query: str, top_k: int = 3) -> List[Dict]:
        """统一对外匹配入口（继承基类接口）三层递进命中即停"""
        clean_q = clean_ipa_str(query)
        if not clean_q:
            return []
        # 1 精准匹配
        res = self._precise_match(clean_q)
        if res:
            return res[:top_k]
        # 2 规则模糊匹配
        res = self._rule_fuzzy_match(clean_q)
        if res:
            return res[:top_k]
        # 3 编辑距离兜底
        res = self._edit_distance_match(clean_q)
        return res[:top_k]