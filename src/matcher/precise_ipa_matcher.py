import pandas as pd
import pickle
import os
from src.utils.common_utils import clean_ipa_str, format_match_result

class PreciseIPAMatcher:
    def __init__(
        self,
        data_path="data/dialect_dict.xlsx",
        map_cache_path="models/ipa_map_cache.pkl"  # 新增：映射缓存路径
    ):
        self.data_path = data_path
        self.map_cache_path = map_cache_path
        
        # 优先加载缓存，没有则重新构建
        if os.path.exists(map_cache_path):
            print("加载IPA映射缓存...")
            self._load_from_cache()
        else:
            print("重新构建IPA映射...")
            self.dialect_df = self._load_dialect_data(data_path)
            self.ipa_to_row = self._build_ipa_mapping()
            self.word_to_ipas = self._build_word_mapping()
            self._save_to_cache()  # 构建完自动保存

    def _load_dialect_data(self, data_path):
        df = pd.read_excel(data_path)
        required_fields = ["方言词", "简易发音", "标准发音", "释义注释"]
        for field in required_fields:
            if field not in df.columns:
                raise ValueError(f"Excel必须包含字段：{required_fields}")
        return df.fillna("")

    def _build_ipa_mapping(self):
        ipa_map = {}
        for _, row in self.dialect_df.iterrows():
            raw_ipa = row["标准发音"]
            clean_ipa = clean_ipa_str(raw_ipa)
            if clean_ipa:
                ipa_map[clean_ipa] = row
        print(f"IPA字典构建完成，共{len(ipa_map)}条数据")
        return ipa_map

    def _build_word_mapping(self):
        word_map = {}
        for _, row in self.dialect_df.iterrows():
            word = row["方言词"]
            ipa = clean_ipa_str(row["标准发音"])
            if word and ipa:
                if word not in word_map:
                    word_map[word] = []
                word_map[word].append(ipa)
        return word_map

    def _save_to_cache(self):
        """保存映射到缓存文件"""
        cache_data = {
            "dialect_df": self.dialect_df,
            "ipa_to_row": self.ipa_to_row,
            "word_to_ipas": self.word_to_ipas
        }
        with open(self.map_cache_path, "wb") as f:
            pickle.dump(cache_data, f)
        print(f"IPA映射已保存到：{self.map_cache_path}")

    def _load_from_cache(self):
        """从缓存文件加载映射"""
        with open(self.map_cache_path, "rb") as f:
            cache_data = pickle.load(f)
        self.dialect_df = cache_data["dialect_df"]
        self.ipa_to_row = cache_data["ipa_to_row"]
        self.word_to_ipas = cache_data["word_to_ipas"]
        print(f"IPA映射加载完成，共{len(self.ipa_to_row)}条数据")

    def debug_find_ipa_by_word(self, word):
        if word in self.word_to_ipas:
            print(f"方言词「{word}」对应的标准发音：")
            for ipa in self.word_to_ipas[word]:
                print(f"  - {repr(ipa)}")
            return self.word_to_ipas[word]
        else:
            print(f"未找到方言词「{word}」")
            return []

    def precise_ipa_match(self, ipa_str, top_k=3):
        clean_ipa = clean_ipa_str(ipa_str)
        if not clean_ipa:
            return ["未匹配到对应IPA的方言词条"]

        if clean_ipa in self.ipa_to_row:
            row = self.ipa_to_row[clean_ipa]
            return [format_match_result({
                "方言词": row["方言词"],
                "简易发音": row["简易发音"],
                "标准发音": row["标准发音"],
                "释义注释": row["释义注释"],
                "score": 1.0
            })]
        return ["未匹配到对应IPA的方言词条"]