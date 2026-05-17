"""拼音映射表构建器

功能：从规范文档和韵母对照表构建拼音↔IPA映射表
支持从Excel文件读取映射规则
"""
import pandas as pd
import os
from typing import Dict, Tuple


class PinyinMappingBuilder:
    def __init__(self):
        self.pinyin_to_ipa = {}
        self.ipa_to_pinyin = {}
    
    def load_from_excel(self, excel_path: str) -> Tuple[Dict, Dict]:
        """从Excel文件加载拼音↔IPA映射表"""
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Excel文件不存在: {excel_path}")
        
        df = pd.read_excel(excel_path, engine='openpyxl')
        
        # 假设Excel包含以下列：拼音, IPA, 说明
        if '拼音' in df.columns and 'IPA' in df.columns:
            for _, row in df.iterrows():
                pinyin = str(row['拼音']).strip()
                ipa = str(row['IPA']).strip()
                
                if pinyin and ipa and ipa != 'nan':
                    self.pinyin_to_ipa[pinyin] = ipa
                    if ipa not in self.ipa_to_pinyin:
                        self.ipa_to_pinyin[ipa] = []
                    self.ipa_to_pinyin[ipa].append(pinyin)
        
        return self.pinyin_to_ipa, self.ipa_to_pinyin
    
    def load_from_yaml(self, yaml_path: str) -> Tuple[Dict, Dict]:
        """从YAML文件加载拼音↔IPA映射表"""
        import yaml
        
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YAML文件不存在: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            mapping = yaml.safe_load(f)
        
        if 'pinyin_to_ipa' in mapping:
            self.pinyin_to_ipa.update(mapping['pinyin_to_ipa'])
        
        if 'ipa_to_pinyin' in mapping:
            self.ipa_to_pinyin.update(mapping['ipa_to_pinyin'])
        
        return self.pinyin_to_ipa, self.ipa_to_pinyin
    
    def build_default_mapping(self) -> Tuple[Dict, Dict]:
        """构建默认的拼音↔IPA映射表"""
        # 声母映射
        initials = {
            'b': 'p', 'p': 'pʰ', 'm': 'm', 'f': 'h',
            'd': 't', 't': 'tʰ', 'n': 'n', 'l': 'l',
            'g': 'k', 'k': 'kʰ', 'h': 'x',
            'j': 'tɕ', 'q': 'tɕʰ', 'x': 'ɕ',
            'z': 'ts', 'c': 'tsʰ', 's': 's',
            'zh': 'tʂ', 'ch': 'tʂʰ', 'sh': 'ʂ', 'r': 'ʐ',
            'w': 'u', 'y': 'i',
        }
        
        # 韵母映射
        finals = {
            'a': 'a', 'o': 'o', 'e': 'ɒ', 'i': 'i', 'u': 'u', 'ü': 'y', 'v': 'y',
            'ai': 'ai', 'ei': 'ui', 'ao': 'au', 'ou': 'u',
            'an': 'aŋ', 'en': 'ɒŋ', 'in': 'iŋ', 'un': 'uŋ', 'ün': 'yŋ', 'vn': 'yŋ',
            'ang': 'aŋ', 'eng': 'ɒŋ', 'ing': 'iŋ', 'ong': 'uŋ',
            'ia': 'ia', 'ie': 'ie', 'iao': 'iau', 'iu': 'iu',
            'ua': 'ua', 'uo': 'uo', 'uai': 'uai', 'ui': 'ui',
            'üe': 'ye', 've': 'ye', 'iong': 'yŋ',
            'er': 'ɒ',
        }
        
        # 合并声母和韵母
        self.pinyin_to_ipa = {**initials, **finals}
        
        # 构建反向映射
        for pinyin, ipa in self.pinyin_to_ipa.items():
            if ipa not in self.ipa_to_pinyin:
                self.ipa_to_pinyin[ipa] = []
            self.ipa_to_pinyin[ipa].append(pinyin)
        
        return self.pinyin_to_ipa, self.ipa_to_pinyin
    
    def save_to_yaml(self, yaml_path: str):
        """保存映射表到YAML文件"""
        import yaml
        
        mapping = {
            'pinyin_to_ipa': self.pinyin_to_ipa,
            'ipa_to_pinyin': self.ipa_to_pinyin
        }
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(mapping, f, allow_unicode=True, sort_keys=False)
    
    def get_mapping(self) -> Tuple[Dict, Dict]:
        """获取当前映射表"""
        return self.pinyin_to_ipa, self.ipa_to_pinyin


if __name__ == "__main__":
    builder = PinyinMappingBuilder()
    
    # 构建默认映射
    pinyin_to_ipa, ipa_to_pinyin = builder.build_default_mapping()
    print(f"构建完成：{len(pinyin_to_ipa)}条拼音→IPA映射，{len(ipa_to_pinyin)}条IPA→拼音映射")
    
    # 保存到文件
    builder.save_to_yaml('data/pinyin_mapping.yaml')
    print("映射表已保存到 data/pinyin_mapping.yaml")