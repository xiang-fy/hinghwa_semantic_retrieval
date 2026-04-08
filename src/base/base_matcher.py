from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseIPAMatcher(ABC):
    """
    IPA匹配器抽象基类：定义统一接口，预留模糊IPA/拼音匹配扩展点
    核心：目前强制实现精准匹配，模糊匹配/拼音匹配暂留空（todo）
    """
    @abstractmethod
    def precise_ipa_match(self, ipa_str: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """精准IPA匹配（核心实现）：输入标准IPA，返回结构化方言词条"""
        pass

    @abstractmethod
    def fuzzy_ipa_match(self, ipa_str: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """预留：模糊IPA匹配接口（后续实现）"""
        pass

    @abstractmethod
    def pinyin_match(self, pinyin_str: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """预留：拼音匹配接口（后续实现）"""
        pass