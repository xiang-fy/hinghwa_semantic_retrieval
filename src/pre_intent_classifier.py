"""前置意图分类器

功能：使用纯规则进行意图分流
支持识别：方言词查询、普通话查询、IPA查询、拼音查询
"""
import re
from typing import Dict, Optional


class PreIntentClassifier:
    """
    前置意图分类器：使用纯规则进行意图分流
    
    识别类型：
    - dialect: 方言词查询（包含方言特色字符）
    - ipa: IPA查询（包含IPA特殊字符或声调数字）
    - pinyin: 拼音查询（纯字母组合）
    - text: 文本查询（包含中文）
    - mixed: 混合查询
    """
    
    def __init__(self):
        # IPA特殊字符集合
        self.ipa_special_chars = set("ɒɔøœŋɬʔβɛɯɾʃʂʈɖʐʑ")
        
        # 方言特色字符（莆仙话常用）
        self.dialect_chars = set("𠂤𠂤𢫫𠍾")
        
        # 声母列表（用于拼音识别）
        self.initials = {'zh', 'ch', 'sh', 'b', 'p', 'm', 'f', 'd', 't', 'n', 'l',
                         'g', 'k', 'h', 'j', 'q', 'x', 'z', 'c', 's', 'r', 'w', 'y'}
        
        # 韵母列表（用于拼音识别）
        self.finals = {'ang', 'eng', 'ing', 'ong', 'ian', 'iao', 'iang', 'iong',
                       'uan', 'uang', 'ueng', 'ui', 'un', 'uo', 'ua', 'uai', 'uei',
                       'ai', 'an', 'ao', 'ou', 'ei', 'en', 'ia', 'ie', 'iu', 'o',
                       'a', 'e', 'i', 'u', 'ü', 'v', 'er', 'üe', 've'}
    
    def classify(self, user_input: str) -> Dict:
        """
        分类用户输入的意图
        
        返回格式：
        {
            "intent": "dialect" | "ipa" | "pinyin" | "text" | "mixed",
            "confidence": 0.0-1.0,
            "features": ["feature1", "feature2", ...]
        }
        """
        features = []
        has_chinese = False
        has_ipa = False
        has_pinyin = False
        has_dialect = False
        
        # 检查是否包含中文
        if re.search(r'[\u4e00-\u9fa5]', user_input):
            has_chinese = True
            features.append("has_chinese")
        
        # 检查是否包含IPA特殊字符
        for char in user_input:
            if char in self.ipa_special_chars:
                has_ipa = True
                features.append("has_ipa_special")
                break
        
        # 检查是否包含声调数字
        if re.search(r'[0-9]', user_input):
            features.append("has_tone_digit")
            # 如果没有中文但有声调数字，很可能是IPA
            if not has_chinese:
                has_ipa = True
        
        # 检查是否包含方言特色字符
        for char in user_input:
            if char in self.dialect_chars:
                has_dialect = True
                features.append("has_dialect_char")
                break
        
        # 检查是否包含拼音模式
        if self._is_pinyin_input(user_input):
            has_pinyin = True
            features.append("is_pinyin")
        
        # 判断意图类型
        intent = self._determine_intent(has_chinese, has_ipa, has_pinyin, has_dialect)
        confidence = self._calculate_confidence(intent, features)
        
        return {
            "intent": intent,
            "confidence": confidence,
            "features": features
        }
    
    def _is_pinyin_input(self, text: str) -> bool:
        """判断是否为拼音输入"""
        # 移除空格和声调数字
        normalized = re.sub(r'[\d\s]', '', text).lower()
        
        if not normalized:
            return False
        
        # 检查是否只包含字母和ü/v
        if not re.fullmatch(r'[a-züv]+', normalized):
            return False
        
        # 尝试解析为拼音音节
        i = 0
        while i < len(normalized):
            matched = False
            
            # 尝试匹配双字母声母
            if i + 1 < len(normalized):
                two_char = normalized[i:i+2]
                if two_char in self.initials:
                    remaining = normalized[i+2:]
                    for j in range(len(remaining), 0, -1):
                        if remaining[:j] in self.finals:
                            i += 2 + j
                            matched = True
                            break
            
            # 尝试匹配单字母声母或韵母
            if not matched:
                one_char = normalized[i]
                if one_char in self.initials:
                    remaining = normalized[i+1:]
                    for j in range(len(remaining), -1, -1):
                        candidate = remaining[:j]
                        if candidate == '' or candidate in self.finals:
                            i += 1 + j
                            matched = True
                            break
                elif one_char in self.finals:
                    i += 1
                    matched = True
            
            if not matched:
                return False
        
        return True
    
    def _determine_intent(self, has_chinese: bool, has_ipa: bool, 
                         has_pinyin: bool, has_dialect: bool) -> str:
        """判断意图类型"""
        # 优先级：方言特色字符 > IPA > 拼音 > 文本
        if has_dialect:
            return "dialect"
        elif has_ipa and not has_chinese:
            return "ipa"
        elif has_pinyin and not has_chinese:
            return "pinyin"
        elif has_chinese:
            return "text"
        else:
            return "text"  # 默认按文本处理
    
    def _calculate_confidence(self, intent: str, features: list) -> float:
        """计算置信度"""
        confidence = 0.5  # 基础置信度
        
        # 根据特征调整置信度
        feature_bonuses = {
            "has_dialect_char": 0.3,
            "has_ipa_special": 0.3,
            "has_tone_digit": 0.2,
            "is_pinyin": 0.3,
            "has_chinese": 0.3
        }
        
        for feature in features:
            if feature in feature_bonuses:
                confidence += feature_bonuses[feature]
        
        return min(confidence, 1.0)


if __name__ == "__main__":
    classifier = PreIntentClassifier()
    
    test_cases = [
        "郎罢",
        "tiau",
        "langba",
        "爸爸用方言怎么说",
        "莆田话的意思",
        "tɕʰiɔŋ"
    ]
    
    for test in test_cases:
        result = classifier.classify(test)
        print(f"输入: {test}")
        print(f"  意图: {result['intent']} (置信度: {result['confidence']:.2f})")
        print(f"  特征: {result['features']}")
        print()