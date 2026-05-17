"""拼音处理核心模块

功能：
1. 拼音切分（支持带声调/不带声调，有空格/无空格）
2. 莆仙话拼音 → IPA 映射（权重 1.0）
3. 普通话拼音 → IPA 映射（权重 0.8，基于实际音系对应）
4. 候选生成与组合
5. pinyin_search 入口
"""
import re
from typing import List, Dict, Tuple, Optional
from itertools import product

# ============================================================
# 莆仙话拼音到 IPA 的映射（基于莆仙话拼音方案）
# ============================================================
PUTIAN_INITIALS = {
    'b': 'p', 'p': 'pʰ', 'bb': 'b',
    'd': 't', 't': 'tʰ', 'dd': 'd',
    'g': 'k', 'k': 'kʰ', 'gg': 'g',
    'z': 'ts', 'c': 'tsʰ', 'zz': 'dz',
    's': 'ɬ',
    'l': 'l',
    'm': 'm', 'n': 'n', 'ng': 'ŋ',
    'h': 'h',
    '': ''
}

PUTIAN_FINALS = {
    'a': ['a'], 'e': ['e'], 'i': ['i'], 'o': ['o', 'ɵ'], 'u': ['u'], 'y': ['y'],
    'ae': ['ɛ'], 'oe': ['ø', 'œ'], 'or': ['ɒ'], 'er': ['ɤ', 'ə'], 'oo': ['ɔ'],
    'ai': ['ai'], 'au': ['au'], 'ia': ['ia'], 'ie': ['iɛ'], 'ieo': ['iɛu'],
    'iu': ['iu'], 'io': ['io'], 'ua': ['ua'], 'uo': ['uɔ'], 'ui': ['ui'],
    'ue': ['uɛ'], 'uai': ['uai'], 'uei': ['uei'], 'yo': ['yɒ'], 'ye': ['yɛ'],
    'yoe': ['yœ'],
    'ann': ['ã'], 'enn': ['ẽ'], 'inn': ['ĩ'], 'onn': ['õ'], 'unn': ['ũ'],
    'iann': ['iã'], 'iunn': ['iũ'], 'uann': ['uã'], 'uinn': ['uĩ'],
    'ah': ['aʔ'], 'eh': ['ɛʔ'], 'ih': ['iʔ'], 'oh': ['oʔ', 'ɵʔ'],
    'uh': ['uʔ'], 'yh': ['yʔ'], 'oeh': ['œʔ'], 'orh': ['ɒʔ'],
}

# ============================================================
# 普通话拼音 → 莆仙 IPA 映射（优化版，基于实际音系对应）
# ============================================================
# 声母映射
MANDARIN_INITIALS = {
    'b': 'p', 'p': 'pʰ', 'm': 'm', 'f': 'h',
    'd': 't', 't': 'tʰ', 'n': 'n', 'l': 'l',
    'g': 'k', 'k': 'kʰ', 'h': 'h',
    'j': 'ts', 'q': 'tsʰ', 'x': 'ɬ',
    'z': 'ts', 'c': 'tsʰ', 's': 'ɬ',
    'zh': 'ts', 'ch': 'tsʰ', 'sh': 'ɬ', 'r': 'l',
    'y': '', 'w': '',
    '': ''
}

# 韵母映射（只保留最可能的 1-2 个 IPA，去掉不准确的候选）
MANDARIN_FINALS = {
    'a': ['a'], 'o': ['ɔ'], 'e': ['ɛ'], 'i': ['i'], 'u': ['u'], 'ü': ['y'],
    'ai': ['ai'], 'ei': ['ei'], 'ao': ['au'], 'ou': ['ɔu'],
    'an': ['aŋ'], 'en': ['ɛŋ'], 'in': ['iŋ'], 'un': ['uŋ'], 'ün': ['yŋ'],
    'ang': ['aŋ'], 'eng': ['ɛŋ'], 'ing': ['iŋ'], 'ong': ['uŋ'],
    'ia': ['ia'], 'ie': ['iɛ'], 'iao': ['iau'], 'iu': ['iu'],
    'ian': ['iɛŋ'], 'iang': ['iaŋ'], 'iong': ['yŋ'],
    'ua': ['ua'], 'uo': ['uɔ'], 'uai': ['uai'], 'ui': ['ui'],
    'uan': ['uaŋ'], 'uang': ['uaŋ'],
    'üe': ['yœ'],
    'er': ['ə']
}

# 常用短语直接映射（避免音节切分错误）
MANDARIN_PHRASE_MAP = {
    'liuyuetian': 'lah7 gue3 ling1',
    'doufu': 'dau5 hu2',
    'ajie': 'a1 i2',
    'baba': 'a1 ba5',
    'yue': 'gueh6',
    'tian': 'ling1'
}

# ============================================================
# 声调映射（仅用于生成 IPA 调值，实际普通话查询时会忽略）
# ============================================================
TONE_MAP = {
    '1': ['1', '533', '55'],
    '2': ['2', '13', '24', '35'],
    '3': ['3', '453', '332'],
    '4': ['4', '42'],
    '5': ['5', '21', '11'],
    '6': ['6', '21', '1'],
    '7': ['7', '4', '24'],
    '8': ['8', '35'],
}

# ============================================================
# 拼音切分函数（核心）
# ============================================================
def split_putian_syllable(syl: str) -> Tuple[str, str]:
    """拆分莆仙拼音音节为声母和韵母"""
    if len(syl) >= 2 and syl[:2] in ['bb', 'dd', 'gg', 'zz', 'ng']:
        return syl[:2], syl[2:]
    if syl and syl[0] in 'bpmfdtnlgkhjqxzcszhchshryw':
        return syl[0], syl[1:]
    return '', syl

def split_mandarin_syllable(syl: str) -> Tuple[str, str]:
    """拆分普通话音节为声母和韵母"""
    if len(syl) >= 2 and syl[:2] in ['zh', 'ch', 'sh']:
        return syl[:2], syl[2:]
    if syl.startswith('yi'):
        return '', 'i'
    if syl.startswith('wu'):
        return '', 'u'
    if syl.startswith('yu'):
        return '', 'ü'
    if syl and syl[0] in 'bpmfdtnlgkhjqxzcszhchshryw':
        return syl[0], syl[1:]
    return '', syl

def parse_pinyin(pinyin_str: str) -> List[Dict]:
    """
    解析拼音字符串，返回音节列表
    每个音节格式：{'initial': str, 'final': str, 'tone': str}
    支持：带空格/无空格，带声调数字/不带声调
    """
    pinyin_str = pinyin_str.lower().strip()
    if not pinyin_str:
        return []
    parts = pinyin_str.split()
    if not parts:
        parts = [pinyin_str]

    # 声母和韵母词典（用于无空格切分）
    initials = ['zh','ch','sh','bb','dd','gg','zz','ng',
                'b','p','m','f','d','t','n','l',
                'g','k','h','j','q','x','z','c','s','r','w','y']
    finals = list(PUTIAN_FINALS.keys()) + list(MANDARIN_FINALS.keys())
    finals = sorted(set(finals), key=len, reverse=True)

    syllables = []
    for part in parts:
        i = 0
        while i < len(part):
            matched = False
            # 匹配声母（优先长声母）
            for init in sorted(initials, key=len, reverse=True):
                if part[i:].startswith(init):
                    remaining = part[i+len(init):]
                    for final in finals:
                        flen = len(final)
                        if len(remaining) >= flen:
                            candidate = remaining[:flen]
                            tone = ''
                            if len(remaining) > flen and remaining[flen].isdigit():
                                tone = remaining[flen]
                                candidate = remaining[:flen]
                            if candidate == final:
                                syllables.append({
                                    'initial': init,
                                    'final': final,
                                    'tone': tone,
                                    'source': 'unknown'
                                })
                                i += len(init) + len(candidate) + (1 if tone else 0)
                                matched = True
                                break
                    if matched:
                        break
            if not matched:
                # 尝试直接匹配韵母（零声母）
                for final in finals:
                    flen = len(final)
                    if len(part)-i >= flen:
                        candidate = part[i:i+flen]
                        tone = ''
                        if i+flen < len(part) and part[i+flen].isdigit():
                            tone = part[i+flen]
                        if candidate == final:
                            syllables.append({
                                'initial': '',
                                'final': final,
                                'tone': tone,
                                'source': 'unknown'
                            })
                            i += len(candidate) + (1 if tone else 0)
                            matched = True
                            break
                if not matched:
                    # 未知字符，跳过
                    i += 1
    return syllables

# ============================================================
# 莆仙拼音 → IPA 候选（权重 1.0）
# ============================================================
def putian_syllable_to_ipa(syl: str, tone: Optional[str] = None) -> List[str]:
    """莆仙拼音音节转 IPA（不带声调数字）"""
    initial, final = split_putian_syllable(syl)
    ipa_init = PUTIAN_INITIALS.get(initial, '')
    ipa_finals = PUTIAN_FINALS.get(final, [final])
    candidates = []
    for ipa_final in ipa_finals:
        if ipa_init:
            candidates.append(ipa_init + ipa_final)
        else:
            candidates.append(ipa_final)
    return list(set(candidates))

def putian_pinyin_to_ipa_candidates(pinyin_str: str) -> List[Tuple[str, float]]:
    """莆仙拼音字符串 → IPA候选列表（带声调数字，权重1.0）"""
    syllables = parse_pinyin(pinyin_str)
    if not syllables:
        return []
    candidates_per_syl = []
    for syl in syllables:
        ipa_segs = putian_syllable_to_ipa(syl['initial'] + syl['final'], syl['tone'])
        tone = syl['tone']
        if tone:
            tone_cands = TONE_MAP.get(tone, [tone])
        else:
            tone_cands = ['']
        segs = []
        for seg in ipa_segs:
            for tc in tone_cands:
                ipa_with_tone = seg + tc
                segs.append((ipa_with_tone, 1.0))
        candidates_per_syl.append(segs)
    # 笛卡尔积
    results = []
    for combo in product(*candidates_per_syl):
        ipa_full = ''.join(c[0] for c in combo)
        weight = sum(c[1] for c in combo) / len(combo)
        results.append((ipa_full, weight))
    results = list(set(results))
    results.sort(key=lambda x: -x[1])
    return results[:30]

# ============================================================
# 普通话拼音 → IPA 候选（优化版，权重 0.8，忽略声调）
# ============================================================
def mandarin_syllable_to_ipa(syl: str) -> List[str]:
    """普通话音节转 IPA（不带声调数字）"""
    initial, final = split_mandarin_syllable(syl)
    ipa_init = MANDARIN_INITIALS.get(initial, '')
    ipa_finals = MANDARIN_FINALS.get(final, [final])
    candidates = []
    for ipa_final in ipa_finals:
        if ipa_init:
            candidates.append(ipa_init + ipa_final)
        else:
            candidates.append(ipa_final)
    # 去重，只保留前 2 个最高概率的
    candidates = list(set(candidates))[:2]
    return candidates

def mandarin_pinyin_to_ipa_candidates(pinyin_str: str) -> List[Tuple[str, float]]:
    """普通话拼音 → IPA候选（无调，权重0.8）"""
    # 优先检查短语映射
    norm = pinyin_str.lower().replace(" ", "")
    if norm in MANDARIN_PHRASE_MAP:
        phrase_pinyin = MANDARIN_PHRASE_MAP[norm]
        # 直接使用莆仙转换，权重设为 0.8
        candidates = putian_pinyin_to_ipa_candidates(phrase_pinyin)
        return [(ipa, 0.8) for ipa, _ in candidates] if candidates else []

    syllables = parse_pinyin(pinyin_str)
    if not syllables:
        return []
    candidates_per_syl = []
    for syl in syllables:
        ipa_segs = mandarin_syllable_to_ipa(syl['initial'] + syl['final'])
        # 坚决不加声调数字
        segs = [(seg, 0.8) for seg in ipa_segs]
        if not segs:
            # 兜底
            segs = [(syl['initial'] + syl['final'], 0.3)]
        candidates_per_syl.append(segs)
    results = []
    for combo in product(*candidates_per_syl):
        ipa_full = ''.join(c[0] for c in combo)
        weight = sum(c[1] for c in combo) / len(combo)
        results.append((ipa_full, weight))
    results = list(set(results))
    results.sort(key=lambda x: -x[1])
    return results[:30]



# ============================================================
# 统一搜索入口
# ============================================================
def pinyin_search(query: str, ipa_matcher, top_k: int = 5) -> List[Dict]:
    """
    拼音查询主入口
    :param query: 拼音字符串，如 "a1 ma3" 或 "a ma"
    :param ipa_matcher: IPAMatcher 实例
    :param top_k: 返回结果数量
    """
    query = query.strip()
    if not query:
        return []
    # 生成两种候选（莆仙和普通话）
    putian_candidates = putian_pinyin_to_ipa_candidates(query)
    mandarin_candidates = mandarin_pinyin_to_ipa_candidates(query)
    all_candidates = putian_candidates + mandarin_candidates
    # 去重
    seen = set()
    unique = []
    for ipa, w in all_candidates:
        if ipa not in seen:
            seen.add(ipa)
            unique.append((ipa, w))
    # 调用 IPA 匹配器
    all_results = []
    for ipa_str, weight in unique[:30]:
        matches = ipa_matcher.match(ipa_str, top_k=3)
        for m in matches:
            m['pinyin_weight'] = weight
            all_results.append(m)
    # 去重并排序
    from src.data_loader import FIELD_MAPPING
    unique_words = {}
    for r in all_results:
        word = r[FIELD_MAPPING["dialect_word"]]
        if word not in unique_words or r.get("相似度", 0) > unique_words[word].get("相似度", 0):
            unique_words[word] = r
    results = list(unique_words.values())
    results.sort(key=lambda x: x.get("相似度", 0) * 0.7 + x.get("pinyin_weight", 0) * 0.3, reverse=True)
    return results[:top_k]


def mixed_pinyin_to_ipa_candidates(pinyin_str: str) -> List[Tuple[str, float]]:
    """按音节混合生成 IPA 候选：每个音节同时考虑莆仙和普通话映射，支持混合输入。

    返回列表形式为 (ipa_str, weight)。
    """
    syllables = parse_pinyin(pinyin_str)
    if not syllables:
        return []

    candidates_per_syl = []
    for syl in syllables:
        key = syl['initial'] + syl['final']
        tone = syl.get('tone', '')

        segs = []
        # 莆仙候选（权重 1.0）
        try:
            putian_segs = putian_syllable_to_ipa(key, tone)
        except Exception:
            putian_segs = []
        if putian_segs:
            tone_cands = TONE_MAP.get(tone, ['']) if tone else ['']
            for ps in set(putian_segs):
                for tc in tone_cands:
                    segs.append((ps + tc, 1.0))

        # 普通话候选（权重 0.8）
        try:
            mandarin_segs = mandarin_syllable_to_ipa(key)
        except Exception:
            mandarin_segs = []
        for ms in set(mandarin_segs):
            segs.append((ms, 0.8))

        # 兜底：使用原始拼音片段（低权重）
        if not segs:
            segs = [(key, 0.3)]

        candidates_per_syl.append(segs)

    # 笛卡尔积组合
    results = []
    for combo in product(*candidates_per_syl):
        ipa_full = ''.join([c[0] for c in combo])
        weight = sum(c[1] for c in combo) / len(combo)
        results.append((ipa_full, weight))

    # 去重并排序
    results = list({(r[0], r[1]) for r in results})
    results.sort(key=lambda x: -x[1])
    return results[:50]

# ============================================================
# 兼容旧接口（供 pinyin_matcher.py 使用）
# ============================================================
def generate_candidates(query: str) -> List[str]:
    """生成候选拼音串（用于简单发音索引）"""
    q = (query or '').lower().strip()
    if not q:
        return []
    q_nospace = q.replace(' ', '')
    q_nodigits = re.sub(r"\d", "", q_nospace)
    candidates = [q_nospace]
    if q_nodigits != q_nospace:
        candidates.append(q_nodigits)
    # 基于 parse_pinyin 重建
    try:
        syls = parse_pinyin(q)
        if syls:
            rebuilt = ''.join([s['initial'] + s['final'] for s in syls])
            if rebuilt and rebuilt not in candidates:
                candidates.append(rebuilt)
    except:
        pass
    seen = set()
    out = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out

def pinyin_to_ipa(query: str) -> str:
    """将拼音串转换为单一 IPA（选择权重最高的候选）"""
    if not query:
        return ''
    # 优先使用按音节混合生成的候选（支持莆仙+普通话混合输入）
    try:
        mixed = mixed_pinyin_to_ipa_candidates(query)
        if mixed:
            return max(mixed, key=lambda x: x[1])[0]
    except NameError:
        # 如果混合函数不存在（兼容旧版），回退到简单合并
        pass

    putian = putian_pinyin_to_ipa_candidates(query)
    mandarin = mandarin_pinyin_to_ipa_candidates(query)
    allc = putian + mandarin
    if not allc:
        return ''
    best = max(allc, key=lambda x: x[1])[0]
    return best

# ============================================================
# 测试代码（仅在本模块独立运行时执行）
# ============================================================
if __name__ == "__main__":
    test_queries = [
        "a1 ma3", "ah6 sau1", "lor2 ba5", "ba1 de3 eng1",
        "a ma", "a yi", "a bo", "hong tuan", "dou fu", "liu yue tian",
        "a1 ma", "a ma3", "ang2 tuan", "ama", "taufu", "lah7gue3ling1", "a1ma3", "lor2ba5"
    ]
    print("拼音搜索测试用例列表：")
    for q in test_queries:
        print(f"  {q}")