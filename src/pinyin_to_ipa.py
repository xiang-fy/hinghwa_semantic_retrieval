"""拼音处理核心模块

功能：
1. 拼音切分（支持带声调/不带声调，有空格/无空格）
2. 莆仙话拼音 → IPA 映射
3. 普通话拼音 → IPA 映射
4. 候选生成与组合
5. pinyin_search 入口
"""
import re
from typing import List, Dict, Tuple, Optional
from itertools import product

# ============================================================
# 一、莆仙话拼音到 IPA 的映射（基于莆仙话拼音方案 + 韵母对照表）
# ============================================================

# 声母映射（莆拼 → IPA，参考规范文档）
# 注意：莆仙话拼音即为数据集中的简易发音字段
PUTIAN_INITIALS = {
    'b': 'p', 'p': 'pʰ', 'bb': 'b',
    'd': 't', 't': 'tʰ', 'dd': 'd',
    'g': 'k', 'k': 'kʰ', 'gg': 'g',
    'z': 'ts', 'c': 'tsʰ', 'zz': 'dz',
    's': 'ɬ',          # 莆拼 s 对应 IPA ɬ（边擦音）或 θ（齿擦音），本系统统一用 ɬ
    'l': 'l',
    'm': 'm', 'n': 'n', 'ng': 'ŋ',
    'h': 'h',          # 主流口音 h 读 h
    'j': 'ts', 'q': 'tsʰ', 'x': 'ɬ',
    'r': 'l',          # 部分口音 r 读 l
    '': ''
}

# 声母类化规则（连读时的声母变化）
INITIAL_SANDHI = {
    'p': ['p', 'b', 'm', ''],   # p→b/m/消失（类化）
    't': ['t', 'd', 'n', 'l', ''],  # t→d/n/l/消失
    'k': ['k', 'g', 'ŋ', ''],   # k→g/ŋ/消失
    'ts': ['ts', 'dz', 'l', 's'],  # ts→dz/l/s
    'tsʰ': ['tsʰ', 'dz', 'l', 's'],
    'ɬ': ['ɬ', 'l', 's'],       # ɬ→l/s
    'm': ['m', 'b', ''],         # m→b/消失
    'n': ['n', 'd', 'l', ''],    # n→d/l/消失
    'ŋ': ['ŋ', 'g', ''],         # ŋ→g/消失
}

# 韵母映射（莆拼 → IPA 候选，支持多候选）
# 依据莆拼规范文档及韵母对照表整理
# 莆仙话拼音即为数据集中的简易发音字段
PUTIAN_FINALS = {
    # 单韵母
    'a': ['a'],
    'e': ['e'],
    'i': ['i'],
    'o': ['o'],            # 标准发音
    'u': ['u'],
    'y': ['y'],
    'ae': ['ɛ'],           # 用于少数词
    'oe': ['ø'],
    'or': ['ɒ'],           # 莆田城里腔
    'er': ['ə'],
    'oo': ['ɔ'],
    # 复韵母
    'ai': ['ai'],
    'au': ['au'],
    'ao': ['au'],          # ao 与 au 同音
    'ia': ['ia'],
    'ie': ['ie'],
    'ieo': ['ieu'],
    'iu': ['iu'],
    'io': ['io'],
    'ua': ['ua'],
    'uo': ['uo'],
    'ui': ['ui'],
    'ue': ['ue'],
    'uai': ['uai'],
    'uei': ['uei'],
    'yo': ['yo'],
    'ye': ['ye'],
    'yoe': ['yø'],
    # 鼻化韵（带鼻音的韵母）
    'ann': ['aŋ'],
    'enn': ['eŋ'],
    'inn': ['iŋ'],
    'onn': ['oŋ'],
    'unn': ['uŋ'],
    'iann': ['iaŋ'],
    'iunn': ['iuŋ'],
    'uann': ['uaŋ'],
    'uinn': ['uiŋ'],
    # 入声韵（带喉塞音结尾）
    'ah': ['aʔ'],
    'eh': ['eʔ'],
    'ih': ['iʔ'],
    'oh': ['oʔ'],
    'uh': ['uʔ'],
    'yh': ['yʔ'],
    'oeh': ['øʔ'],
    'orh': ['ɒʔ'],
}

# 口音差异映射（基于興化各地口音對照表）
ACCENT_VARIANTS = {
    '仙游': {
        'o': ['ɵ'],          # 仙游 o 偏 ɵ
        'ɬ': ['θ'],          # 仙游部分区域 ɬ→θ
    },
    '涵江': {
        'ŋ': ['n'],          # 涵江部分 ng→n
    },
    '城厢': {
        'or': ['ɒ'],         # 城厢 or 读 ɒ
    },
}

# 各地口音差异映射（基于興化各地口音對照表）
# 键为口音名称，值为 (原IPA, 替换后IPA) 的映射列表
ACCENT_IPA_MAP = {
    '仙游': [
        ('ɬ', 'θ'),      # 仙游部分区域 ɬ → θ
        ('o', 'ɵ'),      # 韵母 o 偏 ɵ
        ('iŋ', 'iŋ'),    # 无变化，但可留作扩展
    ],
    '江口': [
        ('aŋ', 'aŋ'),    # 江口部分字读 aŋ 为 aŋ（实际同）
        ('u', 'u'),      # 无显著差异
    ],
    '涵江': [
        ('ŋ', 'n'),      # 部分 ng 声母读 n
    ],
}
# 实际使用时，可根据需要从口音对照表动态读取，此处简化

# ============================================================
# 二、普通话发音描述 → 莆仙话发音映射（核心修正）
# 目的：用户用普通话拼音描述方言词的发音，系统转换为莆仙话IPA
# 例如：用户说"doufu"描述豆腐的发音 → 系统理解为莆仙话"tauhu"
# 参考：聲母類化規律及興化各地口音對照表
# ============================================================

# 声母映射：普通话发音 → 莆仙话发音
# 依据实际音系对应，如：清 qing4 → 莆仙话 cing4 → IPA tshiŋ42
MANDARIN_TO_PUTIAN_INITIALS = {
    'b': ['p'],                # 波 → 坡
    'p': ['pʰ', 'p'],          # 坡 → 坡/波
    'm': ['m'],
    'f': ['h'],                # 佛 → 霍（莆仙话无 f）
    'd': ['t'],
    't': ['tʰ', 't'],
    'n': ['n', 'l'],           # 部分口音 n/l 不分
    'l': ['l', 'n'],
    'g': ['k'],
    'k': ['kʰ', 'k'],
    'h': ['h'],
    'j': ['ts'],               # 鸡 → 资（舌面音变舌尖音）
    'q': ['tsʰ'],              # 七 → 雌（如清 qing → cing）
    'x': ['s', 'ɬ'],           # 西 → 斯/希
    'z': ['ts'],
    'c': ['tsʰ'],
    's': ['s', 'ɬ'],
    'zh': ['ts'],              # 知 → 资
    'ch': ['tsʰ'],             # 吃 → 雌
    'sh': ['s', 'ɬ'],          # 诗 → 斯/希
    'r': ['l'],                # 日 → 啦
    'y': [''],
    'w': [''],
    '': ['']
}

# 韵母映射：普通话发音描述 → 莆仙话发音
# 依据实际音系对应，如：六 liu → 莆仙话 lau，天 tian → 莆仙话 nang
MANDARIN_TO_PUTIAN_FINALS = {
    # 单韵母
    'a': ['a'],
    'o': ['o'],
    'e': ['e'],
    'i': ['i'],
    'u': ['u'],
    'ü': ['y'],

    # 复韵母
    'ai': ['ai'],
    'ei': ['ei'],
    'ao': ['au'],
    'ou': ['u'],               # 豆 dou → tau（ou→u）

    # 鼻韵母（莆仙话前后鼻音不分）
    'an': ['aŋ'],
    'en': ['eŋ'],
    'in': ['iŋ'],
    'un': ['uŋ'],
    'ün': ['yŋ'],
    'ang': ['aŋ'],
    'eng': ['eŋ'],
    'ing': ['iŋ'],
    'ong': ['uŋ'],

    # 齐齿呼
    'ia': ['ia'],
    'ie': ['i', 'ie'],         # 姨 yi → i（ie→i）
    'iao': ['iau'],
    'iu': ['au', 'iu'],        # 六 liu → lau（iu→au）
    'ian': ['aŋ', 'ian'],      # 天 tian → nang（ian→aŋ）
    'iang': ['iaŋ'],
    'iong': ['yŋ'],

    # 合口呼
    'ua': ['ua'],
    'uo': ['uo'],
    'uai': ['uai'],
    'ui': ['ui'],
    'uan': ['uaŋ'],
    'uang': ['uaŋ'],

    # 撮口呼
    'üe': ['yø'],

    # 特殊
    'er': ['ə']
}

# 常用词语发音映射（普通话发音描述 → 莆仙话发音）
# 用户用普通话描述方言词发音，系统直接映射到正确的莆仙话发音
# 参考：莆仙话拼音方案规范文档及实际方言词典数据
MANDARIN_PRONUNCIATION_MAP = {
    # 基础词汇（按拼音首字母排序）
    'a': ['a'],
    'ba': ['pa', 'ba'],
    'dou': ['tau'],           # 豆 dou → tau
    'fu': ['hu'],             # 腐 fu → hu（莆仙话无f音）
    'jie': ['i'],             # 姨 yi → i
    'liu': ['lau'],           # 六 liu → lau（iu→au）
    'ma': ['ma'],
    'ren': ['lang', 'nin'],   # 人 ren → lang
    'shui': ['cui'],          # 水 shui → cui
    'tian': ['nang'],         # 天 tian → nang（ian→aŋ）
    'yue': ['gueh'],          # 月 yue → gueh

    # 常见短语
    'doufu': ['tau hu'],      # 豆腐 doufu → tau hu
    'liuyuetian': ['lah gue nang'],  # 六月天 liuyuetian → lah gue nang
    'ajie': ['a i'],          # 阿姨 ajie → a i
    'baba': ['a pa'],         # 爸爸 baba → a pa
    'mama': ['a ma'],         # 妈妈 mama → a ma
    'shuihu': ['cui hu'],     # 水浒 shuihu → cui hu
    'renmin': ['lang min'],   # 人民 renmin → lang min

    # 方言特色词（直接映射莆仙话发音）
    'langba': ['lor ba'],     # 郎罢（父亲）
    'age': ['a go'],          # 阿公（爷爷）
    'ayi': ['a i'],           # 阿姨
    'nainai': ['a na'],       # 奶奶
    'dian': ['tiang'],        # 店
    'qing': ['cing'],         # 清 qing → cing（q→c）
    'hong': ['ang'],          # 红 hong → ang
}

# ============================================================
# 声调映射（仅用于生成 IPA 调值，实际查询时会忽略调值数字）
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
# 三、拼音切分函数（核心）
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
    finals = list(PUTIAN_FINALS.keys()) + list(MANDARIN_TO_PUTIAN_FINALS.keys())
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
# 四、莆仙拼音 → IPA 候选（权重 1.0，支持多口音）
# ============================================================

def putian_syllable_to_ipa(syl: str, tone: Optional[str] = None) -> List[str]:
    """莆仙拼音音节转 IPA（不带声调数字），包含口音变体"""
    initial, final = split_putian_syllable(syl)
    ipa_init = PUTIAN_INITIALS.get(initial, '')
    ipa_finals = PUTIAN_FINALS.get(final, [final])
    candidates = []
    for ipa_final in ipa_finals:
        if ipa_init:
            candidates.append(ipa_init + ipa_final)
        else:
            candidates.append(ipa_final)
    # 添加口音变体
    accent_candidates = []
    for cand in candidates:
        for accent, rules in ACCENT_IPA_MAP.items():
            new_cand = cand
            for orig, repl in rules:
                new_cand = new_cand.replace(orig, repl)
            if new_cand != cand:
                accent_candidates.append(new_cand)
    candidates.extend(accent_candidates)
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
# 五、普通话拼音 → IPA 候选（优化版，权重 0.8，考虑口音）
# ============================================================

def mandarin_syllable_to_ipa(syl: str) -> List[str]:
    """普通话发音描述转莆仙话IPA，包含口音变体"""
    initial, final = split_mandarin_syllable(syl)

    # 获取莆仙话声母候选
    putian_initials = MANDARIN_TO_PUTIAN_INITIALS.get(initial, [''])

    # 获取莆仙话韵母候选
    putian_finals = MANDARIN_TO_PUTIAN_FINALS.get(final, [final])

    # 组合声母和韵母
    candidates = []
    for ipa_init in putian_initials:
        for ipa_final in putian_finals:
            if ipa_init:
                candidates.append(ipa_init + ipa_final)
            else:
                candidates.append(ipa_final)

    # 添加口音变体（简单替换）
    accent_candidates = []
    for cand in candidates:
        for accent, rules in ACCENT_IPA_MAP.items():
            new_cand = cand
            for orig, repl in rules:
                new_cand = new_cand.replace(orig, repl)
            if new_cand != cand:
                accent_candidates.append(new_cand)
    candidates.extend(accent_candidates)

    # 去重，保留合理数量的候选
    candidates = list(set(candidates))[:3]
    return candidates

def mandarin_pinyin_to_ipa_candidates(pinyin_str: str) -> List[Tuple[str, float]]:
    """普通话发音描述 → 莆仙话IPA候选（权重0.8）"""
    pinyin_str = pinyin_str.lower().strip()

    # 优先检查发音映射
    norm = pinyin_str.replace(" ", "")
    if norm in MANDARIN_PRONUNCIATION_MAP:
        mappings = MANDARIN_PRONUNCIATION_MAP[norm]
        results = []
        for mapping in mappings:
            candidates = putian_pinyin_to_ipa_candidates(mapping)
            results.extend([(ipa, 0.85) for ipa, _ in candidates])
        return results[:30] if results else []

    syllables = parse_pinyin(pinyin_str)
    if not syllables:
        return []

    candidates_per_syl = []
    for syl in syllables:
        ipa_segs = mandarin_syllable_to_ipa(syl['initial'] + syl['final'])
        # 普通话查询不强制加声调（用户可能不知道方言声调）
        segs = [(seg, 0.8) for seg in ipa_segs]
        if not segs:
            # 兜底：使用原始拼音片段
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
# 六、混合拼音输入（莆仙+普通话按音节混合）
# ============================================================

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
# 七、统一搜索入口
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
    # 优先使用混合候选（支持莆仙+普通话混合输入）
    mixed_candidates = mixed_pinyin_to_ipa_candidates(query)
    if mixed_candidates:
        all_candidates = mixed_candidates
    else:
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

# ============================================================
# 八、兼容旧接口（供 pinyin_matcher.py 使用）
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
    mixed = mixed_pinyin_to_ipa_candidates(query)
    if mixed:
        return max(mixed, key=lambda x: x[1])[0]
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
    print("=== 测试修正后的普通话发音描述映射 ===")

    # 测试1：声母映射
    print("\n1. 声母映射测试：")
    test_initials = ['f', 'd', 'zh', 'ch', 'sh', 'r', 'j', 'q', 'x']
    for initial in test_initials:
        putian = MANDARIN_TO_PUTIAN_INITIALS.get(initial, [])
        print(f"   普通话'{initial}' → 莆仙话{putian}")

    # 测试2：韵母映射
    print("\n2. 韵母映射测试：")
    test_finals = ['ou', 'iu', 'ie', 'an', 'en', 'in']
    for final in test_finals:
        putian = MANDARIN_TO_PUTIAN_FINALS.get(final, [])
        print(f"   普通话'{final}' → 莆仙话{putian}")

    # 测试3：常用词发音映射
    print("\n3. 常用词发音映射测试：")
    test_words = ['liu', 'yue', 'tian', 'dou', 'fu', 'liuyuetian', 'doufu', 'ajie']
    for word in test_words:
        mapping = MANDARIN_PRONUNCIATION_MAP.get(word, [])
        print(f"   普通话'{word}' → 莆仙话{mapping}")

    # 测试4：普通话拼音转IPA
    print("\n4. 普通话拼音转IPA测试：")
    test_queries = ['dou fu', 'liu yue tian', 'a jie', 'shui', 'ren']
    for query in test_queries:
        candidates = mandarin_pinyin_to_ipa_candidates(query)
        print(f"   '{query}' → {[c[0] for c in candidates[:3]]}")

    # 测试5：莆仙话拼音转IPA（确保原有功能不变）
    print("\n5. 莆仙话拼音转IPA测试（原有功能）：")
    test_putian = ['a1 ma3', 'ah6 sau1', 'lor2 ba5']
    for query in test_putian:
        candidates = putian_pinyin_to_ipa_candidates(query)
        print(f"   '{query}' → {[c[0] for c in candidates[:3]]}")

    # 测试6：混合拼音测试
    print("\n6. 混合拼音测试（莆仙话+普通话）：")
    test_mixed = ['a1 dou', 'liu lang', 'tiau fu']
    for query in test_mixed:
        candidates = mixed_pinyin_to_ipa_candidates(query)
        print(f"   '{query}' → {[c[0] for c in candidates[:3]]}")