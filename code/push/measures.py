"""Cheap deterministic proxies; none establishes meaning or fluency."""
import re,unicodedata,collections
from diagnose_checks import register_problem
POLITE=re.compile(r'(요|니다|니까|시오|십시오|세)$')
CASUAL=re.compile(r'(어|아|야|지|자|래|게|니|냐|거든|잖아|해|줘|봐|다|대|걸|까|구나|더라|는데|네|렴)$')
NOUNS={'필요','중요','주요','수요','강요','소요','개요','동요','요요','담요','민요'}
STOP={'그리고','그래서','하지만','그냥','정말','조금','아주','하나','오늘','네','응','것','수','거','이','그','저','너','우리','저는','나는'}
SUFFIXES=sorted(['습니다','입니다','해주세요','해줘','하세요','해요','어요','아요','에게','에서','으로','하고','이나','이나요','는','은','이','가','을','를','에','도','와','과','로'],key=len,reverse=True)
def words(text):
    result=set()
    for w in re.findall(r'[가-힣A-Za-z]{2,}',text.lower()):
        for s in SUFFIXES:
            if w.endswith(s) and len(w)-len(s)>=2:w=w[:-len(s)];break
        if w not in STOP:result.add(w)
    return result
def register(text):
    counts=dict(banmal=0,polite=0,unknown=0)
    for s in re.split(r'[.!?…\n]+',text):
        if not s.strip():continue
        tail=s.strip()
        while tail and (tail[-1].isspace() or tail[-1] in '~^ㅋㅎ' or unicodedata.category(tail[-1])[0] in 'PS'):
            tail=tail[:-1]
        m=re.search(r'([가-힣]+)$',tail);w=m.group(1) if m else ''
        if w and w not in NOUNS and POLITE.search(w):kind='polite'
        elif w and CASUAL.search(w):kind='banmal'
        else:kind='unknown'
        counts[kind]+=1
    n=sum(counts.values());shares={k+'_share':v/max(1,n) for k,v in counts.items()}
    return dict(**counts,sentences=n,**shares)
def hangul(c):return '\uac00'<=c<='\ud7a3' or '\u1100'<=c<='\u11ff' or '\u3130'<=c<='\u318f'
def measure(row,base,scenario):
    text=row['reply'];reg=register(text);breg=register(base['reply']);target=scenario['target'];source=scenario['context_register']
    alpha=[c for c in text if unicodedata.category(c).startswith('L')]
    normalized=' '.join(text.split());grams=[normalized[i:i+4] for i in range(max(0,len(normalized)-3))]
    distinct=len(set(grams))/len(grams) if grams else 0
    ws=words(text);bs=words(base['reply']);us=words(scenario['user'])
    others=collections.Counter(unicodedata.name(c,'UNKNOWN').split()[0] for c in alpha if not hangul(c))
    freq=collections.Counter(normalized.split())
    return dict(**reg,flip=bool(breg[source+'_share']>.5 and reg[target+'_share']>.5),base_source_dominant=breg[source+'_share']>.5,
                hangul_share=sum(map(hangul,alpha))/len(alpha) if alpha else 0,distinct4=distinct,repetition4=1-distinct,
                max_word_repeat=max(freq.values(),default=0),max_char4_repeat=max(collections.Counter(grams).values(),default=0),
                length_ratio=row['length']/max(1,base['length']),other_scripts=dict(others),replacement_chars=text.count('\ufffd'),
                app_target_problem=register_problem(text,'banmal' if target=='banmal' else 'haeyo',scenario['user']),
                app_source_problem=register_problem(text,'banmal' if source=='banmal' else 'haeyo',scenario['user']),
                content_jaccard_none=len(ws&bs)/max(1,len(ws|bs)),content_recall_none=len(ws&bs)/max(1,len(bs)),
                content_overlap_user=len(ws&us),content_recall_user=len(ws&us)/max(1,len(us)),content_words=sorted(ws))
