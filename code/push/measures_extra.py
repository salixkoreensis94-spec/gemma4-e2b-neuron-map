"""Frozen heuristic ending subtypes and listener-address counts."""
import re,unicodedata
from measures import POLITE,CASUAL,NOUNS
FRIEND=re.compile(r'(어|아|야|지|네|거든|잖아|해|줘|봐)$')
HAERA=re.compile(r'(니|구나|다|냐|자)$')
PRONOUN=re.compile(r'(?<![가-힣])(?:네가|니가|너의|너는|너도|너)(?:를|에게|한테|와|랑|만|까지|부터|처럼|보다|라도|마저)?(?![가-힣])')
SUBSTRING=re.compile(r'네가|니가|너의|너는|너도|너')
def extra(text,base):
    c=dict(friend=0,haera=0,polite=0,other_casual=0,unknown=0);tails=[]
    for s in re.split(r'[.!?…\n]+',text):
        if not s.strip():continue
        tail=s.strip()
        while tail and (tail[-1].isspace() or tail[-1] in '~^ㅋㅎ' or unicodedata.category(tail[-1])[0] in 'PS'):tail=tail[:-1]
        m=re.search(r'([가-힣]+)$',tail);w=m.group(1) if m else ''
        if w and w not in NOUNS and POLITE.search(w):kind='polite'
        elif FRIEND.search(w):kind='friend'
        elif HAERA.search(w):kind='haera'
        elif CASUAL.search(w):kind='other_casual'
        else:kind='unknown'
        c[kind]+=1;tails.append(dict(tail=w,type=kind))
    n=sum(c.values());count=len(PRONOUN.findall(text));b=len(PRONOUN.findall(base))
    return dict(ending_types=c,ending_type_shares={k:v/max(1,n) for k,v in c.items()},ending_tails=tails,friend_largest=c['friend']>0 and all(c['friend']>c[k] for k in ['haera','polite','other_casual']),second_person=count,second_person_none=b,second_person_delta=count-b,second_person_substring=len(SUBSTRING.findall(text)),second_person_substring_none=len(SUBSTRING.findall(base)))
