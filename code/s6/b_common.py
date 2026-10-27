"""S6 frozen decisions and additive descriptive metrics; no classifier changes."""
import hashlib,json,statistics,time
from pathlib import Path
import numpy as np
from config import ROOT as R
O=R/'S6'
STYLES=('haera','hasipsio','aegyo','eumseum');LAYERS=(8,14,20,27,31);DOSES=(.15,.25,.4,.5,.65)
RULE_HASH='320df4ff3a461b03899e167a2d8035a8f6e6ac9876c4d8c19a1ac76853fc9852'
CORE=('니','냐','구나','렴','거라','단다','주마','하마','으마')
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def freeze_check():
 f=load(O/'freeze.json');assert sha(O/'RULE.md')==RULE_HASH==f['rule_sha256']
 for p,h in f['canonical_input_sha256'].items():assert sha(O/p)==h,p
 assert load(O/'engine_validation.json')['gate_passed']
 return f
def progress(event,**kw):
 x=dict(time=time.time(),event=event,**kw)
 with (O/'logs/progress_b.jsonl').open('a') as f:f.write(json.dumps(x,ensure_ascii=False)+'\n')
 print(json.dumps(x,ensure_ascii=False),flush=True)
def aggregate(rows,style):
 n=len(rows);assert n
 mean=lambda k:statistics.mean(r[k] for r in rows)
 core=sum(r['haera_core']['count'] for r in rows);segments=sum(r['style_segments'] for r in rows)
 return dict(n=n,target_dominant=sum(r['dominant_style']==style for r in rows),
  hangul_pass=sum(r['hangul_share']>=.95 for r in rows),distinct4_pass=sum(r['distinct4']>=.9 for r in rows),
  length_pass=sum(.5<=r['length_ratio']<=2 for r in rows),second_person_delta=mean('second_person_delta'),
  haera_core=dict(count=core,segments=segments,share=core/max(1,segments),reply_mean_share=statistics.mean(r['haera_core']['share'] for r in rows)),
  dominant_counts={s:sum(r['dominant_style']==s for r in rows) for s in (*STYLES,'friend','haeyo','other_casual','unknown')},
  style_shares_mean={s:statistics.mean(r['style_shares'][s] for r in rows) for s in rows[0]['style_shares']},
  **{k:mean(k) for k in ['hangul_share','distinct4','length_ratio','second_person','second_person_substring','content_jaccard_none','content_recall_none','content_recall_user','content_overlap_user','repetition4','max_word_repeat','max_char4_repeat']},
  broken_bytes=sum(r['broken_bytes'] for r in rows),replacement_chars=sum(r['replacement_chars'] for r in rows),truncated=sum(r['truncated'] for r in rows),
  app_target_problems=sum(r['app_target_problem'] is not None for r in rows),
  old_flips=sum(r['flip'] for r in rows),friend_largest=sum(r['friend_largest'] for r in rows))
def decide(a,base,control):
 checks=dict(target=a['target_dominant']>=14,none=base['target_dominant']<=3,random=control['target_dominant']<=3,
             hangul=a['hangul_pass']>=18,distinct4=a['distinct4_pass']>=18,length=a['length_pass']>=18,second_person=a['second_person_delta']<=.5)
 return dict(checks=checks,passed=all(checks.values()),failed_checks=[k for k,v in checks.items() if not v],none_dominant=base['target_dominant'],control_dominant=control['target_dominant'])
def rank(a):return (-a['target_dominant'],-min(a[k] for k in ['hangul_pass','distinct4_pass','length_pass']),a['dose'],a['layer'],a['name'])
