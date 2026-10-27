"""All-condition reports. haera_core is descriptive and never affects selection."""
import collections,csv,gzip,json,statistics,time
from b_common import *

def write(name,s):
 p=O/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:f.write(s)
def tsv(name,rows,columns):
 p=O/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=columns,delimiter='\t',lineterminator='\n',extrasaction='ignore');w.writeheader()
  for row in rows:w.writerow({k:json.dumps(v,ensure_ascii=False,separators=(',',':')) if isinstance(v,(dict,list)) else v for k,v in row.items()})
def core(a):
 h=a['haera_core'];return f'{h["count"]}/{h["segments"]} ({h["share"]:.1%})'
def mcell(text):return text.replace('|','\\|').replace('\n','<br>')
def replycell(r):
 if r is None:return '—'
 return mcell(r['reply'])+'<br><small>'+mcell(f"dominant={r['dominant_style']}; haera_core={core(r)}; H={r['hangul_share']:.3f}; D4={r['distinct4']:.3f}; length×{r['length_ratio']:.2f}; 2pΔ={r['second_person_delta']:+d}; content-recall={r['content_recall_none']:.3f}; broken={r['broken_bytes']}; truncated={r['truncated']}")+'</small>'
def enrich(rows,s):
 a=aggregate(rows,s)
 a.update(ending_counts={k:sum(r['style_counts'][k] for r in rows) for k in rows[0]['style_counts']},
          old_ending_counts={k:sum(r['ending_types'][k] for r in rows) for k in rows[0]['ending_types']},
          other_scripts=dict(sum((collections.Counter(r['other_scripts']) for r in rows),collections.Counter())),
          second_person_substring_delta=statistics.mean(r['second_person_substring']-r['second_person_substring_none'] for r in rows),
          app_source_problems=sum(r['app_source_problem'] is not None for r in rows))
 return a
def main():
 freeze_check();assert (O/'logs/gpu_pipeline_complete.json').exists()
 plans=load(O/'dev_conditions.json');selection=load(O/'selection.json');ns=load(O/'neuron_selection.json')
 checkpoints={phase:{p.stem:load(p) for p in sorted((O/'checkpoints'/phase).glob('*.json'))} for phase in ['dev','heldout']}
 assert len(checkpoints['dev'])==2*len(plans['conditions'])+5
 assert len(checkpoints['heldout'])==5+3*sum(bool(x['selected']) for x in selection['styles'].values())
 # Preserve the reviewed Phase A reports and checksum manifest before replacing them.
 previous=O/'_superseded/phase_A_report';previous.mkdir(parents=True,exist_ok=True)
 for name in ['README_ko.md','summary.md','SHA256SUMS.txt']:
  if (O/name).exists():(O/name).rename(previous/name)
 # The GGUF contains unused raw app layers too; the frozen app comparison scope
 # is the actual original application range4..14, hence only style layers8/14.
 cos_path=O/'map/cosines.json';cos=load(cos_path)
 old_cos=O/'_superseded/app_cosine_scope_correction';old_cos.mkdir(parents=True,exist_ok=True)
 cos_path.rename(old_cos/'cosines_all_raw_artifact_layers.json')
 for l,entry in cos['layers'].items():
  if int(l) not in range(4,15):
   for ref in entry['references'].values():ref['app']=None
 cos['app_scope']='Original applied layers4..14 only; among style layers,8 and14 match. Other raw vectors in the GGUF are outside the frozen scope.'
 save(cos_path,cos)
 allrows=[dict(r,phase=phase,condition_key=key) for phase,by in checkpoints.items() for key,data in by.items() for r in data['rows']]
 columns=sorted({k for r in allrows for k in r})
 with gzip.open(O/'generations.tsv.gz','xt',encoding='utf8',newline='',compresslevel=6) as f:
  w=csv.DictWriter(f,fieldnames=columns,delimiter='\t',lineterminator='\n');w.writeheader()
  for r in allrows:w.writerow({k:json.dumps(v,ensure_ascii=False,separators=(',',':')) if isinstance(v,(dict,list,bool)) or v is None else v for k,v in r.items()})
 dev=[]
 for a in selection['dev_summaries']:
  rr=checkpoints['dev'][a['key']]['rows'];a=dict(a,**{})
  # Enrich descriptive fields while preserving every frozen selection field.
  a.update(enrich(rr,a['style']));dev.append(a)
 held=[]
 for s in STYLES:
  selected=selection['styles'][s]['selected']
  keys=['none',f'prompt.{s}']+([selected['key'],'prompt+'+selected['key'],selected['control_key']] if selected else [])
  for key in keys:
   rows=checkpoints['heldout'][key]['rows']
   for context in ['all','polite','banmal']:
    rr=[r for r in rows if context=='all' or r['context_register']==context]
    held.append(dict(style=s,key=key,context=context,**enrich(rr,s)))
 save(O/'summary.json',dict(rule_sha256=RULE_HASH,selection=selection['styles'],dev=dev,heldout=held,
                           n_generated_replies=len(allrows),dev_conditions=len(checkpoints['dev']),heldout_conditions=len(checkpoints['heldout']),
                           haera_core_definition='Only haera-labeled segments ending in 니/냐/구나/렴/거라/단다/주마/하마/으마; pooled count/all segments; descriptive only'))
 mapcols=['style','layer','dose','name','target_dominant','haera_core','n','none_dominant','control_dominant','hangul_pass','distinct4_pass','length_pass','second_person_delta','passed','failed_checks','style_shares_mean','broken_bytes','truncated']
 tsv('map/style_layer.tsv',[a for a in dev if a['kind']=='direction'],mapcols)
 neuron_rows=[]
 for x in ns['candidates']:
  admitted=any(a['layer']==x['layer'] and a['neuron']==x['neuron'] for a in ns['admitted'][x['style']])
  neuron_rows.append(dict(stage='screen',**x,admitted=admitted))
 for a in dev:
  if a['kind']=='neuron':
   n=int(a['name'].split('.N')[-1]);neuron_rows.append(dict(stage='dev',neuron=n,**a))
 ncols=['stage','style','layer','neuron','lens_score','mean_shift','control_p95','beats_p95','admitted','shifts','kl','entropy_change','top1','plausible','name','dose','target_dominant','haera_core','none_dominant','control_dominant','hangul_pass','distinct4_pass','length_pass','second_person_delta','passed','failed_checks']
 tsv('map/style_neurons.tsv',neuron_rows,ncols)
 lines=['# S6: frozen Phase B results','',f'RULE SHA256 `{RULE_HASH}`. Full dev grid and held-out generation are complete.','',
        '**Directions contrast ending style in minimally differing replies. They do not test tone carried by word choice such as 얘야, 헤헤 or internet vocabulary.**',
        'haera includes plain declarative -다, also common in friend speech (좋겠다, 맛있겠다). haera_core excludes plain -다 and is descriptive only; it never affects qualification or selection.',
        'haera_core is shown as core ending segments / all nonempty segments (pooled share), including unknown segments in the denominator. Full JSON also stores mean per-reply shares.','',
        '## Frozen selections','', '| Style | Result | Selected | Best near-miss |','|---|---|---|---|']
 for s,x in selection['styles'].items():lines.append(f'| {s} | {x["status"]} | {x["selected"]["key"] if x["selected"] else "—"} | {x["near_miss"]["key"] if x["near_miss"] else "—"} |')
 lines+=['','## Dev: every selectable condition','',
         '| Style / condition | Dose | Frozen target dominant /20 | haera_core | None / random dominant | H / D4 / length pass | 2p delta | Decision / failed checks |',
         '|---|---:|---:|---:|---:|---:|---:|---|']
 for a in sorted(dev,key=lambda x:(x['style'],x['name'],x['dose'])):
  lines.append(f'| {a["name"]} | {a["dose"]:g} | {a["target_dominant"]} | {core(a)} | {a["none_dominant"]} / {a["control_dominant"]} | {a["hangul_pass"]} / {a["distinct4_pass"]} / {a["length_pass"]} | {a["second_person_delta"]:+.2f} | '+('PASS' if a['passed'] else 'FAIL: '+', '.join(a['failed_checks']))+' |')
 lines+=['','## Dev: None, prompt-only and every paired random control','',
         '| Target style | Condition | Frozen target dominant /20 | haera_core | H / D4 / length pass | 2p delta |','|---|---|---:|---:|---:|---:|']
 for s in STYLES:
  keys=['none',f'prompt.{s}']+[c['control_key'] for c in plans['conditions'] if c['style']==s]
  for key in keys:
   a=enrich(checkpoints['dev'][key]['rows'],s)
   lines.append(f'| {s} | {key} | {a["target_dominant"]} | {core(a)} | {a["hangul_pass"]} / {a["distinct4_pass"]} / {a["length_pass"]} | {a["second_person_delta"]:+.2f} |')
 lines+=['','## Held-out: same measures, no reselection','',
         '| Style | Condition | Context | Frozen target dominant /N | haera_core | H / D4 / length pass | 2p delta | Content recall vs None |',
         '|---|---|---|---:|---:|---:|---:|---:|']
 for a in held:
  lines.append(f'| {a["style"]} | {a["key"]} | {a["context"]} | {a["target_dominant"]}/{a["n"]} | {core(a)} | {a["hangul_pass"]} / {a["distinct4_pass"]} / {a["length_pass"]} | {a["second_person_delta"]:+.2f} | {a["content_recall_none"]:.3f} |')
 lines+=['','## Measurement limits and execution','',
         'All S3b damage, listener-address, old register/app-checker and content-word proxy fields remain in generations.tsv.gz. Per-condition summaries, script counts and all style shares are in summary.json. Content overlap is not a semantic-accuracy score.',
         'The classifier is the approved s6-endings-1.1.0, including its known -넘겨 miss and unvalidated other_casual label. Broad suffix rules can misread lexical fragments, quotations and malformed text. Sparse aegyo token groups and lexical homographs limit the neuron lens.',
         'Prompt-only comparisons use the one frozen system addition per style. They do not establish the best prompt achievable with wording changes or examples. Directions were built to isolate endings and do not establish control of word-choice tone.',
         'The S1b full-vocabulary z-score is computed through an algebraically equivalent covariance identity in float64 and checked against direct float32 full-vocabulary logits at every layer and every final top40 candidate. No grouping or ranking edits follow the outcomes.',
         'The next-token screen uses the original validated S2 engine and kernels, last-position-only, as specified for the S2 screen. Generation and contrastive extraction use S3b ExactEngine step()/generate() with fma=True. An initial attempt to use the S3b exact arithmetic override with the S2 cache showed large zero-push logit differences despite12/12 top1 agreement. It was stopped before neuron admission or the dev grid; partial screen files were moved to _superseded/screen_engine_correction/. The corrected original S2 screen requires12/12 zero-push top1 agreement and max logit difference<.02 at every screened layer. All wasted GPU job time remains charged to the budget. This numerical path distinction is disclosed against the broad ExactEngine wording of RULE; candidate groups, statistic, p95, doses, insertion, selection and generation engine are unchanged.',
         'Reference work in Phase B included compilation and vocabulary-only llama.cpp tokenization while the parent held the reference CPU lock. Existing20 dev and10 CPU-validation prompt token IDs matched exactly; every build-pair prefix boundary was checked. No reference model inference, model downloads, package installation, git or [unrelated directory] access.',
         'Fresh S6 dev None generation also matched the stored S3b None token sequence in20/20 replies; logs/dev_none_s3b_audit.json records the comparison.',
         'The later interactive review page has not been produced. review_dev.md and review_heldout.md contain the complete frozen review sets.','']
 write('summary.md','\n'.join(lines))
 # Complete reply review tables. Each cell discloses frozen dominant and haera_core.
 for phase,filename in [('dev','review_dev.md'),('heldout','review_heldout.md')]:
  ss=load(O/f'scenarios_{phase}.json');lines=[f'# S6 {phase} complete replies','',
      'Frozen dominant style and descriptive haera_core are shown together in every reply. haera_core never changes pass/selection. H=Hangul share; D4=distinct-4. Content recall is a lexical proxy, not meaning accuracy.','']
  for s,x in selection['styles'].items():
   c=x['selected'];near=x['near_miss']
   arms=[('None','none'),('Prompt-only',f'prompt.{s}')]
   if phase=='dev':arms += [('Selected knob',c['key'] if c else None),('Best near-miss',near['key'] if near else None)]
   else:arms += [('Knob',c['key'] if c else None),('Prompt + knob','prompt+'+c['key'] if c else None),('Random',c['control_key'] if c else None)]
   lines += [f'## {s}: {x["status"]}','',f'Selected: {c["key"] if c else "none"}. Near-miss: {near["key"] if near else "none"}.', '',
             '| Condition | Frozen target dominant /N | haera_core | H / D4 / length pass | 2p delta |','|---|---:|---:|---:|---:|']
   for label,key in arms:
    if key is None:continue
    a=enrich(checkpoints[phase][key]['rows'],s)
    lines.append(f'| {label}: {key} | {a["target_dominant"]}/{a["n"]} | {core(a)} | {a["hangul_pass"]} / {a["distinct4_pass"]} / {a["length_pass"]} | {a["second_person_delta"]:+.2f} |')
   if phase=='dev' and near:lines += ['', 'Near-miss failed checks: '+', '.join(near['failed_checks'])+'.']
   for scenario in ss:
    earlier=' / '.join(m['role']+': '+m['content'] for m in scenario['messages'][1:-1])
    lines += ['',f'### {scenario["id"]} ({scenario["context_register"]})','','Earlier turns: '+earlier,'','User: '+scenario['user'],'',
              '| '+' | '.join(a[0] for a in arms)+' |','|'+'---|'*len(arms)]
    values=[]
    for label,key in arms:
     r=next((r for r in checkpoints[phase][key]['rows'] if r['scenario']==scenario['id']),None) if key else None
     values.append(replycell(r))
    lines.append('| '+' | '.join(values)+' |')
   lines.append('')
  write(filename,'\n'.join(lines)+'\n')
 budget=[json.loads(l) for l in (O/'logs/gpu_budget_b.jsonl').read_text().splitlines()];seconds=sum(x['seconds'] for x in budget if x['event']=='end')
 gpu=[json.loads(l) for l in (O/'logs/gpu_checks.jsonl').read_text().splitlines()]
 ko=['# S6: 어미 말투 결 손잡이','',
     '[잰 값] 승인받은 Phase B의 전체 dev 격자와 held-out 생성을 끝냈다. 규칙 SHA256은 `'+RULE_HASH+'`다.',
     f'[잰 값] dev {len(checkpoints["dev"])}조건, held-out {len(checkpoints["heldout"])}조건, 답변 {len(allrows)}개를 저장했다.',
     '[읽기] 이번 답변쌍은 내용과 어휘를 거의 그대로 두고 어미를 바꾼 최소 대비다. 따라서 방향은 어미 스타일 방향이다. 얘야·헤헤·인터넷 낱말처럼 어휘 선택으로 드러나는 말투까지 바꾸는 실험은 아니다.',
     '[읽기] haera에는 평서형 -다가 들어간다. 친구 반말도 좋겠다·맛있겠다 같은 -다를 흔히 쓴다. haera 우세만으로 어른이 아랫사람에게 하는 말투라고 결론 내릴 수 없다.',
     '[읽기] 부모 요청으로 haera_core를 설명용으로 더했다. haera로 분류된 조각 중 니/냐/구나/렴/거라/단다/주마/하마/으마로 끝나는 것만 센다. 순수 -다는 뺀다. 전체 조각 대비 수와 비율을 summary와 지도·리뷰에 나란히 실었다. 통과나 선택에는 쓰지 않았다.','',
     '| 스타일 | 판정 | 선택 | 가장 가까운 미통과 조건 |','|---|---|---|---|']
 for s,x in selection['styles'].items():ko.append(f'| {s} | {x["status"]} | {x["selected"]["key"] if x["selected"] else "—"} | {x["near_miss"]["key"] if x["near_miss"] else "—"} |')
 ko+=['','| 스타일 | 선택 또는 최상위 미통과 | dev 목표 우세 /20 | haera_core | 한글·D4·길이 통과 /20 | 2인칭 증가 /답변 | 미통과 기준 |',
      '|---|---|---:|---:|---:|---:|---|']
 for s,x in selection['styles'].items():
  a=x['selected'] or x['near_miss']
  ko.append(f'| {s} | {a["key"]} | {a["target_dominant"]} | {core(a)} | {a["hangul_pass"]}·{a["distinct4_pass"]}·{a["length_pass"]} | {a["second_person_delta"]:+.2f} | '+(', '.join(a['failed_checks']) if a['failed_checks'] else '없음')+' |')
 ko+=['','[읽기] 아래는 새 held-out 12개에서 각 목표 말투가 우세한 답 수다. held-out으로 조건을 다시 고르거나 통과 기준을 새로 만들지 않았다. no knob 스타일에는 knob 관련 생성을 하지 않았다.','',
      '| 스타일 | None /12 | prompt-only /12 | knob /12 | prompt + knob /12 | random /12 |',
      '|---|---:|---:|---:|---:|---:|']
 for s,x in selection['styles'].items():
  c=x['selected'];keys=['none',f'prompt.{s}',c['key'] if c else None,'prompt+'+c['key'] if c else None,c['control_key'] if c else None]
  values=[str(next(a['target_dominant'] for a in held if a['style']==s and a['key']==k and a['context']=='all')) if k else '—' for k in keys]
  ko.append('| '+s+' | '+' | '.join(values)+' |')
 ko+=['','[읽기] no knob는 이 고정 층·용량·뉴런 후보와 기준 안에서 통과가 없었다는 뜻이다. 가능한 모든 말투 제어의 부재를 뜻하지 않는다.',
      '[읽기] [summary.md](summary.md)는 전 조건의 판정과 haera_core다. [summary.json](summary.json)에 어미별 비율·대명사·손상·내용어 대리값을 저장했다.',
      '[읽기] [review_dev.md](review_dev.md)는 선택 조건과 가장 가까운 미통과 조건의 20개 답 전체를 None·prompt-only 옆에 놓았다. [review_heldout.md](review_heldout.md)는 스타일별 held-out 12개 전체다.',
      '[읽기] [map/style_layer.tsv](map/style_layer.tsv), [map/style_neurons.tsv](map/style_neurons.tsv), [map/cosines.json](map/cosines.json)에 층·용량 지도, 스타일별 40개씩 총 160개 뉴런 후보 항목의 선별·dev 결과, 같은 층 방향 코사인이 있다. 스타일 사이에 같은 뉴런이 겹칠 수 있다. 맞는 층이 없는 N6893 등은 null이다.',
      '[잰 값] 짝 무작위 대조는 용량을 곱하기 전 실제 벡터 노름의 상대 차이가 3e-7 미만인지 확인했다. dev_conditions.json에는 벡터별 노름·해시가, 각 생성 체크포인트에는 용량 적용 후 실제 주입 노름이 있다. 독립 감사에는 부동소수점 허용치와 최대 차이도 기록했다.',
      '[읽기] 지배 말투는 동결 분류기로 정했다. 모르는 문장 조각도 분모에 넣는다. other_casual은 Phase A에서 검증하지 못했다. 어미·명사 중의성, 인용문, 조각과 깨진 생성은 오판할 수 있다.',
      '[읽기] 한글·distinct-4·길이 통과는 뜻 유지나 자연스러움의 보장이 아니다. 내용어 겹침도 값싼 대리값이다. 원문을 함께 읽어야 한다.',
      '[읽기] prompt-only는 스타일마다 미리 정한 덧줄 하나만 시험했다. 문구나 예시를 다듬었을 때의 최선 성능까지 비교한 것은 아니다.',
      '[읽기] lens는 전체 262144 어휘를 쓰는 S1b의 pre-softcap z-score다. 평균·공분산 항등식으로 계산량을 줄였고, 각 층과 최종 top40 후보는 전체 로짓의 직접 계산과 비교했다. 기록은 lens/top40.json과 층별 파일에 있다.',
      '[읽기] 뉴런 선별은 기존 S2의 검증된 엔진·커널·12개 프레임·마지막 위치 캐시 경로를 썼다. 잔차 추출과 생성은 S3b ExactEngine(fma=True)이다. RULE의 포괄적인 ExactEngine 문구와 구분되는 수치 경로다. 선별 통계·토큰·대조·삽입 위치·기준·용량·선택법은 그대로다.',
      '[읽기] 처음 S3b의 exact 연산을 S2 캐시 경로에 적용하자 0밀기 로짓 차이가 컸다. top1은 12/12 같았지만 선별을 중단했다. 뉴런 채택·dev 생성 전에 기존 S2 경로로 고쳤고, 층마다 0밀기 top1 12/12와 로짓 차이 0.02 미만을 확인했다. 이전 부분 파일은 _superseded/screen_engine_correction/에 보존했다. 실패 시도도 GPU 예산에 포함했다.',
      '[읽기] 재개 중 이전 방향 체크섬 파일이 남아 한 번 더 멈췄다. 이전 체크섬을 보존 위치로 옮기고 새 방향 JSON의 체크섬을 기록한 뒤, 같은 코드·활성값 체크포인트로 재개했다. 이 실패 시간도 예산에 포함했다. 앱 코사인은 실제 적용 범위 4–14층과 겹치는 8·14층만 남겼다. GGUF의 다른 원시 층을 비교했던 중간 파일도 보존했다.',
      f'[잰 값] Phase B GPU 작업 누적 벽시계 시간은 {seconds/60:.2f}분이다. 예산은 180분 안팎, 강제 상한은 210분이었다. 이 값은 GPU 사용률을 적분한 시간이 아니다.',
      f'[잰 값] 전체 GPU 점유 확인 {len(gpu)}회 중 외부 점유로 멈춘 기록은 {sum(x["pause"] for x in gpu)}회다. 검사 시 외부 점유 최대는 {max(x["external_mib"] for x in gpu):.0f} MiB다. Phase A 확인도 포함한다.',
      '[읽기] 부모가 reference CPU lock을 잡은 상태에서 기존 llama.cpp의 어휘 전용 토큰화를 했다. 기존 dev 20개·검증 10개의 토큰열이 같았다. 기준 CPU 모델 추론은 하지 않았다.',
      '[잰 값] Phase B에서 새로 생성한 dev None도 저장된 S3b None과 20/20 토큰열이 정확히 같았다. logs/dev_none_s3b_audit.json에 기록했다.',
      '[읽기] 새 모델·패키지를 받지 않았다. 네트워크는 [compute host] ssh/scp뿐이다. git과 [unrelated directory]를 쓰지 않았다. Phase A 보고서와 이전 체크섬은 _superseded/phase_A_report/로 옮겼다.',
      '[읽기] SHA256SUMS.txt는 자기 자신·캐시·큰 기존 입력·보존한 이전 버전과, 작성 중인 driver_b.log·b_verify.log 및 종료 표시 DONE_B를 제외한다. 종료 표시는 미러링하지만 이 체크섬 묶음에는 넣지 않는다.',
      '[읽기] 부모가 이 결과를 읽기 전에는 별도 리뷰 페이지를 만들지 않는다. Phase A에서 정한 페이지용 6개 선택은 그대로 보존했다.',
      '[읽기] Phase A의 분류기 검증과 답변쌍 설명은 _superseded/phase_A_report/README_ko.md와 classifier/에 있다. 동결 파일은 바꾸지 않았다. 실행 종료 표시는 실행 환경 logs/DONE_B다.','']
 write('README_ko.md','\n'.join(ko))
 print(json.dumps(dict(generated_replies=len(allrows),selections={s:x['selected']['key'] if x['selected'] else None for s,x in selection['styles'].items()}),indent=2))
if __name__=='__main__':main()
