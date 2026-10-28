"""Audit exhaustive coverage, apply the frozen rule, select all-position checks."""
import argparse
import csv
import gzip
import hashlib
import json
import math
import time
import html
from pathlib import Path
import numpy as np

from config import ROOT, MODEL;OUT=ROOT/'S2'
WIDTHS=[6144]*15+[12288]*20
NAMED={(26,6893),(27,1939),(28,3794),(34,6703)}

def load(p):return json.loads(Path(p).read_text())
def save(p,obj):
    with Path(p).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def rank(r):return (abs(float(r['mean_shift'])),-int(r['layer']),-int(r['neuron']),-float(r['dose']))
WORST={}
def retain(pool,r,limit):
    key=(int(r['layer']),int(r['neuron']))
    value=rank(r)
    if key in pool and rank(pool[key])>=value:return
    if key not in pool and len(pool)==limit:
        worst,worstkey=WORST[id(pool)]
        if value<=worst:return
        del pool[worstkey]
    pool[key]=r
    if len(pool)==limit:
        worstkey=min(pool,key=lambda k:rank(pool[k]));WORST[id(pool)]=(rank(pool[worstkey]),worstkey)

def analyze():
    start=time.monotonic();tops={'polite':{},'banmal':{}};strong={};named=[];qualified=[];strict_qualified=[]
    frames=load(OUT/'frames.json')
    expected_ending_ids=[set(f['banmal_ids']+f['polite_ids']) for f in frames]
    thresholds=[];counts={'neurons':0,'controls':0};layer_times=[]
    for kind in ('screen','controls'):
        assert not (OUT/(kind+'.tsv.gz')).exists(),'Refuse replacing merged output'
    with gzip.open(OUT/'screen.tsv.gz','wt',compresslevel=1,newline='') as sf,gzip.open(OUT/'controls.tsv.gz','wt',compresslevel=1,newline='') as cf:
        writers={False:csv.writer(sf,delimiter='\t',lineterminator='\n'),True:csv.writer(cf,delimiter='\t',lineterminator='\n')}
        header_written={False:False,True:False}
        for l,width in enumerate(WIDTHS):
            p=OUT/'layers'/f'layer_{l:02d}';complete=load(p/'complete.json')
            layer_times.append(dict(layer=l,seconds=complete['seconds']))
            for name,digest in complete['sha256'].items():assert sha(p/name)==digest,'Layer hash mismatch'
            bounds={d:[math.inf,-math.inf] for d in (-.8,-.5,.5,.8)}
            for control in (True,False):
                filename='controls.tsv.gz' if control else 'screen.tsv.gz'
                expected=512 if control else width
                with gzip.open(p/filename,'rt',newline='') as f:
                    reader=csv.DictReader(f,delimiter='\t');writer=writers[control]
                    if not header_written[control]:writer.writerow(reader.fieldnames);header_written[control]=True
                    nrows=0
                    for r in reader:
                        d=float(r['dose']);idx=int(r['neuron'])
                        individual=json.loads(r['shifts']);assert len(individual)==12
                        v=math.fsum(individual)/12
                        assert abs(v-float(r['mean_shift']))<1e-7
                        # Compare strict control extrema using the full saved
                        # per-frame precision, not a rounded display column.
                        r['mean_shift']=format(v,'.17g')
                        assert int(r['layer'])==l and d==(-.8,-.5,.5,.8)[nrows//expected] and idx==nrows%expected
                        assert math.isfinite(v)
                        writer.writerow([r[k] for k in reader.fieldnames]);nrows+=1
                        if control:
                            bounds[d][0]=min(bounds[d][0],v);bounds[d][1]=max(bounds[d][1],v)
                        else:
                            if (l,idx) in NAMED:named.append(r)
                            if abs(d)==.5:
                                direction='polite' if v>0 else 'banmal'
                                retain(tops[direction],r,30);retain(strong,r,50)
                                signframes=int(r['positive_frames' if v>0 else 'negative_frames'])
                                beat=(v>bounds[d][1]) if v>0 else (v<bounds[d][0])
                                strictbeat=(v>max(bounds[-.5][1],bounds[.5][1])) if v>0 else (v<min(bounds[-.5][0],bounds[.5][0]))
                                if beat and signframes>=10 and int(r['plausible_frames'])>=11:
                                    r['expected_ending_frames']=sum(t in expected_ending_ids[i] for i,t in enumerate(json.loads(r['top1'])))
                                    qualified.append(r)
                                if strictbeat and signframes>=10 and int(r['plausible_frames'])>=11:strict_qualified.append(r)
                    assert nrows==expected*4
                    counts['controls' if control else 'neurons']+=nrows
            thresholds.append(dict(layer=l,by_dose={str(k):dict(min_mean=v[0],max_mean=v[1]) for k,v in bounds.items()}))
            print('audited layer',l,flush=True)
    assert counts==dict(neurons=337920*4,controls=35*512*4)
    qualified_keys={(int(r['layer']),int(r['neuron'])) for r in qualified}
    strict_keys={(int(r['layer']),int(r['neuron'])) for r in strict_qualified}
    ending_keys={(int(r['layer']),int(r['neuron'])) for r in qualified if r['expected_ending_frames']>=11}
    strict_ending_keys={(int(r['layer']),int(r['neuron'])) for r in strict_qualified if r['expected_ending_frames']>=11}
    top={k:sorted(v.values(),key=rank,reverse=True) for k,v in tops.items()}
    strongest=sorted(strong.values(),key=rank,reverse=True)
    selected=[dict(layer=int(r['layer']),neuron=int(r['neuron']),subset='strongest',selection_mean=float(r['mean_shift']),selection_dose=float(r['dose'])) for r in strongest]
    exclude={(x['layer'],x['neuron']) for x in selected};rng=np.random.default_rng(20261026)
    cumulative=np.cumsum(WIDTHS);seen=set(exclude)
    while len(selected)<100:
        flat=int(rng.integers(337920));l=int(np.searchsorted(cumulative,flat,side='right'));n=flat-(int(cumulative[l-1]) if l else 0)
        if (l,n) in seen:continue
        seen.add((l,n));selected.append(dict(layer=l,neuron=n,subset='random',flat_index=flat))
    save(OUT/'validation_selection.json',dict(seed=20261026,rng='NumPy PCG64 default_rng',neurons=selected))
    result=dict(coverage=counts,distinct_neurons=337920,distinct_controls=17920,frames=12,
                frame_evaluations=12*(counts['neurons']+counts['controls']),thresholds=thresholds,
                qualified_distinct_neurons=len(qualified_keys),qualified_rows=qualified,
                stricter_both_dose_control_envelope_distinct_neurons=len(strict_keys),stricter_qualified_rows=strict_qualified,
                qualified_with_11_expected_ending_top1_distinct_neurons=len(ending_keys),
                stricter_controls_and_11_expected_ending_top1_distinct_neurons=len(strict_ending_keys),
                S3_bar='PASS' if len(qualified_keys)>=10 else 'FAIL',
                stricter_both_dose_bar='PASS' if len(strict_keys)>=10 else 'FAIL',
                top=top,named=named,layer_times=layer_times,screen_layer_seconds=sum(x['seconds'] for x in layer_times),
                postprocess_seconds=time.monotonic()-start)
    save(OUT/'screen_summary.json',result)
    write_top(result)

def write_top(r):
    frames=load(OUT/'frames.json');vocab=load(ROOT/'inputs/vocabulary.json')
    tokens={x['id']:x['string'] for x in vocab};qset={(int(x['layer']),int(x['neuron']),float(x['dose'])) for x in r['qualified_rows']}
    lines=['# S2 dynamic register screen','',f"Frozen S3 rule: **{r['S3_bar']}**, {r['qualified_distinct_neurons']} distinct qualifying neurons.",
           f"Conservative check against controls at BOTH +/-0.5 doses: **{r['stricter_both_dose_bar']}**, {r['stricter_both_dose_control_envelope_distinct_neurons']} distinct neurons.",'',
           f"Additional check using only the originally specified ending IDs as top-1 in >=11/12 frames: {r['qualified_with_11_expected_ending_top1_distinct_neurons']} qualifying neurons; {r['stricter_controls_and_11_expected_ending_top1_distinct_neurons']} also beat the BOTH-dose control envelope.",'',
           'Positive shift favors polite endings; negative favors casual endings. Rank uses mean shift at |dose|=0.5.',
           'Plausibility is the pre-fixed Hangul script proxy. It does not establish grammatical or semantic preservation.','',
           'KL is KL(pushed || baseline). Entropy change is in nats. Top-1 surfaces below are raw GGUF strings.','']
    def card(row,title):
        l=int(row['layer']);n=int(row['neuron']);d=float(row['dose']);passing=(l,n,d) in qset
        mean=float(row['mean_shift']);bounds=r['thresholds'][l]['by_dose'][str(d)]
        extreme=bounds['max_mean' if mean>0 else 'min_mean']
        margin=mean-extreme if mean>0 else extreme-mean
        lines.extend([f'## {title}: L{l} N{n}, d={d:+g}','',
                      f"Mean shift {float(row['mean_shift']):+.6f}; positive/negative frames {row['positive_frames']}/{row['negative_frames']}; Korean-proxy frames {row['plausible_frames']}/12; mean KL {float(row['mean_kl']):.6f}; frozen-rule eligible: **{passing}**.",'',
                      f'Same-layer, same-dose control extreme {extreme:+.6f}; excess in the tested direction {margin:+.6f}.','',
                      '| Frame | Reply prefix | Shift | KL | Δ entropy | Top-1 ID / surface | Korean proxy |',
                      '|---|---|---:|---:|---:|---|---|'])
        arrays={k:json.loads(row[k]) for k in ('shifts','kl','entropy_change','top1','plausible')}
        for i,f in enumerate(frames):
            tid=arrays['top1'][i];token=html.escape(tokens[tid]).replace('|','&#124;').replace('\n','\\n')
            lines.append(f"| {f['name']} | {f['reply_prefix']} | {arrays['shifts'][i]:+.5f} | {arrays['kl'][i]:.5f} | {arrays['entropy_change'][i]:+.5f} | {tid} / <code>{token}</code> | {bool(arrays['plausible'][i])} |")
        lines.append('')
    for direction in ('polite','banmal'):
        lines.extend([f'# Top 30 toward {direction}',''])
        for i,row in enumerate(r['top'][direction],1):card(row,f'{direction} rank {i}')
    lines.extend(['# Four named S1 candidates, regardless of qualification',''])
    for row in sorted(r['named'],key=lambda x:(int(x['layer']),int(x['neuron']),float(x['dose']))):card(row,'Named S1 candidate')
    with (OUT/'top_register.md').open('x') as f:f.write('\n'.join(lines)+'\n')

def report():
    r=load(OUT/'screen_summary.json');v=load(OUT/'engine_validation.json');a=load(OUT/'validation_allpos.json')
    pushed_cpu=[x['allpos_vs_S1_CPU'] for x in v['push_checks'] if 'allpos_vs_S1_CPU' in x]
    benchmarks=sorted((OUT/'logs').glob('benchmark_batch*.json'))
    estimate=load(benchmarks[-1]) if benchmarks else {}
    rows=['# S2: 뉴런을 직접 민 다음 토큰 검사','',
          f"[잰 값] S3의 고정 기준은 **{r['S3_bar']}**다. 서로 다른 뉴런 {r['qualified_distinct_neurons']}개가 통과했다.",
          f"[잰 값] 두 부호의 대조를 모두 합친 더 엄격한 검사도 {r['stricter_both_dose_bar']}다. 통과 뉴런은 {r['stricter_both_dose_control_envelope_distinct_neurons']}개다.",'',
          f"[잰 값] 통과 뉴런 중 {r['qualified_with_11_expected_ending_top1_distinct_neurons']}개는 11/12문장에서 미리 정한 어미 ID가 top-1이었다.",
          f"[잰 값] 이 조건과 두 부호 대조의 극값을 모두 넘은 뉴런은 {r['stricter_controls_and_11_expected_ending_top1_distinct_neurons']}개다.",
          '[읽기] 이 추가 집계는 더 좁은 확인이다. 고정한 S3 판정 규칙을 바꾸지 않았다.','',
          '[읽기] 이 판정은 짧은 문장의 다음 토큰에 관한 것이다. 생성 문장의 뜻과 말투 보존을 입증하지 않는다.',
          '[읽기] 반말과 존댓말의 정적 렌즈는 S1b에서 실패했다. 이번 검사는 문맥 안의 실제 개입 효과를 쟀다.','',
          f"[잰 값] 0층을 포함한 35층의 뉴런 337,920개를 모두 쟀다. 대조는 층마다 S1의 기존 방향 512개를 그대로 썼다.",
          f"[잰 값] 용량은 −0.8, −0.5, +0.5, +0.8이다. 뉴런 표는 {r['coverage']['neurons']:,}행이다. 대조 표는 {r['coverage']['controls']:,}행이다.",
          f"[잰 값] 문장 12개를 합쳐 {r['frame_evaluations']:,}회의 분포를 비교했다. 토큰을 샘플링하지 않았다.",'',
          '[읽기] 같은 어간 6개를 반말 문맥과 존댓말 문맥에 각각 넣었다. 프롬프트 끝은 캐릭터 답의 어간이다.',
          '[읽기] 어미 후보를 먼저 고정했다. 실제 GGUF 토크나이저에서 토큰 하나가 적어 둔 어미 문자열을 정확히 내보내는지 검사했다.',
          '[읽기] 처음에는 문장을 다시 토큰화할 때 어간과 합쳐지는 어미를 잘못 제외했다. 이미 입력된 어간 뒤에는 그 어미 토큰을 생성할 수 있다.',
          '[읽기] 0·1층과 일부 2층을 잰 뒤 이 구현 오류를 고쳤다. 효과 순위와 통과 개수는 보기 전이었다. 후보 어미 목록은 바꾸지 않았다.',
          '[읽기] 이전 규칙·부분 결과를 _superseded/token_audit_v1에 보존했다. 새 규칙과 해시를 먼저 고정하고 0층부터 다시 쟀다.',
          '[읽기] RULE.md의 변경 기록과 frames.json에 실제 토큰 ID와 디코딩 문자열을 남겼다.',
          '[읽기] 점수는 존댓말 어미의 로그 확률합에서 반말 어미의 로그 확률합을 뺀 값의 변화다. 12문장 평균으로 대조와 비교했다.',
          '[읽기] 최종 평균은 저장한 12개 변화값에서 배정밀도로 다시 계산했다. 반올림한 표시 열로 극값을 비교하지 않았다.',
          '[읽기] S3 판정에는 ±0.5만 썼다. 10/12문장 부호 일치와 11/12문장 한국어 토큰 조건을 함께 적용했다.','',
          '[읽기] 12문장은 별도의 검증 문장도 독립 표본도 아니다. 같은 어간과 문맥을 짝지어 썼다.',
          '[읽기] 뉴런 337,920개를 함께 검색했다. 대조의 최댓값을 넘었다는 사실만으로 통계적 유의성을 주장하지 않는다.',
          '[읽기] PASS는 S3 생성 검사를 시작할 선별 기준이다. 검증된 말투 조절 장치라는 뜻은 아니다.','',
          f"[잰 값] 엔진 검증에서 S1의 40문장 top-1은 {v['top1_agreement']}/40이 같았다. top-10 평균 겹침은 {v['mean_top10_overlap']:.4f}/10이었다.",
          f"[잰 값] 전체 어휘의 최대 로짓 차이는 {v['max_abs_diff']:.8f}다. 수치가 완전히 같지는 않다.",
          f"[잰 값] S2 문장 12개의 무개입 top-1은 CPU와 {sum(x['top1_equal'] for x in v['frames'])}/12가 같았다. 이 비교의 최대 로짓 차이는 {max(x['max_abs_diff'] for x in v['frames']):.8f}다.",
          f"[잰 값] 첫 S2 문장의 전체 위치 개입 10건은 CPU와 top-1이 {sum(x['top1_equal'] for x in pushed_cpu)}/10 같았다. 최대 로짓 차이는 {max(x['max_abs_diff'] for x in pushed_cpu):.8f}다.",
          '[읽기] S1과 같은 CPU 라이브러리와 실행 코드를 기준으로 삼았다. 검증 실패 버전과 로그도 보존했다.',
          '[읽기] PyTorch와 Triton으로 GGUF의 텍스트 순전파를 구현했다. 새 모델 가중치나 Hub 설정은 받지 않았다.',
          '[읽기] 가중치는 원본 GGUF에서 역양자화했다. CPU의 Q8_K/bf16 활성 반올림과 fp16 KV·어텐션 누적을 재현했다.',
          '[읽기] S1 CPU의 GELU 표를 읽어 썼다. 행렬곱은 배치 크기가 바뀌어도 누적 순서가 같은 TF32x3 커널이다.',
          '[읽기] 이 수치 규칙을 모든 입력 위치에 고정했다. 긴 prefill에서 CPU가 BLAS 경로로 바뀌는 것까지 같다고 주장하지 않는다.',
          '[읽기] CPU와의 개입 비교, 캐시 0 밀기, 전체 입력과 캐시의 비교는 engine_validation.json에 있다.','',
          '[읽기] 밀기는 post-FFN RMSNorm 뒤다. FFN 잔차, PLE 잔차, 층 출력 배율도 지난 l_out에 더한다. S1 sanity와 같은 위치다.',
          '[읽기] 크기는 S1의 층별 잔차 노름 중앙값에 용량을 곱했다. W_down의 해당 열과 대조를 각각 단위 길이로 만들었다.',
          '[읽기] 선별은 마지막 입력 위치만 민다. 앞부분의 KV와 선택 층 출력은 재사용했다.','',
          f"[잰 값] 전체 선별의 층별 실행 시간 합은 {r['screen_layer_seconds']/3600:.3f}시간이다.",
          f"[잰 값] 전체 위치 검증은 {a['seconds']/60:.2f}분이었다.",
          '[읽기] 선별 전의 실측 예측은 logs/benchmark_batch*.json에 있다. 층마다 GPU 메모리를 확인하고 체크포인트를 남겼다.',
          '[읽기] runtime의 프로세스별 메모리가 불완전해 GPU 전체 사용량에서 이 프로세스의 예약량을 뺐다. 외부 사용량이 3GiB를 넘으면 GPU를 비우고 기다리도록 했다.','',
          '[잰 값] 가장 강한 뉴런 50개와 무작위 뉴런 50개에서 모든 입력 위치를 함께 밀었다.',
          f"[잰 값] 전체 효과의 Pearson 상관은 {a['correlations']['all']['pooled']['pearson']:.5f}다. Spearman 상관은 {a['correlations']['all']['pooled']['spearman']:.5f}다.",
          f"[잰 값] 효과 부호 일치율은 {a['correlations']['all']['pooled']['sign_agreement']*100:.2f}%다.",
          f"[잰 값] 상위 50개의 Pearson 상관은 {a['correlations']['strongest']['pooled']['pearson']:.5f}다. 무작위 50개는 {a['correlations']['random']['pooled']['pearson']:.5f}다.",
          f"[잰 값] 0–13층 표본은 {sum(x['layer']<14 for x in a['selection']['neurons'])}개다. Pearson 상관은 {a['correlations']['early_layers_0_to_13']['pooled']['pearson']:.5f}이고, 부호 일치율은 {a['correlations']['early_layers_0_to_13']['pooled']['sign_agreement']*100:.2f}%다.",
          f"[잰 값] 14–34층 표본 {sum(x['layer']>=14 for x in a['selection']['neurons'])}개에서는 두 방식의 효과가 정확히 같았다.",
          '[읽기] 집단별·용량별 상관과 각 문장의 짝지은 변화는 validation_allpos.json에 있다.',
          '[읽기] 15층부터는 13·14층의 KV를 재사용한다. 따라서 14층 이상을 민 경우 다른 입력 위치의 변화가 마지막 위치로 전파되지 않는다.',
          '[읽기] 이 구간의 두 밀기 방식은 구조적으로 같다. 높은 전체 상관만으로 앞층에도 마지막 위치 선별이 유효하다고 읽으면 안 된다.',
          '[읽기] 이를 구별하려고 0–13층과 14–34층의 상관도 따로 남겼다.','',
          '[읽기] 손상 지표는 전체 어휘의 KL(pushed || base), 엔트로피 변화, top-1의 한국어 문자 검사다.',
          '[읽기] 한국어 문자가 남아도 문법과 뜻이 망가질 수 있다. 사람의 문맥 타당성 판정과 생성 검사는 아직 하지 않았다.',
          '[읽기] 큰 말투 점수와 낮은 손상은 다르다. 예를 들어 L34 N6703의 +0.5는 평균 KL이 약 29.33이고, top-1이 주로 문제입니다가 됐다.',
          f"[읽기] S3_review_candidates.md는 더 엄격한 어미 조건을 통과한 {r['stricter_controls_and_11_expected_ending_top1_distinct_neurons']}개를 KL 순으로 정리한 추가 읽기다. 결과 뒤에 한 정렬이며 주 기준은 바꾸지 않았다.",
          '[읽기] compute_summary.json에 설치·초기 검증·폐기 실행까지 포함한 관측 시간 구간을 남겼다. GPU 사용률을 적분한 시간은 아니다.',
          '[읽기] S1 중앙값은 짧은 40문장에서 잰 값이다. 긴 대화, LoRA, 다른 기기와 다른 양자화는 검증하지 않았다.','',
          '[읽기] top_register.md에 방향별 상위 30개와 모든 문장의 top-1을 적었다. L26 N6893, L27 N1939, L28 N3794, L34 N6703도 네 용량 모두 적었다.',
          '[읽기] screen.tsv.gz와 controls.tsv.gz는 한 행에 12문장의 배열을 담는다. 배열 순서는 frames.json 순서다.',
          '[읽기] 층과 뉴런 번호는 0부터 센다. controls.tsv.gz의 neuron 열은 대조 방향 번호다.',
          '[읽기] layers/에는 재개용 압축 표와 해시가 있다. RULE, 입력, 엔진 해시가 다르면 재개하지 않는다.',
          '[읽기] 설치는 별도 [configured environment]에만 했다. [unrelated directory]는 바꾸지 않았다. 모델 원본과 앱도 수정하지 않았다.',
          '[읽기] S1 엔진의 헤더와 수치 연산 소스만 읽어 이 폴더에 복사했다. 경로와 해시를 logs/에 남겼다.',
          '[읽기] 파일을 지우지 않았다. 이전 구현과 실패 결과는 _superseded/에 보존했다. git은 쓰지 않았다.',
          '[읽기] 네트워크는 노트북의 ssh/scp와 그 실행 환경의 PyPI 설치에만 썼다.','',
          '재실행: tools/prepare.py → tools/build_reference.py → tools/reference → 입력 복사 → tools/validate_engine.py → tools/run_screen.py benchmark → tools/run_screen.py screen → tools/postprocess.py analyze → tools/validate_allpos.py → tools/postprocess.py report → tools/provenance.py.',
          '기존 파일은 먼저 _superseded/로 옮겨야 한다. venv와 GGUF는 S2 밖에 둔다.']
    errors={k:max(x['max_abs_errors'][k] for x in a['independent_float64_metric_checks']) for k in ('shift','kl','entropy_change')}
    rows.extend(['',f"[잰 값] CPU 배정밀도 검산의 최대 차이는 말투 변화 {errors['shift']:.3g}, KL {errors['kl']:.3g}, 엔트로피 변화 {errors['entropy_change']:.3g}다."])
    if (OUT/'output_validation.json').exists():
        audit=load(OUT/'output_validation.json');assert audit['all_passed']
        rows.extend(['[잰 값] 전체 표의 범위, 전역 순위, 통과 개수를 별도 코드로 다시 계산했다. output_validation.json의 all_passed는 true다.',
                     f"[잰 값] 선택한 100개 뉴런의 선별 효과 재계산 차이는 최대 {audit['screen_replay_max_abs_shift_diff']:.3g}이다."])
    if (OUT/'compute_summary.json').exists():
        budget=load(OUT/'compute_summary.json')
        rows.append(f"[잰 값] 첫 GPU 확인부터 계산 파이프라인 완료까지의 관측 구간은 {budget['observed_full_session_wall_envelope_hours']:.3f}시간이다. 설치와 CPU 작업도 포함한다. GPU 대기는 {budget['final_run_gpu_pause_events']}회였다.")
    with (OUT/'README_ko.md').open('x') as f:f.write('\n'.join(rows)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['analyze','report']);a=p.parse_args()
    analyze() if a.mode=='analyze' else report()
