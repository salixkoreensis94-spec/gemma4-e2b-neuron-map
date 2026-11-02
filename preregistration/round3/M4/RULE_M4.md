# M4 인과 검정 규칙 — 부모 검토용 초안, 미동결

2026-11-02. the reviewer 승인: [reviewer-approved scope; original wording withheld], [reviewer-approved scope; original wording withheld], [reviewer-approved scope; original wording withheld].
이 파일과 selection, 도구의 해시는 초안의 식별자일 뿐이다. 부모가 FREEZE.json을 만들기 전에는 연구 창에 대한 순전파,
활성 재현, P 발화 판정, attribution/effect/U 계산을 모두 금지한다. M4a 테스트는 명시적 합성 토큰 ID열만 쓴다.
M2의 H 기각과 M3 사전등록 경로 중단·L5/L27 사후 탐색 결과를 바꾸지 않는다. 심사원·재학습·새 다운로드는 없다.

## A. 엔진과 정확한 분해

엔진 d: M0/resume1/tools/fp32_reference.py, M0/resume3/tools/fast_d.py 및 동일 GGUF.
FP32, TF32 off, autocast 없음, F.gelu(approximate='tanh'), final_logit_softcapping 적용.
새 M4 model.py는 같은 연산 순서·shape를 유지하고 수치 hook과 한 위치 개입만 추가한다. 기저 파일은 변경하지 않는다.

토큰 임베딩 h0 = token_embd[token] × sqrt(D). D=1536. PLE 입력은 토큰별 per_layer_token_embd × sqrt(P)와
h0의 per_layer_model_proj / sqrt(D)를 per_layer_proj_norm한 값의 합 /sqrt(2)이다. 이는 모든 층에 한 번 준비된다.
각 층 순서는 다음과 같다. 모든 norm은 weight × x /sqrt(mean(x²)+eps)이다.

1. layer_input → attn_norm → Q/K/V, Q/K norm, RoPE, causal/sliding attention.
2. attn_context를 head별 o_proj slice로 투영한 합 = attn_projection.
   h_after_attention = layer_input + norm(attn_projection, post_attention_norm).
3. x = h_after_attention, 즉 MLP pre-norm 직전 residual. ffn_norm(x) → ffn_gate, ffn_up.
   a = GELU_tanh(gate) × up. raw mlp_output = a @ ffn_down.T.
   h_after_ffn = x + norm(raw, post_ffw_norm).
4. ple_gate = h_after_ffn @ inp_gate.T; ple_product = GELU_tanh(ple_gate) × pe[L];
   ple_projection = ple_product @ proj.T; ple_norm = norm(ple_projection, post_norm).
5. l_out = (h_after_ffn + ple_norm) × layer_output_scale.weight.

KV sharing은 후반 층에서 동일 attention 유형의 앞 층 K/V를 재사용한다. attention 확률과 K/V 생성은 base forward 그대로
계산한다. 성분은 원인들의 완전한 경로별 미분이 아니라 **해당 위치 residual의 가산 성분**이다. head의 actual context를
원자로 두므로 KV sharing과 attention softmax는 합의 정확성을 깨지 않는다. 별도 alt-up/router/stream mixing은 d에 없다.

x의 성분: embedding 1개; j<L의 actual ple_norm 각각; j≤L의 H개 head별 o_proj 각각; j<L의 MLP 출력 각각.
현재 층 L의 PLE는 MLP 뒤이므로 x에는 0이다. 각 raw attention head 투영에는 합산 attention_projection의 실제 RMS를
고정하고 post_attention_norm.weight를 곱한다. MLP 성분은 실제 raw output의 post_ffw_norm RMS/weight를 거친다.
PLE는 비선형 게이트를 지난 **actual ple_norm 전체**를 별도 원자로 둔다. 이를 초기 token PLE table 자체로 대체하지 않는다.
이전 모든 성분에는 해당 경로의 layer_output_scale을 차례로 곱한다. gate/up bias가 없는 것을 검사한다.
이 방식으로 sum(x_c)=x. pre-MLP RMS도 actual 값에 고정하면 xhat_c=x_c*rms_x*ffn_norm.weight이고
sum(g·xhat_c)=z_g, sum(u·xhat_c)=z_u다. FP32 합산 차이만 허용한다.

성분별 attribution = GELU'_tanh(z_g)*z_u*(g·xhat_c)+GELU_tanh(z_g)*(u·xhat_c).
real z_g/z_u는 full engine의 실제 `ffn_gate`/`ffn_up` hook 값을 쓴다(다른 shape의 dot으로 대체하지 않는다).
P는 actual xhat에 새 gate/up를 적용한 FP32 dot이다. GELU 값은 실제 F.gelu_tanh를 쓰고 미분만 해석식으로 계산한다.
GELU_tanh(z)=z/2*(1+tanh(sqrt(2/pi)*(z+.044715*z³))); 그 식을 해석적으로 미분한다.
원래 부호를 유지해 L2 정규화한다. 성분 좌표는 embedding, 층 오름차순 head 오름차순, MLP, PLE의 고정 ID로 대응한다.
이는 GELU의 Euler 합을 activation과 같다고 가정하지 않는다. gate/up **pre-activation 분해**가 exact인 것이다.

## B. 대상·창·선정

M2 main 분석 main_3judges_20261030.json의 majority와 main/private_manifest.jsonl을 조인한다.
c448/a631/b280. u/unknown은 제외한다. unit별 창은 해당 M2 packet의 첫 20개(slot=0) 그대로이며 fallback으로 들어간 창도
그대로 쓴다. context text, band 창, judge prompt는 사용하지 않는다. offset은 content 기준 0-based이고 자체 BOS 뒤 위치는
offset+1. 전체 chunk(최대256 content tokens)+BOS를 replay한다. ±16 window만 잘라 모델에 넣지 않는다.

검증 바닥: M2 main mixed210의 원래 두 donors, 교대 순서 10+10 top 창, donor 활성값을 사용한다.
보정 바닥: 각 층20쌍, 총700, NumPy Generator(PCG64(2026110201)), 층0..34/쌍0..19 순서.
M2 main/pilot 모든 neuron와 mixed donor를 제외한다. 각 층에서 available IDs 정렬 후 rng.choice(2,replace=False),
선정한 두 donor를 제거한다. 보정쌍끼리 donor도 겹치지 않게 해 바닥의 의존성을 줄인다.
각 donor의 M1 top20 순서에서 처음10개; 앞 donor와 chunk가 중복되면 다음 후보를 사용한다. 10개를 못 채우면 중단한다.
post-initial RNG state와 excluded/used IDs를 저장한다. 발화 실패 시에만 이 state에서 다음 쌍을 뽑고 available에서 제거한다.
교체는 layer/pair ID 순서, 다음 재시작에서도 동일 state/attempt 기록을 복구한다. 원래쌍과 대체쌍 모두 보존한다.
각 층 available 소진 시 중단; 임계값을 낮추지 않는다. M4a에서 P 발화는 계산하지 않는다.

trace: M2 또는 M3 majority-c L5/L27 합집합, L5N3862, form_unified_c_candidates 22개 합집합.
동일 뉴런이 여러 목록에 있으면 ID 하나에 reasons를 모으고 원래 packet 창 우선순위 M2→M3→M1을 쓴다.
L5/L27 밖은 성분 단위 기술만 한다. trace 38개에는 별도 trace ID를 주어 동일 창의 기술적 cluster를 계산하고, 주 집단
분모에는 중복 합산하지 않는다. selection/에는 ID/숫자/해시만 있고 원문은 없다.

## C. P와 발화

각 gate/up/down에 독립적으로 wP=(wX/||wX||+wY/||wY||)*(||wX||+||wY||)/2.
0 norm이면 실패 처리한다. P 활성은 actual xhat에서 GELU_tanh(gP·xhat)*(uP·xhat).
P의 새 raw MLP 주입은 aP*wP_down이며 기존 뉴런을 대체하거나 제거하지 않는다.
donor own activation 부호가 0이면 그 창은 발화 아님. aP*donor_activation>0이고
abs(aP)≥.25*abs(donor_activation)이면 발화. 각 절반 10개 중≥8개 필요.
보정 실패는 위 seeded 교체, 검증 실패는 제외하고 210을 분모로 제외율을 보고한다. 임계값 .25는 brief 그대로다.
V1은 P 자체가 아니라 각 창의 **donor 실제 활성**을 기록값과 비교한다.

## D. 아래쪽 효과

real: 한 firing position의 해당 z[neuron]만0, BOS·다른 토큰·다른 뉴런은 그대로. 모든 이후 연산은 실제 재계산한다.
total=logsoftmax(final softcapped base logits)-logsoftmax(ablated logits).
P: raw mlp_output에 aP*wP를 한 위치만 더한다. total=logsoftmax(with)-logsoftmax(base).
변화의 부호는 두 경우 모두 unit-on이다.

right-after-layer는 PLE와 layer_output_scale까지 끝난 l_out. base와 arm의 그 residual 차이를 위 부호에 맞춰 dL로 둔다.
이 dL에는 현재 층의 PLE 반응도 포함된다. 이후 층에서는 layer_output_scale.weight만 차례로 곱해 d_final을 만든다.
F_act(h)=softcap(unembedding(output_norm(h)))로 정의한다. output_norm은 **각 인자 h 자체의 실제 RMS**를 쓴다.
real direct=logsoftmax(F_act(h_base))-logsoftmax(F_act(h_base-d_final)).
real indirect=total-direct=logsoftmax(F_act(h_base-d_final))-logsoftmax(F_act(h_ablated)).
P direct=logsoftmax(F_act(h_base+d_final))-logsoftmax(F_act(h_base)).
P indirect=total-direct=logsoftmax(F_act(h_with))-logsoftmax(F_act(h_base+d_final)).
따라서 final norm 자체의 반응을 indirect에 넣지 않는다. indirect는 L 뒤 층이 더한 변화만 actual final norm을 통해 읽은 것이다.
수치 구현은 위 logsoftmax 두 항의 차이로 indirect를 계산하고 total=direct+indirect closure≤1e-12를 합성 검사한다.
indirect의 vocabulary 평균을 빼고 L2 정규화한다. 전체 vocabulary, FP64 logsoftmax/centering/Gram 사용.
L34는 downstream verdict/threshold/shares/CI에서 제외한다. upstream에는 포함한다.

## E. 통일도와 결정적 tie

20 normalized vectors의 cosine S, W=(S+1)/2. brief의 식 그대로 self-similarity 대각도 유지한다.
D=diag(W행합), normalized Laplacian I-D^-1/2 W D^-1/2.
float64 eigh 두 번째 고유값의 eigenspace(1e-10 이내)에서 canonical basis e0,e1,..의 첫 nonzero projection을 사용한다.
첫 유효 좌표를 양수로 맞춘다. Fiedler 값 12자리 반올림 후 (값,원래 창 순번) 정렬.
cut k=3..17에서 ncut=cut(A,B)/vol(A)+cut(A,B)/vol(B)를 최소화한다.
1e-12 이내 동률은 작은 k, 다음 창 ID 순서. U=두 군 사이 S 평균. mean pairwise S는 민감도만.
known donor split은 절대 clustering에 넣지 않는다. zero/nonfinite vector나 degree0은 해당 side invalid이며 임의0/1로 대체하지 않는다.
한 invalid unit이라도 있으면 해당 side는 읽지 않고 부모에게 보고한다(추가 자료 의존 제외 없음).

## F. 문턱·표준화·통계

bands L0–6/L7–13/L14–20/L21–27/L28–34. side별 τ=그 band의 accepted calibration U NumPy percentile(95,method='linear').
shared는 엄격히 U>τ. downstream은 calibration L34도 제외한다.
f_c=해당 side c의 비율. f_a/f0/b는 같은 side c의 band 구성비로 가중한다.
downstream에서 L34 제외 후 c의 mix도 다시 계산한다. 빈 필요한 stratum은 해당 side FAIL closed.
ρ=(f_c-f0)/(f_a-f0), 0..1 clipping하지 않는다.

bootstrap PCG64(2026110202), B10000. side마다 seed 재설정, population c,a,validation,b, band0..4 순.
threshold는 독립 calibration에서 한 번 정한 고정값으로 조건화한다. calibration threshold uncertainty를 포함한 CI로 부르지 않는다.
각 population-band 안 unit을 복원추출하는 shared-count Binomial(n,observed fraction) 등가 구현; band weights 고정.
ρ percentile95=[2.5,97.5], Bonferroni 두 side용97.5=[1.25,98.75]. f_a-f0의95% percentile도 같은 draws.
각 side에 f_c/f_a/f_0/f_b와 **동일한 bootstrap draws**에서 구한 각각의 95% percentile interval을 출력한다.
추가 RNG draw나 판정 조건은 없다. share_estimates의 f_c/f_a/f_0/f_b에는 value/CI95를, 기존 shares에는 기존 population 이름을 유지한다.
각 band에 c/a/b/validation/calibration/trace별 (n, shared)를 기술한다. n은 해당 side 범위에서 P 제외 후의 단위 수다.
planned/excluded/invalid_vectors도 함께 남기며 τ나 유효 벡터가 없어 shared를 정할 수 없으면 null이다.
calibration과 trace의 수는 기술 통계에만 쓰고 f_c/f_a/f_0/f_b 분모에 합치지 않는다.
bootstrap denominator≤0이 하나라도 있으면 그 side 읽지 않음(샘플 제거 금지).
97.5 lower>.5 숨은 공통점 지지; upper<.5 중첩; 그 외 inconclusive. 0.5 동률은 inconclusive.

## G. 검증과 중단

순서: parent freeze 확인→V1 replay 및 P eligibility/교체→V1 전체 gate→보정/검증/a/b/c/trace의 성분·효과 수집→
V2→calibration 문턱과 V3/V4→읽을 수 있는 side의 verdict→trace 군별 기술. U/진단 수치는 gate 이전에 저장할 수 있지만
gate 실패 side에 대한 통계적 해석·form 상관·지배 성분·trace 군별 해석은 출력하지 않는다.
V1 abs(actual-recorded)≤2e-5+2e-5*abs(recorded), 모든 사용/교체시도 창 중≥99%. 실패하면 전부 중단.
V2 gate/up 각각 abs(sum-actual)/max(abs(actual),1e-6)≤1e-3, 둘 다 통과한 창≥99%. upstream만 읽지 않음.
V3 side별 standardized f0≤.10 AND validation exclusion/210≤.20. V4 side별95% lower(fa-f0)≥.20.
V3/V4 실패 side는 숫자·실패 이유만 쓰며 과학적 verdict 금지. 완결성·hash·nonfinite·오프셋·checkpoint 실패는 중단.
calibration replacement과 validation exclusion 이외의 데이터 의존 필터/선정은 없다.

## H. 기술 통계와 trace

band별ρ, bρ, c의 up/down2×2, M3 firing/next-token/lens form scores와 U의 Spearman(동률 평균순위),
shared-up c의 attribution 절댓값 합으로 가장 큰 class embedding+PLE/attention/MLP를 보고한다. 군별도 기술한다.
L27 trace: L5 raw MLP를 k32 TC의 my+sy*(skip(u)+out_bias+Σ z_f decoder_f)로 근사해,
각 feature sy*z_f*decoder_f를 base post_ffw RMS/weight와 L5..26 scale에 통과시킨 뒤 L27의 같은 gradient×input 식에 투영한다.
skip+mean+bias 및 reconstruction residual을 별도 성분으로 함께 남긴다. TC feature들만으로 정확히 합쳐진다고 주장하지 않는다.
PLE/attention의 향후 변화는 이 frozen residual path의 몫이 아니다.
L5 trace: base 대 ablated actual L27 ffn_norm에서 k32 TC encode 전체16384의 차이, 군별 feature ID/top abs 숫자.
기여0 동률은 작은 feature ID. 각 trace의 up/down cluster별 창 ID와 feature/성분 기여 평균 top50도 기록한다.
L5/L27 밖은 feature 추적 순전파를 추가하지 않고 성분 기록만 요약한다. L15 TC 사용 없음. raw context/lens token/이름은 저장하지 않는다.

## I. 실행·출력·저장

M4b 진입점 tools/start_detached.py --freeze M4/FREEZE.json. 부모의 freeze schema는
{parent_frozen:true,rule_sha256,selection_manifest_sha256,sha256:{M4 상대파일 또는 round3 상대파일:hash}}.
frozen map에는 RULE, selection SHA 목록과 모든 선정파일, 실행/통계/model/trace 코드, 엔진 d/c_dequant/bridge,
GGUF, M1 chunk index/token stream, TC state와 관련 M3 freeze를 포함한다. 도구 prepare_freeze.py는 초안 템플릿만 만든다.
parent_frozen false에서는 시작 못 한다. M4a는 detached study job을 시작하지 않는다.

run/checkpoints/<phase>/<unit>/COMMIT.json을 마지막에 쓰며 files SHA를 검증한다. 모든 immutable JSON은 pending 파일
fsync 후 atomic rename하고 디렉터리도 fsync한다. 중간에 죽은 pending은 _superseded/로 보존한다. unit마다 commit, 중단된 unit의 최대20창은
재계산한다. cumulative 합산 금지; ID별 정확히 하나. replay 수치와 inference-unit 결과를 phase별 분리한다.
재시작은 COMMIT가 있는 것만 읽고 미완 디렉터리는 _superseded/로 이동한다. RUN/freeze hash가 달라지면 중단.
supervisor pid/boot_id/start_ticks, worker pid, DONE/BLOCKED/HOLD, stdout에는 ID/count만.
GPU external>3072MiB 또는 telemetry 실패는 worker exit75로 전 VRAM 반환. total<2500MiB 연속300초 뒤 재시작.
GPU guard 문턱은 M3와 같다: external>3,072MiB에서 정지, GPU total<2,500MiB가 300초 유지되면 재시작.
Steam watch5초, idle120초 뒤 supervisor launch. non-Steam은 독립 VRAM guard가 처리한다. C free≥15.1GB gate, 실패65.
HOLD/SIGTERM도 같은 unit 경계 복구. game watch/worker는 FREEZE 확인 후만 실행한다.

최종 출력: window별 replay/split 진단·up vector, side Gram20×20, U/cluster, direct/indirect RMS, P eligibility/교체 내역,
ID별 gate 및 aggregate JSON(side별 f_c/f_a/f_0/f_b의 95% bootstrap interval, band별 population의 n/shared 포함).
추가 share interval과 counts는 기술 출력이며 verdict를 추가하거나 바꾸지 않는다. full-vocab down vectors는 unit 처리 중 RAM에만 보유, 파일로 누적하지 않는다.
unit commit 후 RAM을 재사용한다. upstream vectors와 Gram만 저장하여 약2GB 이내(추정), GPU base 가중치 약11GB,
host unit down행렬 약20×vocab×8 bytes. study chunk/token IDs는 일시적 RAM, 원문 decode·로그·저장 없음.
세부 추정은 합성 측정 기반 README에 기록한다. tolerance/seed/selection은 부모 freeze 뒤 바꾸지 않는다.
