# M3 사전 규칙 — 2026-10-30

이 파일은 특징 내용, 학습 결과, 형태 점수를 보기 전에 고정한다. M2의 H 기각은 바꾸지 않는다. 대상은 L5/L15/L27(0부터), GGUF와 FP32 엔진 d는 M1과 동일하다. TF32 off, autocast 없음. 모델은 동결한다.

## A. skip TopK transcoder

입력 x는 d의 `emit(l,'ffn_norm', norm(h_after_attention, blk.l.ffn_norm.weight))`, 마지막 차원 1536이다. 타깃 y는 `F.linear(GELU_tanh(ffn_gate(x))*ffn_up(x), blk.l.ffn_down.weight)` 직후, 새 읽기 hook `mlp_output`의 1536차원 값이다. `ffn_norm`과 `mlp_output`을 같은 forward에서 읽는다. BOS는 학습/통계에서 제외한다. 학습 forward는 L27의 raw MLP output에서 끝낸다.

교체는 raw y 한 곳이다. `h_after_attention + norm(y, post_ffw_norm.weight)`의 post-FFN norm과 residual, 이후 `inp_gate`/PLE projection/post_norm 경로와 `layer_output_scale.weight`는 모두 원래대로 실행한다. transcoder가 PLE, post-FFN norm, layer_output_scale 자체를 재구성하지 않는다. CE는 교체층 뒤 모든 35층을 실행한다.

각 층 독립: 폭 m=16384, 기본 k=32. 첫 8192 dev content positions의 x/y 평균과 scalar population RMS를 calibration 상수로 쓴다(타깃 mean-subtracted RMS, x도 동일). calibration 토큰은 버퍼에 보관하고 한 번씩 학습하므로 dev 한 pass에서 빠지거나 중복되지 않는다. s의 하한은 1e-6.

`u=(x-mu_x)/s_x; q=ReLU(W_enc u+b_enc); z=TopK_k(q); yhat=mu_y+s_y*(W_skip u+b_out+z D)`.

TopK는 양수 중 큰 k개, 같은 값이면 feature ID가 작은 쪽을 우선한다. 양수 아닌 선택값은 0이다. D는 [16384,1536], 각 decoder 행은 unit L2 norm. loss는 normalized-target MSE + (1/32)*AuxK MSE. AuxK: train에서 1,000,000 content tokens 동안 안 켜진 feature(처음 250,000 tokens 이후 미발화도 포함)에 한정한 top-256 양수 encoder 값으로 main residual을 재구성한다. residual은 detach, 유효 dead 수가 256보다 적으면 그 수만 사용, 없으면 AuxK=0. AuxK 경로는 encoder/decoder만 갱신한다.

초기화 seed 2026103003, Torch CPU generator: Gaussian decoder를 unit normalize, encoder=0.01*D, encoder bias/out bias/skip=0. Adam(lr=3e-4, betas=.9/.999, eps=1e-8, weight_decay=0), gradient L2 clip=1, decoder gradient의 행 방향 성분 제거 후 step, step 뒤 unit normalize. LR은 content 학습 progress 기준 첫 1% 선형 warmup(최저 1% lr), 이후 cosine으로 마지막 10% lr. FP32 parameters/gradients/optimizer; matmul FP32. minibatch 1024 positions, 마지막 잔량도 학습한다. 3층 loss/optimizer는 독립이고 하나의 model forward를 공유한다.

## B. 데이터, 재시작, 측정

dev는 M1의 49,438,284 content tokens/263,910 chunks, heldout는 561,716 tokens. 학습 chunk 순서는 NumPy PCG64 seed 2026103004의 dev chunk permutation, shuffle buffer는 층별 x/y에 동일한 8192-position CPU buffer. 버퍼가 찰 때 동일 RNG로 1024개 비복원 draw하여 학습/제거하고 refill, 끝의 잔량도 무작위 draw. BOS 뒤 content만 쓴다. calibration도 첫 shuffled positions에서 읽는다. chunk permutation/버퍼 RNG는 별도 PCG64(seed 2026103005). 모든 RNG/cursor/buffer/model/optimizer/calibration/학습 토큰 수를 checkpoint에 저장한다.

checkpoint는 신규 forward content 1,000,000 tokens를 넘기기 전에, SIGTERM/guard 때, 마지막에 commit한다. payload fsync+SHA256 후 COMMIT 마지막; 복원시 모든 hash 확인. 누적 checkpoint를 합하지 않는다. incomplete 디렉터리는 보관하고 무시한다. 데이터/모델/코드 hash를 FREEZE에서 확인한다. pid/boot/start identity와 DONE, supervisor 재사용. 외부 VRAM>3072 MiB 또는 telemetry 실패면 commit 후 worker exit(전체 VRAM 해제). 재시작은 total VRAM<2500 MiB가 연속 300s, poll10s. Steam RunningAppID game watch는 5s마다 확인, game이면 supervisor SIGTERM, idle120s 뒤 launch; HOLD/DONE/BLOCKED 준수. C:는 checkpoint 전 15.1GB gate. source와 옛 checkpoint는 지우지 않는다.

smoke는 약 1M dev tokens, 별도 run root에서 실시, full 학습은 동일 seed로 새로 시작한다. clean y identity splice의 heldout 20 chunks CE/logits는 bit-identical이어야 한다. smoke normalized MSE는 초반 100 updates 대비 마지막 100 updates 평균이 낮아야 한다. checkpoint restore의 동일 next batch에 대해 model/optimizer/RNG/buffer 및 다음 update가 exact 같아야 한다. CPU fake-worker pause/exit/restart도 검사한다.

smoke 실측 forward+train+save throughput으로 49,438,284-token train 시간을 투영한다. >30 GPU h면 `N=max n such that prefix_tokens(n)/smoke_tps <= 30*3600`으로 앞 N shuffled chunks만 쓴다. 이 경우 이 파일의 예산 필드만 실행 전 채운 뒤 이전본을 _superseded에 보존하고 새 hash를 기록한다. 그 외 정의 변경 금지. budget: FULL unless the preceding mechanical cap is triggered. heldout metrics overhead와 guard waits는 별도 보고한다.

5M forwarded tokens(첫 초과 chunk 직후)와 마지막에 전체 heldout만 평가하고 immutable status JSON을 기록한다. 5M은 중간 상태, 최종 gate만 판정한다. 각 층 FVU=sum||y-yhat||² / sum||y-mean_heldout(y)||². heldout mean은 clean heldout pass에서 구하고 학습에 사용하지 않는다. CE는 각 chunk 독립 BOS, content token을 다음 토큰 target으로 하여 BOS→첫 content도 포함, chunk 밖 예측은 제외; content token 수로 가중한다. clean CE, 해당 층 TC splice CE, 해당 층 heldout-mean-y ablation CE. recovered=(CE_mean-CE_TC)/(CE_mean-CE_clean), denominator<=0이면 gate FAIL. dead=heldout 동안 z>0인 적 없는 비율, dense=heldout >10% positions에서 z>0 비율, L0의 min/p01/p10/p50/p90/p99/max/mean 기록. heldout norm/means는 평가 전용이다.

층마다 recovered>=.80 AND dead<=.10이어야 후속 심사에 간다. 실패층은 단 한 번, 같은 width/seed/data/optimizer/schedule에 k=64만 바꿔 처음부터 다시 학습하는 대안을 허용한다. 자동 튜닝/그 외 대안 없음. k64도 실패면 중단/보고. 이번 brief는 학습 launch에서 보고/종료하므로 최종 대안 실행은 다음 resume에서 이 규칙대로 처리한다.

## 후속 심사 — 이번에는 실행하지 않음

최종 heldout live feature 중 층당120개를 PCG64 seed2026103010으로 균일 비복원 선택. 비교군은 각 층 M2 main neurons40개, 동일 3심사원으로 새 판정. control은 층당18개 std-scaled Gaussian activation directions와18개 mixed features(별도 donors), M2 방식; pilot/main/donor disjoint, seeds2026103011/12/13. 문맥은 동일 dev search range에서 수집하며 실제 capture는 후속 brief. packet/prompt는 M2와 동일(top20+4×6,±16), 단 E의 boundary PII를 모든 종류에 공통 적용. feature/뉴런/controls 전부 같은 필터·family cap·fallback. 어떠한 feature 문맥도 이번에는 모델에 보내지 않는다.

주 비교는 3층 pooled feature-minus-neuron **one-concept a share**, 모든 planned item을 분모로 하고 unknown도 남김. Newcombe independent-Wilson95 차이 하한>0을 primary improvement bar로 고정한다. p=b/(b+c)의 feature-minus-neuron Newcombe95는 사전 지정 secondary(방향 기대: 더 깨끗하면 p가 낮아짐); one-concept만 유리하게 보고하지 않고 함께 보고한다. Wilson z=1.959963984540054. M2와 동일 3명 exact-category majority≥2/3, all3 valid 필수, unanimity sensitivity, κ/missingness. control validity는 feature p - direction p Newcombe95 하한>0 AND mixed c/all planned>=.80. primary claim은 위 control bars와 학습 gate도 모두 통과할 때만 한다. 층별 결과는 descriptive, M2 기존 표와 pooled하지 않는다.

## C. 모델 심사 없는 형태 점수

M2 main 1820×44 windows의 원 private manifest/corpus를 읽어 firing/previous/next token ID를 복원한다. previous/next는 같은 document의 이어지는 청크까지 찾고, document 경계는 -1 sentinel. sentinel은 44 분모에 남기되 집중도 최빈 후보 및 lens hit에서 제외한다. score1=max firing-ID count/44; score2=max valid next-ID count/44; score3=sum(next ID in lens top50)/44.

뉴런 top50은 M1 join 그대로. directions write=W_down @ M2 frozen v. mixed write=unit((donor write1+donor write2)/2); donor를 미리 unit normalize하지 않는다. 동일 S1b lens: a=output_norm.weight*write/sqrt(mean(write²)+eps), scores=E@a pre-softcap, top50 descending score then smaller token ID. 모델 inference 아닌 정적 가중치 행렬 계산이며 laptop CPU에서 한다. neuron lens join과 보조 직접 계산의 top50 집합도 표본으로 비교한다.

방향 null: L0–14의 main90 directions, L15–34의 main120 directions 각각에서 score별 np.quantile(.95,method='linear'). 초과는 strict >, threshold equality 제외. kind×majority category a/b/c(unknown 별도)×layer band 및 pooled, 초과 비율 Wilson95. mixed에도 동일 direction threshold 적용, second-null 기술 통계. form-unified c-neuron=majority c AND (score1>threshold1 OR score3>threshold3). 후보에 key/scores/threshold/M2의 모든 judge concept fields 보존, 원문 없음. M2 H의 추가 검정 아님.

## D. 120뉴런의 split 기록

L5/15/27 M2 main40 neurons 각각, heldout 모든 content positions에서 neuron activation a_j와 TC nonnegative feature z_f의 Pearson correlation r_jf를 streaming float64 moments로 구한다. zero-variance pair는 r=0. 음수 correlation은 기여를 설명하는 양의 association으로 세지 않는다. 각 뉴런 top20 원 M1 distinct chunks를 재실행하여 저장 offset의 z에서 `score_f=max(r_jf,0)*z_f/std_heldout(z_f)` 최대를 dominant feature로 정한다. tie smaller feature ID; 모든 score<=0이면 NONE. numerator는 지배 feature의 window count, 분모20(NONE도 포함), NONE은 distinct-feature 수에서 제외하고 개수 별도. distinct dominant 수와 mode share, category a/b/c별 분포(unknown 별도) 보고. c이고 동일 feature가 >=16/20인 경우 hidden-commonality M4 후보로 사전 정의한다. 예측-인과 주장은 하지 않는다.

synthetic validation: 서로 독립 centered signals에서 a=positive coefficient*z0 + noise, window별 z0/다른 z가 알려진 경우 correlation 및 score argmax 정답을 검사; zero variance/NONE/tie도 검사한다. D의 실제 계산은 학습 완료 후 다음 resume, estimator 합성 검증은 지금 한다.

## E. boundary PII (과거 M2 불변)

기존 M0 whole-chunk patterns를 그대로 먼저 적용. 같은 document이며 document_token_offset이 연속인 직전 청크의 tail256 characters + current head256 characters에 동일 patterns 적용하되 match가 경계에 걸치거나 current head를 포함할 때 flag. 현재 chunk 첫32 characters에서 시작하는 digit-group fragment(ASCII digit groups joined by hyphen/en-dash/em-dash/dot, total>=7digits, >=1 separator)도 flag; 첫32에서 시작하면 전체 이어진 group을 읽어 판정. 현재 chunk를 통째로 reject한다. 인접 document를 잘못 연결하지 않는다. 다음 corpus chunk도 이어질 수 있으므로 future callers는 이 boundary check로 차단한 현재/이전 chunk를 pre-index하고 previous chunk도 보수적으로 차단한다. 개인정보 여부를 확정하는 분류기가 아니며 pattern 기반이다.

M2 main 80,080 windows는 변경/재판정 없이 would-reject counts만 kind/source별 보고. 알려진 chunk148365 offset14는 신규 flag 대상인지 검사. 향후 M3 packets/local viewers 및 M5 snippets는 이 adapter의 window gate를 사용해야 한다. 기존 M2 builder/packets/results는 byte-identical하게 둔다.
