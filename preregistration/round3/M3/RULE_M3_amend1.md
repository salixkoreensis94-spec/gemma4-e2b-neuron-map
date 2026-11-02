# M3 amend1 — KL fidelity 사후 개정 (2026-10-31)

## 지위와 범위

이것은 사전등록 결과를 본 **뒤에** 정한 post hoc amendment다. Jiho의 2026-10-31 선택은 [reviewer-approved scope; original wording withheld]이며 parent M3c brief로 전달됐다. 기존 RULE_M3.md sha256 `8cbfd05ee919708193c0cb1cab24200c0055b553503edbdc8c1d574ad73e74af`, RUN.md, evaluate.py는 변경하지 않는다.

사전등록된 원 CE gate는 L5/L15/L27 모두 FAIL로 영구 보존한다. M3b 최종 원 mean-minus-clean 분모는 각각 -0.123523541, -0.268860922, -0.142892206 nats다. BOS 보존 arm도 각각 -0.083386117, -0.266199327, -0.132773884였다. L15 dead=0.27032470703125도 원 dead bar를 넘었다. 이후 모든 요약은 원 frozen FAIL과 amended 판정을 함께 표시한다. M2의 H 기각은 바꾸지 않는다.

바꾸는 것은 RULE §B gate의 fidelity 반쪽(CE loss recovered≥.80)뿐이다. §A의 tensor/모델/학습, width/seed/data/optimizer/schedule, dead≤.10, 단 한 번 k64 retry, C/D/E와 후속 심사 정의는 유지한다. 단, parent가 명시한 대로 실패층은 같은 forward에서 함께 학습하고 가장 깊은 실패층 raw MLP에서 forward를 끝낼 수 있다.

## KL 정의와 gate

full heldout 561,716 targets, 2,952 chunks, chunk마다 독립 BOS=2. 원 RULE §B의 content-token next-target 위치만 평가하고 chunk 밖 예측은 제외한다. **BOS 위치의 첫 content 예측도 포함**한다. 모든 arm에서 BOS의 raw mlp_output은 true y[:,0]을 유지하므로 그 위치 KL은 0이다. 제외한다면 각 arm 평균에 같은 배율만 적용되어 비율은 변하지 않는다.

층별 단일 intervention: raw `mlp_output`의 content positions만 (a) TC yhat 또는 (b) zero=0으로 교체한다. clean/원 y를 나머지 위치와 층에 유지한다. post-FFN RMSNorm, residual, PLE, layer_output_scale, 나머지 35층과 최종 output norm/embedding/softcap은 원 d 그대로다. TF32 off, autocast 없음, model/TC FP32. 각 위치의 full vocabulary 262,144개 **최종 softcapped logits**에서

`lp=log_softmax(clean_logits); lq=log_softmax(arm_logits)`

`KL_t(clean||arm)=sum_v exp(lp_v)*(lp_v-lq_v)` nats.

log_softmax, entropy, KL 및 token 합은 **float64**로 계산한다. logits는 원 FP32 값을 float64로 승격한다. full vocabulary를 생략/상위 토큰만 사용하지 않는다. 숫자를 유리하게 clamp하지 않는다. 동일 logits KL=0을 검사하며, 유한성/의미 있는 음수(<-1e-10)가 발생하면 구현 검사 FAIL로 중단한다. 아주 작은 반올림 음수도 원값을 보존한다. heldout KL_arm은 token-weighted mean, `KL_score=1-KL_TC/KL_zero`.

층별 amended gate: finite KL_zero>0 AND finite KL_score≥.80 AND dead≤.10. KL_zero≤0 또는 nonfinite이면 FAIL. dead는 clean 입력에서 TC가 heldout content positions에 한 번도 z>0인 적 없는 feature의 비율이며 BOS 제외. 후속 심사는 세 층이 모두 amended gate를 통과해야 한다.

## 사전 코드 검사와 기록

heldout KL 전에 새 `tools/evaluate_kl.py`를 CPU 합성분포로 검증한다: identical logits→0; p=(.5,.5),q=(.75,.25)의 KL=.5*ln(4/3); 비대칭 분포 p=(.25,.75),q=(.5,.5)의 직접합과 방향; 상수 logit shift 불변성. float64 오차≤1e-12. 원 evaluate.py는 유지한다. amendment hash와 UTC/KST freeze 시각을 `resume_c/AMENDMENT_FREEZE.json`에 먼저 기록한다. 첫 heldout KL 실행은 이 기록과 코드 test PASS가 있어야 한다.

## retry와 D

어느 반쪽이든 실패한 층은 원 §B의 단 한 번 **k=64** retry를 받는다. width16384, 초기화 seed2026103003, dev 49,438,284 tokens/263,910 chunks와 frozen shuffled order, buffer8192/minibatch1024/PCG64 seed2026103005, calibration 첫8192 dev content tokens, Adam/lr/clip/decoder constraints/AuxK는 그대로다. scratch부터 시작하고 k32 가중치를 이어 쓰지 않는다. 실패층만 한 forward에서 읽고 가장 깊은 실패층 MLP 뒤 종료, 같은 위치의 minibatch를 공유한다. k64도 같은 amended gate로 평가하며 실패하면 중단/보고, 추가 튜닝은 없다.

k32 통과층의 D를 retry **전에** 실행한다. 원 §D의 heldout float64 Pearson과 `max(r,0)*z/std(z)` association, top20 원 chunk/offset, NONE/tie, c의≥16/20 후보를 그대로 사용한다. a/b/c/unknown별 기술 통계와 C의22개 후보와의 cross-tab만 보고하고 인과 주장하지 않는다. 실패층 D는 retry 판정 후 다음 resume다. feature capture, packet, judge 호출은 이번에 하지 않는다.

## 기술 통계 (gate/선택에 사용하지 않음)

최종 k32 CE(clean/TC/mean/zero, BOS replaced/kept)는 M3b의 기존 값들을 재사용한다. full과 stratified sample의 측정 범위를 명확히 나누며 없는 full zero 값을 만들어 내지 않는다. KL(clean||heldout raw-mean-y)도 BOS 보존으로 측정한다. clean 평균 entropy와 clean CE를 source 및 target content offset [0],[1..15],[16..63],[64..127],[128..255]별 token-weighted로 보고한다. 원 CE recovered는 원 분모가 양수인 경우에만 기술 통계로 계산하고, 아니면 None/FAIL 기록을 유지한다. retry 종료에는 FVU/dead/dense/L0, amended KL gate와 이 CE 값들을 자동 기록한다.

## 실행·복구

평가 및 D는 per-chunk numeric commit으로 resume하고 완료 chunk를 중복 합산하지 않는다. train은 ≤1M 신규 forwarded tokens, SIGTERM/guard/끝에서 누적 checkpoint(payload SHA+COMMIT last)를 저장한다. RNG/buffer/calibration/cursor/Adam/parameters 및 trained count를 복원한다. source/input/frozen-rule hash와 cursor accounting을 검증한다.

M3와 같은 supervisor/pid identity/Steam watch, HOLD/DONE/BLOCKED. external GPU>3072MiB 또는 telemetry 실패면 저장 후 worker exit로 VRAM 해제. total<2500MiB 연속300초 후 supervisor restart, Steam registry poll5초/game SIGTERM/idle120초. C: free≥15.1GB gate. 새로운 laptop 파일은 round3 아래, 원 파일/이전 결과는 삭제하지 않고 _superseded로 옮긴다. 원문/feature context는 보고서·로그·모델에 출력하지 않는다.
