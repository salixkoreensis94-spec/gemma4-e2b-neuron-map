# M3 탐색 addendum — L5 / L27 (2026-11-01)

이 문서는 사전등록 경로가 중단된 뒤의 **post hoc 범위 변경**이다. Jiho의 11-01 선택을 그대로 기록한다: **[reviewer-approved scope; original wording withheld]**. L15 k64 final은 KL_TC=0.06408134548255329, KL_zero=2.4989452720482466, KL score=0.9743566431008593이나 dead=8173/16384=0.49884033203125로 amended gate FAIL이다. k64도 실패하여 원 RULE의 후속 심사 조건인 세 층 통과를 충족하지 못했다. 원 CE 세 층 FAIL과 사전등록 M3 경로 중단을 영구 보존한다. 재학습/추가 튜닝은 없다.

변경은 후속 심사의 층 집합을 [5,15,27]에서 [5,27]로 줄이는 것뿐이다. RULE_M3.md (8cbfd05ee919708193c0cb1cab24200c0055b553503edbdc8c1d574ad73e74af), amend1, 두 RUN 및 frozen code를 변경하지 않는다. L5/L27은 최종 k32를 사용한다. M2의 표와 합치지 않는다. 결과는 전부 **exploratory**이며 primary claim이라는 표현/판정 필드는 사용하지 않는다. 층별 결과는 기술 통계다. M4는 시작 전에 별도 규칙을 동결하며 이번에는 시작하지 않는다.

## 표본과 동일 draw의 보존

각 층 final k32 heldout live(z>0인 적 있음) features 중120개, 해당 층 M2 main neurons40개, std-scaled Gaussian directions18개, 별도 feature donor 쌍18개: 층당196개, 두 층392개다. M2 뉴런에 새 심사를 하며 기존 심사 표를 재사용/합산하지 않는다. feature와 donor는 같은 층에서 모두 disjoint이고 donor도 중복되지 않는다. direction은 dense support이므로 support disjoint를 요구하지 않는다. TC pilot은 없다.

원 RULE은 층별 RNG 파생법을 더 세분화하지 않았으므로 **joint RNG를 원 층 순서 [5,15,27]로 소비**한다. 가상의 3층 final-k32 선택과 L5/L27 draw가 동일하도록 L15의 k32 live 집합을 사용한 draw와 Gaussian draw도 소비하되 L15 item을 실행/심사하지 않는다. 실패 L15를 다른 population으로 바꾸어 후반 RNG를 이동시키지 않는다.

- NumPy Generator(PCG64), seed2026103010: 오름차순 live feature IDs에서 층별 choice(120,replace=False).
- seed2026103011: 각 층 float64 standard_normal([16384,18]) G. 행/열 순서 그대로, 부호 선택/회전/최적화 없음.
- seed2026103012: 각 층 live IDs 중 선택 feature를 제외하고 choice(36,replace=False), 순서대로 두 개씩18 donor pairs.
- seed2026103013: 원 3층 순서에서 각 층 feature120, neuron40(M2 manifest 순서), direction18, mixed18로588 items를 만들고 permutation(588)의 위치로 opaque `T%04d` IDs. L15의 ID 자리는 비워 두고 나머지392개를 유지한다. M2의 I namespace와 구분한다.

선택 JSON과 원 G float64 파일의 SHA를 context capture 전에 동결한다. selection은 feature content로 교체하지 않는다. live mask/선택에 사용한 숫자 해시도 보관한다.

## 방향, reservoir, capture와 packet

M2와 동일한 dev 49,438,284 content tokens /263,910 chunks, 동일 순서/독립 BOS=2. teacher는 frozen d, TF32 off/autocast 없음. TC도 FP32이고 encoder 전체 폭의 TopK32를 계산한 뒤 필요한 feature를 읽는다. feature firing은 BOS를 제외한 z>0이다. 원 M1 seed20261028의8192 global positions에서 full TC z rows를 재계산하여 population std s를 얻는다. 이 단계는 선택 동결 후 시작한다. vector 정의는 M2와 동일: valid=s>=1e-6, invalid G=0, V[j,k]=float32(G[j,k]/(s[j]*sqrt(sum_valid G[:,k]^2))), invalid V=0. float64 정규화 후 한 번 cast, centering/whitening/추가 scale 없음. V와 std의 해시는 본 capture 전에 별도로 동결한다. projection은 FP32 z@V. 동일 row 재계산과 reservoir projection 허용오차는 M2와 동일 2e-5+2e-5*abs(reference); float64 대조는 gamma_width*sum(abs(z*V))+2e-5다. nonfinite/허용오차 초과는 중단한다.

각 selected feature/donor/direction의 count, histogram1024, total/positive/max, language/script positive counts와 exact distinct-chunk top20을 M1/M2 형식으로 수집한다. 순서는 activation 내림차순→작은 chunk ID→chunk 내 earliest content offset이며 BOS를 저장하지 않는다. fixed reservoir8192 rows도 저장한다. calibration full-z reservoir는 방향 정의/완료 대조용이다.

packet은 원 M2 `packets.build_packet` 코드 경로를 사용한다. top20 +4 bands×6=44 windows, ±16 tokens, firing token 표시. empirical rank bands [0,.5),[.5,.9),[.9,.99),[.99,1], stable activation/global-position 순서와 M2 seed2026102904/SeedSequence([seed,layer,local_item_id,band]). layer 내 local ID 공간은 feature f, neuron16384+n, direction32768+d, mixed32786+m로 분리한다. mixed donor0/1 교대, top10+10, 각 band3+3. 모든 종류에 M0 whole-chunk PII + E `pii_boundary.py` index를 먼저 적용하고, family cap2/Jaccard.70/중복 chunk 제외/원 reservoir fallback을 동일 적용한다. drop/fallback은 audit하며 INCOMPLETE면 중단한다. 원문을 log/report/transcript에 출력하지 않는다. 실제 packet은 capture DONE 후 다음 resume에서만 쓴다. 이번 smoke builder 검사는 RAM에서만 만든다.

심사원은 Claude Opus5.5, GPT gpt-6.1-sol high, Gemini4 Argon High이고 M2 prompt/JSON schema/44창 packet을 그대로 쓴다. Jiho의 이번 선택은 이392개×3명의 judge usage 승인이다. 이번 resume에는 judge 호출이 없다. batch size6, 고정 judge shuffle seeds는 M2와 동일하다.

## 탐색 통계 — 정의/threshold는 원 RULE 유지

L5/L27을 pooled하여 one-concept a / 모든 planned item을 feature 대 neuron으로 비교하고 Wilson/Newcombe independent95 intervals(z=1.959963984540054)를 쓴다. unknown/missing/u도 planned 분모에 남긴다. feature-minus-neuron 차이 하한>0이라는 원 수치 bar는 탐색 지표로만 표시하고 primary claim을 만들지 않는다. secondary p=b/(b+c), feature-minus-neuron Newcombe95도 함께 보고한다. 3명 모두 valid여야 exact-category majority>=2/3, missing/invalid 또는1/1/1은 unknown. 만장일치 민감도, Fleiss 및 짝별 κ, missingness/usage를 보고한다.

control validity는 feature p−direction p Newcombe95 하한>0 AND mixed c/전체 planned36>=.80. Wilson interval도 보고하되 mixed bar는 점추정 기준이다. 분모0은 undefined, 유리하게 제거하지 않는다. per-layer/per-kind는 descriptive; M2/pilot/실패 L15와 pooling하지 않는다. 통계/threshold를 결과에 따라 바꾸지 않는다.

## 실행과 보관

원 supervisor/pid identity, HOLD/DONE/BLOCKED, Steam watch를 사용한다. 외부 GPU>3072MiB 또는 telemetry 실패 때 commit 후 **worker exit로 VRAM 해제**. supervisor는 total<2500MiB 연속300s 후 재시작한다. Xbox/non-Steam game도 GPU guard가 감지한다. C: 저장 전15.1GB+payload reserve를 요구하고 공간 부족이면 BLOCKED로 중단한다. 신규≤1M tokens마다/SIGTERM/guard/끝에 SHA+COMMIT-last, RNG 없이 결정론적 stream cursor와 reservoir prefix를 복원한다. 과거 checkpoint를 합하지 않는다. 삭제/다운로드/Git/게시/Mac inference/심사 호출은 없다.

smoke는 dev의 고정 prefix, 실제 SIGTERM commit 뒤 재개와 uninterrupted run의 모든 top list/누적 fields/reservoir prefix가 exact 같은지 검사하고 throughput을 측정한다. 완료 전 body가 없는 partial checkpoint는 보관하고 무시한다. full capture는 smoke 후 같은 frozen selection/vector/engine으로 새 root에서 시작한다. 실제 feature/context capture 전에 이 addendum의 UTC/KST/hash를 `resume_d/EXPLORE_FREEZE.json`에 기록한다.
