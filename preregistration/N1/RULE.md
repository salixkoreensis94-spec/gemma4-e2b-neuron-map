[Release copy: private identities, package references and environment details redacted; original and copy hashes are recorded separately.]

# N1 센서 시험 — N0에서 동결

이 규칙은 probe를 학습하거나 held-out 예측/점수를 보기 전에 동결한다. N0에서는 probe 학습, steered generation,
held-out 성능 계산을 하지 않는다. N0의 무개입 greedy 생성은 엔진 검증 10문맥 × 4 adapter에만 쓴다.

## 데이터와 분할

R1=사용자/캐릭터의 사람·사건·사실 소유 혼동(그 사건이 누구 일인지 canon 혼동 포함),
R2=사람 캐릭터의 AI/도우미 정체성 이탈, R3=AI review v2 (다)의 상투적 AI 말투,
R4=근거 없는 이전 대화·기억·날짜·시간 주장. 양성은 그 문제가 있음이다.
라벨 신뢰도는 [creator] 원표시 > 독립 AI 두 판정 일치 > 명시적 결정 규칙. 원래 평점/메모/판정 필드와
변환 규칙을 보존한다. 전체 bad나 good 평점을 곧바로 모든 범주의 양성/음성으로 바꾸지 않는다.
범주 태그를 나중에 붙인 사람은 [creator]가 아니다. 단일 AI 추정 태그를 [creator] 라벨로 승격하지 않는다.
결정 규칙 라벨은 규칙 라벨 성능을 잰 것이므로 사람 라벨 성능과 따로 보고한다.

N0/data/split.json을 고정한다. seed=20261027. 대화/시나리오를 약 70/30 dev/heldout으로 나눈다.
한 대화의 모든 turn, 동일 시나리오의 arm·캐릭터·variant는 같은 쪽에 둔다. 같은 prompt/reply의 중복,
복제된 평가 사례는 연결성분으로 묶어 같은 쪽에 두고 같은 범주에서는 중복 제거한다. 그룹 경계와 식별자
생성법은 N0 도구와 split에 보존한다. heldout은 재분할하지 않는다. heldout 양성 <40이면 underpowered다.
실제 [creator] 봇/채팅과 섞인 파일 전체는 제외한다. 완전한 context가 없는 자료는 별도 격리하고 probe에 쓰지 않는다.
캐릭터, 원 생성 model/adapter와 현재 app adapter의 차이, greedy 여부, 고정 앱 문구/후처리를 보존한다.

## 잔차와 모델

같은 Q4_K_M GGUF와 명시된 GGUF LoRA를 쓴다. 현재 A는 v5-kto2-e2, bench 세 명은 v1-e2.
LoRA delta는 원래 입력에서 scale*alpha/rank*B(Ax), alpha=0이면 scale*B(Ax).
층 번호는 0-based {2,6,10,14,18,22,26,30,34}. l_out은 attention·FFN·PLE 잔차 및 layer_output_scale 뒤다.
정확한 prompt+원 reply를 teacher-force한다. EOS/EOG와 app가 붙인 고정 접두사는 reply 통계에서 제외하고,
그 접두사는 conditioning에 넣는다. reply 표면 글자가 raw 생성과 다르면 그 사실을 명시한다.
reply의 각 토큰을 입력한 뒤 그 위치의 l_out을 읽는다(다음 토큰을 예측하는 상태). 평균과 마지막 내용 토큰을
float16로 저장한다. 합산은 float32, 저장만 float16. token ids, span offsets, 입력·코드·모델·adapter 해시를 남긴다.
원래 greedy, 같은 model·adapter, 동일 template/prefix인 경우에만 replay token equality를 요구한다.
불일치하면 probe 해석 전에 보고한다. LoRA와 base residual을 모두 보고한다. app adapter 잔차가 주분석이다.
원 생성 adapter가 다르면 app adapter로 재해석한 off-policy teacher forcing임을 표시한다.

## 선택 절차 (dev에서만)

각 범주에 StandardScaler + L2 logistic regression. 층 9개 × feature(mean,last) 2개.
C ∈ {0.001,0.01,0.1,1,10,100}; class_weight=None; max_iter=5000; seed=20261027.
dev 그룹 수와 두 클래스의 그룹 수가 허용하는 최대 5-fold StratifiedGroupKFold(shuffle=True, seed 고정),
최소 2-fold. 성립하지 않으면 해당 범주 학습 불가. scaler는 각 훈련 fold에서만 fit한다.
모든 dev OOF 예측을 합친 AUROC로 C/층/feature를 고른다. 동률은 작은 C, 앞층, mean 순.
같은 CV split을 baseline에도 쓴다. 단어 TF-IDF ngram=(1,2), 문자 TF-IDF ngram=(2,5), min_df=1,
max_features=100000, sublinear_tf=True, L2 logistic regression의 C는 같은 격자와 dev OOF AUROC로 고른다.
N0에서 저장한 dev-only 기준선은 C=1인 준비 모델이며, N1의 OOF 선택 절차가 최종 비교를 정한다.
입력은 답글 텍스트뿐이다. 명시적 label/provenance/메모를 feature에 넣지 않는다. 기존 결정 규칙도 비교한다.

각 모델의 threshold는 dev OOF에서 precision>=0.80인 후보 중 recall 최대, 동률이면 precision 최대,
그 다음 높은 threshold. 최소 5개의 predicted positive 필요. 없으면 threshold=+inf, 검출 실패로 처리한다.
best text baseline은 dev OOF의 위 threshold recall 최대(동률 AUROC, word, char, rule 이름순).
선택한 probe 하나, 선택한 word/char 모델, 미리 등록한 rule들을 전체 dev로 fit한 뒤 heldout에 딱 한 번 적용한다.
층/특징/C/threshold/비교 baseline을 heldout으로 바꾸지 않는다. base 모델용 선택도 dev만 사용하고 보조 결과로 표시한다.

## held-out 통과 기준 (범주별 모두 만족)

* AUROC >=0.85.
* dev에서 precision>=0.80로 정한 threshold에서 heldout precision>=0.75.
* 같은 운영 정밀도 조건에서 probe heldout recall이 dev로 정한 최선 text baseline보다 >=0.10 높음.
  baseline에도 자기 dev precision>=0.80 threshold를 고정 적용하며 heldout precision을 함께 적는다.
  두 모델이 heldout precision>=0.75를 지킬 때만 이 recall 비교로 통과를 주장한다.
* probe AUROC - max(word AUROC,char AUROC) >=0.05. 이 차이가 없으면 잔차의 추가 가치 없음.
* heldout을 보고 threshold를 맞춘 precision-recall 비교는 탐색적 도표일 뿐 통과 판단에는 쓰지 않는다.

dev label-shuffle control: seed 20261028..20261047로 20회, 그룹 크기와 클래스 수를 유지하는 범위에서
dev row 라벨을 permute하되 CV 그룹은 유지한다(그룹 내부 라벨 구조를 보존하는 group-block permutation도
그룹 크기가 같은 묶음에서 보조 실행). 각 회 층·feature·C 선택 전체를 다시 수행해 dev 선택 편향을 포함한다.
실제 OOF AUROC와 shuffle 최대/분포를 보고한다. heldout에는 shuffle을 적용하지 않는다.
95% percentile bootstrap CI, seed=20261027, 2000회 conversation/scenario 연결성분 재표집.
한 클래스만 남은 표본은 빼고 수를 보고한다. AUROC, precision, recall, 두 recall/AUROC 차이에 대해 짝 bootstrap.
overall와 캐릭터별 양성/음성 수·성능·CI, provenance별 결과, prompt completeness/adapter mismatch를 보고한다.
점수 못 내는 표본, 누락, underpowered 범주는 성공으로 처리하지 않는다. 의미 정확성을 문자열 손상 지표로 대체하지 않는다.

## R1 추가 분석 — 보고만, 성공 기준이나 W2 허가가 아님

correct R1 replies의 명시적 소유/주어 span만 사용해 별도 owner logistic probe(user vs character)를 dev에서 훈련한다.
주 R1 bad-reply sensor와 다른 타깃이다. 두 AI 일치 또는 명시적 정답표로 true owner와 span을 확보해야 한다.
정규식으로 후보 span은 찾을 수 있지만 그 자체로 true owner 라벨을 만들지 않는다. 모호·공유 소유는 제외한다.
span rule: 경계에서 나/내/저/제/너/네/니/우리/저희/사용자/캐릭터 + 선택적 의/는/가/의 것, 또는
1인칭·2인칭 주어 조사. UTF-8 byte overlap으로 토큰에 대응한다. span 내부와 바로 직전 위치를 별도 저장한다.
주 분석은 span 마지막 토큰 l_out, 보조는 span 직전(단어를 이미 본 영향 분리), 9층을 dev correct OOF로 선택.
올바른 답의 heldout owner accuracy를 먼저 제시한 뒤, 동결한 owner probe를 heldout slip의 같은 위치에 적용한다.
true-context owner와 surface-asserted owner 양쪽 확률/정확도·사례별 결과를 보고하고 conversation bootstrap CI를 붙인다.
비교: majority owner, span 문자 TF-IDF owner 분류기, base residual. 올바른 dev에 두 owner 클래스가 없거나
정답표가 없으면 NOT ESTABLISHED. 자동 span locator와 센서 점수에서 소유 정답을 역으로 만들지 않는다.
틀린 말 속에서 true owner를 읽어도 인과적 지식 또는 고칠 수 있음의 증거로 단정하지 않는다.

N0 종료 후 자동으로 N1/N2를 시작하지 않는다.
