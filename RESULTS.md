# Results and selected original replies

All counts retain the original stage rules. Generated reply quotations are model outputs, not reviewer notes. Qualitative readings are nonblind and do not change pass/fail. Layer/neuron indices are zero based.

## S3 automatic passes

| Condition | Dose | Dominant / N | Flips | H / D4 / length | Random flips | None word recall |
|---|---:|---:|---:|---|---:|---:|
| L27N01939 | 0.5 | 10/10 | 9 | 10/9/10 | 0 | 0.323 |
| L27N01939 | 0.8 | 10/10 | 9 | 9/10/10 | 0 | 0.144 |

None of 39 strict S2 neurons passed. L27 N1939 was a separate named S1 candidate. Requiring all ten Hangul/distinct-4 passes yields zero passing conditions. App/reference vector doses .25/.5/.8 have respectively 1/7/5 flips and 2/8/6 target-dominant replies. Its raw multi-layer scale is not the same norm as a neuron dose.

## S3b comparisons

| Condition | Dose | Dominant /10 | Flips | Friend + flip | H / D4 / length | Pronoun delta | Random flips | Decision |
|---|---:|---:|---:|---:|---|---:|---:|---|
| minus_you | 0.5 | 9 | 8 | 4 | 10/10/10 | +0.50 | 0 | FAIL |
| minus_you | 0.8 | 10 | 9 | 0 | 10/10/10 | +0.20 | 0 | FAIL |
| friend_last | 0.65 | 8 | 7 | 7 | 10/10/10 | +0.20 | 0 | FAIL |
| mix_minus_you | 0.5 | 10 | 9 | 3 | 10/10/10 | +0.60 | 0 | FAIL |
| mix_friend | 0.5 | 6 | 6 | 6 | 10/10/10 | +0.50 | 0 | FAIL |
| reverse_friend_last | 0.15 | 10 | 6 | 0 | 10/10/10 | +0.00 | 0 | FAIL |
| N1939 | 0.5 | 10 | 9 | 4 | 10/9/10 | +0.90 | 0 | FAIL |
| N6893 | 0.5 | 10 | 6 | 0 | 10/10/10 | +0.00 | 3 | FAIL |
| app | 0.5 | 8 | 7 | 6 | 10/10/10 | +0.00 | 0 | FAIL |

No S3b condition passed. Polite-target flips cannot exceed six under the fixed None-source rule. Nine flips for N1939 +.5 still give only four friend+flip replies and +.9 pronouns/reply.

## S6 selected and closest nonpassing conditions

| Style | Status | Condition | Dominant /20 | H / D4 / length | Random dominant | Failed checks |
|---|---|---|---:|---|---:|---|
| haera | no knob | haera.dir.L27.d0.65 | 13 | 20/15/15 | 1 | target, distinct4, length |
| hasipsio | knob | hasipsio.dir.L27.d0.65 | 18 | 18/20/19 | 0 | none |
| aegyo | no knob | aegyo.dir.L08.d0.15 | 0 | 20/20/20 | 0 | target |
| eumseum | no knob | eumseum.dir.L08.d0.65 | 4 | 20/20/9 | 0 | target, length |

Prompt-only dev dominance: hasipsio 20/20, haera 17/20, eumseum 10/20, aegyo 2/20. Hasipsio L8 +.4 reaches 20/20 but fails length; this is not a pass. All 100 style × layer × dose direction rows are in `data/s6_style_layer.json`; all 285 dev and 8 held-out condition aggregates, including controls/None/prompt-only, are in `data/s6_all_conditions.json`.

| Held-out style | None | Prompt | Knob | Prompt + knob | Random |
|---|---:|---:|---:|---:|---:|
| haera | 0/12 | 10/12 | not run | not run | not run |
| hasipsio | 0/12 | 12/12 | 11/12 | 12/12 | 1/12 |
| aegyo | 0/12 | 1/12 | not run | not run | not run |
| eumseum | 0/12 | 10/12 | not run | not run | not run |

Held-out does not reselect conditions. No-knob means none passed this fixed search space, not proof that a style is impossible to control.

## Direct readings and why damage bars miss errors

S3b nonblind reading notes report nonce words and altered referents under N1939/mixed directions, broken text at minus_you +.8 despite ten surface passes, and awkward/invented words under friend_last +.65. Known suffix misses do not alter frozen counts. S6 parent reading finds malformed phrases and factual errors under the selected hasipsio direction; L8 can change prose genre while discarding content. These are observations, not a blind rating or measured semantic-error rate. Script, character repetition and length can all pass while the reply answers the wrong question. None content-word recall is only lexical overlap.

## Fixed illustrative replies

Selection rule: dev dinner/gift in each register; held-out museum/cinema/chess. These are post-hoc S4 illustrations, held consistent across conditions and doses. They are not a preregistered sample or an estimate of overall output quality. Full selected reply sample and authored context are in `data/replies.json` and `data/scenarios.json`. The quotes below are a smaller fixed condition/topic subset for reading.

### S3: L27N01939 +0.5 / polite_gift

None:

요즘 인기 있는 베스트셀러를 찾아보는 건 어때요? 아니면 예쁜 디자인의 문구류도 좋을 것 같아요.

Push:

네가 좋아하는 작가의 신구나 네가 찜했던 책을 선물하는 거 추천할게. 책갈피 같은 네가 좋아하는 네가 널리 알려진 굿즈도 괜찮다.

Hangul 1.000; D4 0.899; length ratio 1.400; truncated False. These flags do not assess meaning.

### S3: L27N01939 +0.8 / polite_gift

None:

요즘 인기 있는 베스트셀러를 찾아보는 건 어때요? 아니면 예쁜 디자인의 문구류도 좋을 것 같아요.

Push:

네가 좋아하는 책의 마브리브르브 같은 짧은 키우는 후브러브르브를 추천할게. 아니면 특별한 에디션의 책도 괜찮을 거야.

Hangul 1.000; D4 1.000; length ratio 1.400; truncated False. These flags do not assess meaning.

### S3b: friend_last +0.65 / polite_gift

None:

요즘 인기 있는 베스트셀러를 찾아보는 건 어때요? 아니면 예쁜 디자인의 문구류도 좋을 것 같아요.

Push:

네온사인으로 빛나는 저자들의 사인본이 꽤 괜찮을 것 같아. 아니면 네가 좋아하는 분위기의 핫트루피스트를 줄 수도 있어.

Hangul 1.000; D4 1.000; length ratio 1.267; truncated False. These flags do not assess meaning.

### S3b: friend_last +0.8 / polite_gift

None:

요즘 인기 있는 베스트셀러를 찾아보는 건 어때요? 아니면 예쁜 디자인의 문구류도 좋을 것 같아요.

Push:

걔가 좋아하는 재밌는 소설을 하나 줄래. 아니면 새로 나온 걔가 관심 있어 할 만한 책을 찾아볼 정도는 해.

Hangul 1.000; D4 1.000; length ratio 1.100; truncated False. These flags do not assess meaning.

### S3b: minus_you +0.65 / polite_gift

None:

요즘 인기 있는 베스트셀러를 찾아보는 건 어때요? 아니면 예쁜 디자인의 문구류도 좋을 것 같아요.

Push:

네가 좋아하는 작가의 에디션이구나. 아니면 네가 요즘 널 추천하는 책을 선물하는 것도 좋겠다.

Hangul 1.000; D4 1.000; length ratio 0.933; truncated False. These flags do not assess meaning.

### S3b: minus_you +0.8 / polite_gift

None:

요즘 인기 있는 베스트셀러를 찾아보는 건 어때요? 아니면 예쁜 디자인의 문구류도 좋을 것 같아요.

Push:

너의 취향을 고려해서 재미있는 에디스디스큐브를 추천할게. 아니면 네가 요즘 � 않 싶었던 신구나 짧은 시도권도 좋아할 거야.

Hangul 1.000; D4 1.000; length ratio 1.467; truncated False. These flags do not assess meaning.

### S6: hasipsio.dir.L27 +0.65 / polite_gift

None:

요즘 인기 있는 베스트셀러를 찾아보는 건 어때요? 아니면 예쁜 디자인의 문구류도 좋을 것 같아요.

Push:

새로운 장의 소서를 준비하여 선물하는 것은 어떠십시오. 친구의 관심사를 고려하여 평소 선호하는 분야의 책을 선택하십시오.

Hangul 1.000; D4 0.984; length ratio 1.100; truncated False. These flags do not assess meaning.

### S6: hasipsio.dir.L27 +0.65 / heldout_banmal_chess

None:

비숍은 대각선 방향으로 원하는 만큼 이동할 수 있어요.

Push:

비숍은 모든 방향으로 교차하여 이동할 수 있습니다.

Hangul 1.000; D4 1.000; length ratio 0.933; truncated False. These flags do not assess meaning.

## S5a aggregate review only

One reviewer, 30 cells; blind and post-reveal marks were identical. Category orders are fixed. No reviewer words or individual marks are included.

| Condition | Naturalness good/neutral/bad | Relevance fits/partial/no | Better/worse/same vs vector |
|---|---|---|---|
| L27 N1939 +0.25 | 6 / 1 / 3 | 9 / 0 / 1 | 0 / 5 / 5 |
| L27 N1939 +0.5 | 8 / 2 / 0 | 9 / 0 / 1 | 1 / 3 / 6 |
| reference register vector ×0.5 (L4–14) | 9 / 1 / 0 | 9 / 1 / 0 | reference |

## Round 2: residual sensors

**Negative result: no residual sensor passed the frozen bars.** The question was whether residual readouts can detect character failures better than text or token surprise, without changing the reply. R1 tests ownership confusion (whose person, event or fact); R2 assistant/AI identity drift; R3 formulaic AI tone; R4 unsupported memory, prior-conversation or date/time claims. The primary model was Gemma 4 E2B Q4_K_M with four per-character LoRA adapters (A–D); adapters are not distributed.

Frozen reply bars required AUROC ≥0.85, ≥0.05 above the best word/character/token-surprise AUROC, held-out precision ≥0.75 at a dev precision ≥0.80 threshold, and recall ≥0.10 above the dev-selected text/rule comparator while both retained precision ≥0.75. Pre-generation bars required AUROC ≥0.80, ≥0.05 above pre-generation word/character baselines, and held-out precision ≥0.50 plus recall ≥0.30 at a dev precision ≥0.60 threshold. Every sensor also needed ≥40 agreed positive held-out sensor units. Thresholds required at least five predicted positives on dev; no qualifying threshold meant +∞ and failure, with no held-out retuning.

| Reply sensor | Residual AUROC [95% family CI] | Best text AUROC (character) | Token-surprise AUROC | Residual − best baseline | Held-out precision / recall | Verdict |
|---|---|---:|---:|---:|---|---|
| R1 ownership | 0.744 [0.648, 0.823] | 0.762 | 0.566 | −0.018 | 0.355 / 0.193 | FAIL |
| R3 AI tone | 0.906 [0.851, 0.947] | 0.913 | 0.719 | −0.007 | 0.000 / 0.000 | FAIL |
| R4 memory | 0.802 [0.658, 0.902] | 0.778 | 0.577 | +0.024 | 0.000 / 0.000 | FAIL |

Token surprise selected mean NLL on dev from mean, maximum and highest-NLL 10% tail; it was computed from untempered logits using the same model/adapter as the residual. These AUROCs measure ranking against agreed AI labels, not calibrated awareness of an error. R3's high reply AUROC did not improve on the character baseline or yield a qualifying dev operating threshold. R4 missed both the absolute AUROC and added-value bars. R2 was descriptive only: its frozen held-out raw counts were 11 positive / 842 negative / 11 disagreements, with 10 positives in the base arm and a label-definition problem around ordinary professional disclaimers. No R2 sensor was trained or passed.

| Pre-generation sensor | Residual AUROC [95% family CI] | Best pre-text AUROC | Held-out positive / negative units (positive families) | Dev operating threshold | Verdict |
|---|---|---:|---|---|---|
| R1 | 0.645 [0.422, 0.865] | 0.612 (word) | 16 / 295 (7) | none qualifying | UNDERPOWERED |
| R3 | 0.854 [0.737, 0.928] | 0.653 (word) | 34 / 209 (17) | none qualifying | UNDERPOWERED |
| R4 | 0.766 [0.613, 0.908] | 0.806 (word) | 14 / 262 (6) | none qualifying | UNDERPOWERED |

All three pre-generation sensors also had zero detections at their frozen thresholds. R3 is a descriptive signal, not a usable sensor: it missed the power and operating bars. The same prompt readout cannot distinguish LoRA arm a from base-generated arm c when their prompts are identical under the primary LoRA capture model.

| Selection diagnostic (dev OOF only) | Actual AUROC | Maximum of 20 row shuffles | Maximum of 20 family-block shuffles |
|---|---:|---:|---:|
| R1 reply | 0.918 | 0.637 | 0.741 |
| R1 pre | 0.928 | 0.710 | 0.841 |
| R3 reply | 0.871 | 0.599 | 0.788 |
| R3 pre | 0.729 | 0.632 | 0.713 |
| R4 reply | 0.809 | 0.603 | 0.779 |
| R4 pre | 0.829 | 0.703 | 0.773 |

Each shuffle repeated the full layer/feature/C selection with fixed grouped folds. Family-block shuffles moved label blocks only among equal-sized families; limited movable groups constrain that control. Their higher maxima than row shuffles are consistent with selection exploiting family structure. The R1 dev-to-held-out drop also includes the N0b→N0c distribution change; these diagnostics do not isolate its causes or establish a significance test. In particular R3 pre dev OOF 0.729 is close to the family-block maximum 0.713.

The separate owner probe was reporting only, trained on correct dev spans rather than bad-reply labels. On 44 correct held-out spans its LoRA span-last true-owner accuracy was 0.977. At 19 natural slip spans it read the asserted wrong surface owner much more often than the context's true owner:

| Reporting-only owner readout | Correct held-out true-owner accuracy | Slip true-owner / surface-owner accuracy |
|---|---:|---|
| Majority owner | 0.795 | 0.947 / 0.053 |
| Span-character baseline | 0.977 | 0.053 / 0.947 |
| LoRA span-last (L2) | 0.977 | 0.105 / 0.895 |
| LoRA immediately before span (L30) | 0.886 | 0.105 / 0.895 |

| Separate minimal owner flips | Pairs | Flipped − original span NLL, nats [95% family CI] | Span-last Δp(true owner) [95% family CI] |
|---|---:|---|---|
| N0b dev analysis | 28 | +2.633 [2.156, 3.041] | −0.087 [−0.109, −0.074] |
| N0c held-out analysis | 23 | +4.471 [3.624, 5.754] | −0.424 [−0.461, −0.390] |

Before the edited span, the counterfactual readout difference was zero because the prefix was identical. Constructed flips were more surprising, but they were not natural errors and never entered sensor training or slip totals. The natural-slip readout offers no evidence here of a correct hidden owner representation that a steering intervention could recover. This is a limited probe reading, not proof about all internal knowledge, causal mechanisms or fixability. No steering/N2 was run. **No sensor passed.**

<!-- BEGIN OPTIONAL INPUT FOLLOW-UP -->
### Input-side follow-up

At ownership slips the owner probe usually read the wrong owner, so an input-side alternative was tried: add a note entry for a user-side person or pet whose name collides with the character or its own relations. An a versus a′ check used 46 scenarios with byte-identical app-off and recorded N0c-a notes, reusing 138 baseline replies and generating 138 a′ replies; both were judged again together by two independent AI judges. All three bars missed: better 10 / worse 12 / tied 24 (one-sided sign-test p=0.738; required p≤0.10 and net improvement≥5), agreed ownership slips 27→26 (required ≤18), and note echo 6/138=4.35% (required ≤3%). Disagreements remained unknown. Nonblind reading suggested that a contrast inside parentheses primed the character's own ownership claim; that explanation was not causally isolated. Four separately reported scenarios did not enter the bars. This is a controlled prompt replay, not a live-app effectiveness measurement. N0c-a used a hand-authored ledger with the app's collision exclusion, rather than the app's full reader; the matched-note restriction matters. No app source, prompt text or generated reply is included here.
<!-- END OPTIONAL INPUT FOLLOW-UP -->

Round-2 aggregate data: `data/round2_sensors.json`, `round2_counts.json`, `round2_owner.json` and `round2_namesakes.json`. Frozen rule copies and original/copy hashes are under `preregistration/`; their publication redactions do not create a new historical preregistration.

<!-- OPTIONAL BENCH KEY: insert approved A–D names/descriptions here, or leave empty. -->
<!-- OPTIONAL ROUND2 SAMPLES: insert only approved fixed illustrations here, or leave empty. -->
<!-- OPTIONAL NOTE PATTERN: insert an approved generic line here, or leave empty. -->
