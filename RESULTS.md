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
