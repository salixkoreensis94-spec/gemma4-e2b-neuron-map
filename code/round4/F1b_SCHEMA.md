# F2b adapter contract / F2b 연동 계약

Schema version 1 is explicitly extended to both kinds: `f1b/neuron_L00.json` through `neuron_L34.json`, `feature_L05.json` and `feature_L27.json`. Unit keys are decimal IDs. Neuron source data stay byte-identical; the neuron renderer reads its own new F1b file. / 뉴런·특징 두 종류와 별도 파일을 명시합니다. 기존 뉴런 데이터는 바이트 그대로입니다.

Before adapting, verify passed DONE, the 38 output manifest entries, all code-freeze files, resolved-unit and literal hashes, calibration freeze and its posthoc flag, and the amendment. Public provenance retains the verified hashes. / 완료·manifest·코드·단위·literal·보정·수정안 해시를 먼저 검증합니다.

Flatten target and both greedy-control directions with original strength, start index, seed/greedy and direction IDs. Copy the filtered shared 12 baseline rows to each shard. Preserve the source agreement array exactly; baseline lift uses all 12 cases and random lift uses target greedy 4 versus control greedy 8. No browser recomputation. / 원 배열·분모·범위를 보존합니다.

Apply M5 PII/identity patterns and its fragment/boundary rules to every complete generated string, including baseline and controls. Outputs are not split. Withheld rows contain neither text nor token IDs. Offensive lexical matches require a click. Fully withheld units are omitted; missing file/unit shows pending. Other load, parse or filter-state errors show retry and are not cached as success. / 모든 글을 검사하고 차단 글·ID를 빼며 어휘 가림은 클릭 뒤 보입니다.

Every generation stores `same_as_baseline` against the same start/seed, null when either side is withheld. Units publish the M5-scrubbed `top_context_token_ids` (IDs only) and flag summary. Nonwithheld rows include aligned `token_pieces`, with empty EOG and null scrubbed pieces. Mark only if pieces concatenate exactly to stored text; otherwise render plain text. Every token in an exact reconstruction is marked iff its ID belongs to the published set. / 일치 flag·요약·필터 뒤 토큰 집합을 저장하고 조각과 원문이 정확히 같을 때만 표식으로 보입니다.

The lexical-recovery gate failed, best 4/200. Production has no explanation column or explanation content. All F1b3 values are post hoc, with amendment and calibration hashes; cards link to the results page. / 회수 게이트 실패로 설명 칼럼·내용은 없습니다. 모든 F1b3 수치는 사후 수정으로 표시합니다.
