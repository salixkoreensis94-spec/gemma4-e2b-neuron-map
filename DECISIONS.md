# Release decisions

Date: 2026-10-27

1. Repository name: `gemma4-e2b-neuron-map`.
2. Author and copyright credit: use the author recorded in `CITATION.cff`, with English spelling as shown in the README headers and copyright notices. Retain @_lyraaaa_ credit for the neuron-explorer idea; none of their code was used.
3. Code: Apache-2.0 (`LICENSE`). Data, documentation, vectors and generated replies: CC BY 4.0 (`LICENSE-DATA`). Include the Gemma provenance and trademark notice in `NOTICE`.
4. Include all S6 vectors and the selected S3b neuron, contrastive, mixed, reversed, auxiliary second-person and paired random vectors in `vectors/`, with construction, scaling and provenance metadata.
5. Exclude the app's register vector, its controls, `register_mean_raw.npy` and `reply_activations.npz`. Retain stored aggregate results under the label “reference register vector”.
6. Human review remains aggregate counts only. Include no reviewer quotations, individual marks or free-text notes.

This record applies local release preparation only. Nothing has been published.

## Round 3 staged decisions

Display choices confirmed on 2026-11-02: AI labels are shown on labelled cards with the input-only AI-judge caveat. Removal requests use a GitHub Issue naming a neuron, context number or pointer; never paste the text itself. Post-hoc floor note A is included after the M4 tables under its own heading; it does not change the failed frozen gates or create a verdict. The separate ordering note is excluded.

Open-license excerpts only; FineWeb has statistics and exact pointers. Large map/snippet data has one Pages copy. All 20 context ranks remain eligible, with 32-token windows and conservative whole-chunk/boundary/privacy checks. No commit, push or publication was made during staging. Final push approval and release-date setting remain separate steps.

## Round 4 plan · 2026-11-03

Owner’s requested order / 소유자가 요청한 순서: “특징 지도 먼저 진행하고 LoRA가 바꾸는 지도, 높임말 경로 추적 순서로 진행해줄 수 있어?”

F (feature map) → L (LoRA changes) → H (honorific-path tracing). F2 is a local stage for parent review; publication requires the owner’s separate OK. F1 is descriptive; no new test. F1b3 is now included as post hoc evidence in F2b. / F2는 부모 검토용 로컬 스테이지이며 공개 반영에는 소유자의 별도 OK가 필요합니다. F1은 기술 통계이고 새 검정은 없으며 F1b3는 F2b에 사후 수정 자료로 넣었습니다.

## 2026-11-03 — strength-rule amendment / 세기 규칙 수정

The owner’s choice, in his words: **“잣대만 고쳐 재판정 (Recommended)”**. The parent froze amendment 1 at 15:29:29 KST before computing amended values, after the original 15:18 FAIL. All amended F1b3 results remain post hoc. / 소유자가 고른 말 그대로입니다. 원 FAIL 뒤 수정 수치를 보기 전에 수정안을 동결했으며 이후 F1b3 수치는 전부 사후 수정입니다.

Amendment SHA256: `51ff412c1fe16dbd4a85d6ce3ab4e8f395e27d4aa15d8d300f00ebb718c11276`. F2b is a local review stage only.
