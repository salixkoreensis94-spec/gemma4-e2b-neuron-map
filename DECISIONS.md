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
