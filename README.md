# Gemma 4 E2B neuron map

**Author: Ryou Jiho**

Copyright 2026 Ryou Jiho.

Can adding a single MLP neuron's write direction, or a small contrastive direction, change Korean register while retaining the answer's content? This study follows @_lyraaaa_'s neuron-explorer idea (none of their code was used) and tests it on one base Gemma 4 E2B Q4_K_M GGUF, without LoRA in round 1.

The lens reads all 337,920 MLP write columns across 35 zero-indexed layers. It projects each column through final RMSNorm and the tied output embedding, ranking **pre-softcap** logits. A contextual screen then measures changes in ending-group next-token log odds. Generation adds `dose × layer median residual norm × unit direction` at the layer output, after FFN/PLE residuals and layer output scaling, at every prompt/generated position. This adds a write vector; it does not directly edit a neuron's activation.

Measured findings:

| Stage | Result under its frozen rule |
|---|---|
| S1b | Static casual/polite face validity failed: 0/20 and 4/20 |
| S2 | 582 neurons passed the original screen; 366 passed a two-sign control-envelope check; 39 also met the stricter expected-ending requirement |
| S3 | None of those 39 passed generation; separately named L27 N1939 passed at +0.5 and +0.8, with 9/10 flips at each |
| S3b | No condition passed the stricter friendly-register bar |
| S6 | Only hasipsio direction L27 +0.65 passed dev; target-dominant 18/20, then 11/12 held-out |
| Round 2 | Negative result: no residual sensor passed; R1/R3/R4 reply AUROC 0.744/0.906/0.802 versus best text 0.762/0.913/0.778; all pre-generation sensors underpowered |
| Round 3 | Full 337,920-neuron input map; pre-registered M2 H rejected; M3 L5/L27 judging exploratory; M4 gate failure, no verdict |

Round 2 tests residual readouts in the same base model with four private per-character LoRA adapters, using separate synthetic data and two independent AI judges. R2 was descriptive only. See [RESULTS.md](RESULTS.md#round-2-residual-sensors) for the negative result and the optional input-side follow-up; round-2 data are aggregate only, with no character names or LoRA replies.

**Automatic pass did not establish meaning preservation.** Direct nonblind readings found invented words, changed referents and factual errors even when script, repetition and length bars passed. Fixed prompt-only style lines produced 20/20 hasipsio and 17/20 haera dev replies, compared with 18/20 for the selected hasipsio knob and 13/20 for the best haera near-miss. These tests support register effects in specific settings, not a reliable character-control system or superiority over prompting/LoRA.

Browse the map: [docs/index.html](docs/index.html) — Korean first, with an English toggle.

Online map (GitHub Pages): https://salixkoreensis94-spec.github.io/gemma4-e2b-neuron-map/

The earlier-round page is self-contained. The full round-3 map loads only same-origin static shards; serve docs locally or use GitHub Pages, with no external runtime hosts. `RESULTS.md` gives tables and selected original replies; `METHODS.md` distinguishes measurements, proxies, selection and reading. `data/` contains compact JSON with all condition summaries and fixed illustrative reply samples, without checkpoints; round-3 map shards additionally retain vocabulary IDs for lookup. S5a appears only as aggregate counts; no reviewer words or individual marks are included.

Reproduce without a model: `python3 code/check_offline.py` verifies Python syntax, reruns the frozen classifier on all S6 validation pairs, and checks regression probes. `python3 code/build_page.py` rebuilds both page formats from data and the single strings object. See `code/README.md` for model-dependent reconstruction, configurable inputs and numerical dependencies. The user supplies the GGUF and compatible dependencies; weights are excluded. Selected steering vectors are included in `vectors/`; see its [README](vectors/README.md) and manifest for exact scaling and provenance. The app's register vector is excluded; results quoted for the reference register vector come from stored outputs. Inference-dependent portability has not been tested in this release step.

Limits: round 1 uses one quantized base model, synthetic short Korean dialogs, greedy decoding, 96-token cap, heuristic ending classification and no measured semantic-accuracy rate. Shared templates are not independent examples. S2 searches many neurons, and empirical random-control quantiles are not significance tests. CPU agreement is established only for the stored validation scope, not arbitrary long/pushed generation. Different methods use different injection norms. Round 1 did not compare LoRA steering. Round 2 uses four private adapters, synthetic short dialogs and greedy plus sampled replies, with agreed AI labels rather than human-validated truth. Shared families, small positive-family counts and the R1 distribution shift limit transfer; it establishes no reliable sensor, self-knowledge, corrective intervention or live-app benefit. Round-2 model inference is not reproducible from this aggregate-only addition alone.

## License

Code in `code/` and its JavaScript/CSS serving copies in `docs/map/` is Apache-2.0 ([LICENSE](LICENSE)). Our data in `data/`, map outputs in `docs/map/data/round3/stats/`, documentation and original project vectors/replies are CC BY 4.0 ([LICENSE-DATA](LICENSE-DATA)). Third-party excerpts in `docs/map/data/round3/snippets/open/` retain the source-specific CC BY 4.0, MIT or Apache-2.0/public-domain terms in their NOTICE/license files. Documentation licensing does not override these code/data exceptions. See [NOTICE](NOTICE) for Gemma provenance and trademarks; other dependencies retain their own licenses.

[DECISIONS.md](DECISIONS.md) records the 2026-10-27 release decisions. See [SCRUB_REPORT.md](SCRUB_REPORT.md) for the privacy scan.

## Round 3: full input/output map

[Open the full map](docs/map/index.html) (Korean first, English toggle). It covers all 35 layers and 337,920 neurons on 49,438,284 dev tokens. Histogram quantiles and reservoir samples are labeled. Static lens associations are separate from observed firing contexts. M2 includes 1,400 neurons and 420 controls, not 1,820 real neurons. Optional AI labels are in a separate file and shown on labelled cards with the one-line AI-judge caveat.

Round-3 data has one large copy under `docs/map/data/round3/`: our `stats/` outputs are CC BY 4.0 and `snippets/open/` excerpts retain each source’s NOTICE/license. `data/round3/` holds compact aggregate/metadata copies only. FineWeb statistics and exact pointers retain ODC-By database attribution and Common Crawl terms; source web text is neither redistributed nor relicensed.

Removal requests go through a repository [GitHub Issue](https://github.com/salixkoreensis94-spec/gemma4-e2b-neuron-map/issues) naming the neuron, context number or pointer. **Never paste the text itself into the issue.**

<!-- BROWSE_START -->
### Browse without typing

The full map opens on **Whole map**, with all 35 layers on one canvas. Click a strip to open that layer at the indicated neuron range. **Layer grid** uses 64 neurons per row, in neuron order; filters dim nonmatching cells. **Layer list** uses 50 rows per page, with number/frequency/maximum sorting and strongest-context/lens top 3 token pieces. Clicking a cell or row opens its card, and Back returns to the previous view.

Color modes are main script, positive frequency, maximum activation and AI label. Legends show the actual bin bounds. Maximum activation is the default; the last chosen mode is saved locally when storage is available. Frequency and maximum use seven discrete viridis colors matching their legends. Script mode uses muted Hangul and vivid alternatives without brightness mixing; AI retains frequency brightness. Reduced pixels use the majority category (smaller-code ties). On phones, list rows become four-line blocks and the language control has its own row. A grid or overview tap opens the next view without hover. AI labels cover only the judged subset; unknown consensus remains separate from unreadable u. Use the grid for individual cells.

Routes: `#L5` (grid), `#L5-list` (list), `#L5-N1234` (card). Korean comes first, with an English toggle. The grid canvas has a 288 px minimum width: 64 columns minus a 0.5 px gap leave cells at least 4 px wide; narrow overflow stays within its own box. Its desktop width is capped at 768 px.

The card starts with counts of token IDs in the 20 strongest contexts (ties by best rank), then the evidence, then the output lens. No meanings are assigned by these counts. Frequency, signed mean/max, histogram-estimated quantiles, language/script shares and the 8,192-position positive reservoir sample remain in a closed Numbers in detail fold. The list top 3 now uses the same strongest-context counts, not reservoir frequencies. Search covers all distinct top-20 context tokens as well as the existing reservoir top 10 and lens top 10. A layer catalog is loaded lazily for within-layer maximum ranks; ties are included.

Starting points now also require a Hangul, Latin or Han letter in the strongest context token and in at least 10 of the 20 context tokens. Empty five-layer bands advance to the next band cyclically; ties use layer/neuron order. L27 N1939 remains the published exception. Token labels make carriage returns, other controls, invisible format characters and unusual spaces visible; the stored token strings, excerpts and search normalization are unchanged.

<!-- BROWSE_END -->

<!-- SNIPPET_COUNTS_START -->
### Open-source snippet counts

All 20 contexts remain eligible for text; no size cap was needed. Each window has at most 32 tokens (15 before + firing token + 16 after, clipped). Web rows show statistics/pointers and a single firing-token string only. Snippet privacy checks are conservative patterns, not a guarantee of de-identification.

| Open source | Candidate chunks | Unique snippets | Neuron-context references | Hidden references | Covered tokens | Dev coverage | Full fixed-slice coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| songys/Chatbot_data | 6,957 | 8,996 | 12,680 | 8 | 129,093 | 60.546% | 59.900% |
| mkd-chanwoo/keural-conversation-ko | 48,873 | 250,602 | 285,496 | 365 | 3,607,514 | 72.896% | 72.150% |
| allenai/soda | 29,755 | 486,209 | 594,320 | 516 | 4,506,324 | 91.073% | 90.126% |
| deepmind/pg19 | 5,871 | 150,756 | 199,336 | 2 | 1,380,098 | 92.007% | 92.007% |

References count neuron/context slots; identical clipped windows are stored once. Coverage is the union of token intervals per corpus chunk. Dev and full fixed-source-slice denominators are distinct; no heldout text is published.

| Drop category | Chunks | Context references |
|---|---:|---:|
| Host-name pattern | 4 | 124 |
| Project-private name list | 67 | 527 |
| account | 0 | 0 |
| address | 97 | 1,743 |
| boundary_address | 79 | 1,025 |
| boundary_phone | 64 | 931 |
| boundary_to_next_chunk | 1 | 125 |
| email | 0 | 0 |
| head_digit_fragment | 3 | 256 |
| missing_previous_chunk | 0 | 0 |
| phone | 66 | 1,030 |
| resident_registration | 0 | 0 |

The project-private name row is a distinct union, counted once per chunk/reference across its patterns. Other categories may overlap. There are 229 distinct rejected chunks and 3,618 rejected references; additional snippet-only rejections are zero. The 891 hidden references are retained behind a click, not deleted.
<!-- SNIPPET_COUNTS_END -->

## Round 4 F: feature map (F2 stage)

The feature map adds all 32,768 transcoder features at L5 and L27 beside the 337,920 neurons. Switch maps in the guide, overview or list; token search returns separate neuron and feature result groups. Cards show strongest-context token counts, licensed context evidence, static output lens, five same-layer weight neighbors, exploratory AI labels and folded descriptive statistics. Neighbor links work in both directions; the inherited neuron data is unchanged.

F1 recorded the same 49,438,284 development tokens in 263,910 chunks. Both layers have zero dead features in this capture. Rates below one firing per million tokens occur for 91 L5 and 20 L27 features; above 10% for 0 and 1. Median rates are 0.050859% and 0.085892%; mean active features per token are 31.999829 and 31.998973. These are descriptive F1 counts, with no new hypothesis test. F1 reservoirs were checked against M3d by the capture review (maximum difference 0.0).

A transcoder learns a sparse approximation to a layer output; each learned decoder direction with its activation is a feature. At k32 about 32 of 16,384 features are active per token. Feature and neuron numbers identify different units. Maximum and firing-rate color bins are separately derived within each feature layer using percentiles 0,10,25,50,75,90,99,100. Identical colors across layers or kinds do not imply equal magnitudes. Starting points are the highest-maximum feature in each layer, with smaller ID breaking ties: L5 F1602, L27 F11851. No semantic titles or outcome-driven selection are used.

All three layers failed the original Round 3 CE gate. Under the post hoc KL amendment L5 and L27 passed; L15 had 27.0325% dead features at k32 and 49.8840% at the k64 retry, above the 10% bar. The preregistered M3 path stopped. The map reuses the later exploratory branch, not a passed preregistered test.

Feature lenses contain top/bottom 50 vocabulary IDs and pre-softcap scores from the RMS-normalized decoder direction through the actual output norm and unembedding, following the S1 write-column convention. This static proxy measures neither probability change nor intervention effect. Neighbor links use the five largest same-layer cosines between transcoder decoder rows and neuron ffn_down columns in each direction. They measure weight similarity, not co-firing or causes; the reverse top-five is ranked independently and need not be reciprocal.

Exactly 240 features have the majority category from the three M3 exploratory judges, mapped through the frozen packet manifest. Unknown consensus remains unknown. No judge concepts, rationales or input packets are distributed. The caveat states that these are input-context-only exploratory AI judgments, not human-verified and not the preregistered test. The exploratory one-concept proportions (86.3% features versus 45.0% neurons, rounded) motivated browsing, not a new inferential claim.

All top20 distinct-chunk contexts retain activation, token and pointer data. Text windows use 15 tokens before, the firing token and 16 after, clipped to 32. Only the four open sources allowed in M5 publish snippets, with the same attribution and source-license labels. Web sources publish statistics and pointers only. Whole-chunk PII and identity patterns, document-boundary checks, the frozen boundary index and window filters precede publication; fixed lexical matches are concealed behind an explicit reveal button. These conservative rules cannot guarantee de-identification. Every histogram stores all 1,024 exact counts; the display groups them in 64 bars and keeps exact nonzero bins in a table. Positive language/script counts use positive firing counts as denominators. The firing-token sample comes from positive positions within the uniform 8,192-position reservoir and is labelled as a sample, not a full count.


### F1b3 — direct clamp generations (all amended results post hoc)

This is descriptive intervention evidence, without semantic ratings or a hypothesis test. F1b3 covered 438 neurons across layers and 440 features (878 units). A neuron is set to strength × its observed maximum after GELU×up and before down projection. A feature adds strength × maximum × its frozen decoder direction to the raw MLP output before post-FFN norm. BOS and padding are untouched; later valid positions are intervened on throughout prefill and decode. This does not establish a unit’s meaning.

Four raw starts (BOS, BOS plus the two fixed Korean starts, BOS plus the fixed English start) each have greedy and two independently seeded temperature-0.8, top-p-0.95 samples, capped at 32 new tokens. The shared unclamped baseline uses the same starts and seeds. Two frozen Gaussian directions provide four greedy starts each, matching the local raw-MLP perturbation L2 norm rather than downstream effects. EOG IDs 1/106 are excluded from overlap counts. The cards pair the baseline with both selected strengths, put samples and controls in folds, and show unchanged source agreement counts with denominators in a details fold.

The original frozen strength rule FAILED/BLOCKED at 15:18 KST on 2026-11-03: fluency passes 166/240, 169/240, 163/240 and 155/240 (0.692/0.704/0.679/0.646) at 0.25/0.5/1/2, below 0.80. The unclamped model itself failed the repetition criterion in 3/12 cases; the parent identified an unreachable 80% bar under that calibration yardstick. The owner chose “잣대만 고쳐 재판정 (Recommended)”. Amendment 1 was frozen at 15:29:29 KST, before any amended value was computed. The original FAIL remains on record. The amendment adds baseline UTF-8 and repetition eligibility to the existing empty/nonfinite exclusions; the minimum of 8 eligible cases, 80% bar, edit threshold 0.15 and smallest-two selection stay fixed. Production and baseline overlap still use all 12 cases.

Under the post-hoc amendment, 9/12 cases are eligible. Fluency passes are 151/180, 154/180, 145/180 and 142/180, with median normalized edits 0.502397, 0.593403, 0.673264 and 0.699653; the selected strengths are 0.25 and 0.5 (post hoc). The saved laptop re-run at 2026-11-03 19:07 KST used `tools/compare_old_tasks.py --old F1b/tasks --new F1b3/tasks`. Its output, `F/F1b3/parent_checks/compare_old_tasks_20261103.out` (SHA256 `00de6c58738eca18bc7f84faf4f15e685b36a4a91077974d4340ba90272423d2`), records 1,092 cases, 99/99 matched tasks and zero token-ID, metric and NLL mismatches (also zero missing/extra tasks). The bundled `task_comparison_dry_run.json` remains a separate CPU self-comparison. The public `f1b_provenance.json` stores the saved re-run’s hash, counts, command and time. F1b3 DONE passed at 18:43 KST (878 units, 38 shards, 7,169 new and 19 reused batches; post hoc).

Self-explanation was tried and failed its frozen lexical-recovery bar: best 4/200 (2/100 per language) at L5 ×2 or ×4, against 100/200 overall and 40/100 per language; L15 and L27 recovered zero. No production unit has explanations. This bar was not relaxed. All F1b3 generation, calibration and overlap results are post hoc. Amendment SHA256: `51ff412c1fe16dbd4a85d6ce3ab4e8f395e27d4aa15d8d300f00ebb718c11276`. Calibration freeze SHA256: `43dbb5477e5f86b5a7eb77c468b3539f56a63ce6c154a4b702b8c207a7459ac9`, with `posthoc: true`.

Forcing a unit on mostly leaves text close to the unclamped model’s text; only some units pull in more of their own top-context tokens. Static lens top-10 hits are rare. Token overlap is neither a semantic rating nor proof of meaning. Individual cards preserve the full source agreement array; no rates are recalculated in the browser. All target, baseline and control strings pass M5’s PII/identity and concealment filters before publication into this local stage. Withheld rows ship neither text nor recoverable token IDs. Lexical offensive matches require a click. Original private shards stay outside the stage.

Cards store token-ID equality against the same start and seed as `same_as_baseline` (null if either side is withheld). Each card counts 8 most-likely-token and 16 seeded target cells separately, with withheld comparisons apart. Starts whose target cells all match are folded. The published `top_context_token_ids` are the source agreement set after M5 card-token scrubbing; rejected IDs are omitted. Nonwithheld rows have aligned token pieces (end tokens have empty pieces; scrubbed pieces are null). Text is marked only when the pieces are strings and concatenate exactly to the stored text; otherwise it stays plain. Marks denote strongest-context tokens, never meaning; lens counts stay in the metrics fold. Raw output preserves markdown symbols. / 기준선 일치·가림은 따로 세며 전부 같은 시작은 접습니다. 상위 문맥 토큰 집합은 M5 카드 토큰 필터 뒤 ID만 공개하고, 토큰 조각의 연결이 원문과 정확히 같을 때만 표식으로 보입니다. 표식은 뜻이 아니며 마크다운 기호도 원 출력 그대로입니다.
