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
