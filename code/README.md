# Reproduction tools and verification boundary

Run `python3 code/check_offline.py` from the release directory. It uses the standard library, compiles every distributed Python source in memory (no imports of GPU engines), checks all 128 authored S6 validation replies against the frozen per-reply predictions and confusion matrix, and reruns 22 classifier regression probes. Expected result: 126/128, with two unknown friend replies. `other_casual` has no gold support. No model is loaded.

Run `python3 code/build_page.py` to rebuild `docs/index.html` and a content-only `artifact_body.html` from `data/`, `code/page.js` and `code/page.css`. All Korean/English interface prose is in the one `STRINGS` object. Generated quotations and measurement identifiers remain source data. The artifact has no HTML/head/body wrapper, and both forms share the same body. Existing outputs move to `S4/_superseded/page_builds/` in this workspace, or a sibling `_superseded/page_builds/` folder when used standalone.

The numerical modules preserve the stored study's arithmetic and rules. Model-dependent portability and end-to-end reconstruction have **not** been executed during this release preparation. A syntax check is not evidence of numerical equivalence in another environment. Use the original model hash and the CPU validation gates before running any screen/generation.

| Area | Code | Behavior / inputs |
|---|---|---|
| S1b lens | `lens/lens.cpp` | Full-vocabulary pre-softcap scoring, explicit GGUF and work directory flags; generic CBLAS |
| S2 | `s2/engine.py`, `kernels.py`, `metrics.py`, `run_screen.py` | GGUF text forward, cached last-position screen, all 512 controls, original damage statistics |
| S3 | `s3/` | Original generation engine, exact numerical compatibility, arm/dose generation and measures |
| S3b | `push/` | Exact generation, residual-pair direction construction, projection removal/mixes, measures and fixed arms |
| S6 | `s6/b_lens.py`, `b_compute.py`, `b_common.py` | Covariance lens, final-token pair directions, original S2 cached screen, complete dev/held-out grid, frozen selection |
| Classification | `s6/style_classifier.py`, `push/measures*.py`, `push/diagnose_checks.py` | Original frozen suffix logic and descriptive proxies |
| CPU reference | `reference/reference.cpp`, `export_rope_op.cpp`, `export_exp_corrections.cpp`, `export_gelu.py` | CPU references, RoPE operation table, configured libm and GELU compatibility tables |
| Tokenization | `tokenize.cpp`, `tokenize_inputs.cpp` | Vocab-only local GGUF tokenization; `INPUT OUTPUT MODEL.gguf` |

The S1b lens command keeps `lens-pilot`, `lens` and `sanity`; `inspect`, `calibrate` and `controls` expose existing GGUF inspection/calibration and the original control-generation procedure for reconstruction. No command in this release preparation has executed model inference. The covariance lens retains CUDA/Triton as an implementation dependency; removing machine-specific paths does not make these GPU algorithms CPU algorithms.

Supply dependencies yourself: Python with NumPy, PyTorch, Triton and the compatible GGUF reader for model-dependent tools; a C++17 compiler, zlib, a CBLAS implementation, nlohmann JSON headers and a compatible built llama.cpp source/library tree. The original CPU source fingerprints are in `data/reference_source_fingerprints.json`. No dependencies, model, weights or third-party source tree are downloaded or bundled here. Dependency versions were not exhaustively pinned in the stored record; validate behavior rather than assuming a package version guarantees equivalence.

Configuration is explicit:

```sh
export NEURON_WORKDIR="$PWD/work"
export NEURON_GGUF="/path/to/model.gguf"
export NEURON_NVIDIA_SMI="/path/to/nvidia-smi"
python3 code/prepare_workspace.py --workdir "$NEURON_WORKDIR"
```

The preparation command copies the adapted tools to the original stage layout under the chosen work directory and writes the included text-only fixtures. It does not load the model, generate token IDs, freeze a new run or claim a validation gate passed. It refuses an existing destination. Model path and GPU status executable come from configuration, and none is hard-coded to a personal machine. The GPU guard still pauses on external allocation >3072 MiB. Run numerical tools from the work directory; their stage-relative data layout is retained deliberately.

Build the lens with CMake, supplying paths to your compatible llama.cpp source/library tree and CBLAS headers if needed:

```sh
cmake -S code/lens -B work/build-lens \
  -DLLAMA_SOURCE=/path/to/llama.cpp \
  -DLLAMA_LIBRARY_DIR=/path/to/llama-libraries \
  -DCBLAS_INCLUDE_DIR=/path/to/cblas/headers
cmake --build work/build-lens
work/build-lens/lens inspect --model "$NEURON_GGUF" --workdir "$NEURON_WORKDIR"
work/build-lens/lens controls --model "$NEURON_GGUF" --workdir "$NEURON_WORKDIR"
```

Reconstruct and validate in dependency order:

1. Verify the GGUF SHA256 against `METHODS.md`. `inspect` writes vocabulary metadata for local token-group reconstruction. Remap the included whole-token surface allowlists to exact ordinary IDs with this supplied GGUF; do not substitute missing strings or use substring matching. `data/s1_group_surfaces.json` contains all ten original S1b group surface allowlists; `data/lens_map.json` contains map groups and `data/s6_token_group_plan.json` contains S6 target and partner lists. S2 ending candidates are in `data/s2_frame_candidates.json`.
2. Use vocab-only tokenization to create local tokenized frames/scenarios/pairs/tasks. `tokenize_inputs INPUT OUTPUT MODEL` tokenizes text tasks; `tokenize INPUT OUTPUT MODEL` checks pair prefix boundaries. Retain original chat templates/BOS/EOG conventions. Add S2 `banmal_ids`/`polite_ids` by exact decoded ending eligibility, not concatenated-stem canonical tokenization. Public fixtures omit all stored vocabulary IDs; they must be regenerated locally.
3. Calibration uses the included 40 S1 prompts. The stored scalar residual medians are included in `data/residual_norms.json` for comparison; selected residual-derived steering vectors are distributed in `vectors/`; see `vectors/README.md` for construction and application. The full calibration activations and original 512-per-layer control pool are not distributed. Original random controls are reconstructed with `mt19937_64(2026102600+layer)`, normal floats and the original random magnitude selection. Standard-library normal distributions can differ across platforms. Original control file SHA256 is `25d5035abf364bcee009c88664ff83ebf45a3c1695201173bba3155e440e414f`; an unequal regeneration is a new control set, not exact reproduction.
4. Build the CPU reference and numerical exporters against the same compatible source libraries. `reference.cpp` expects the configured GGUF and stage-relative text fixture files, tokenizes scenarios and produces greedy CPU references. `export_gelu.py --library LIBRARY --output FILE` reads the reference CPU lookup table. RoPE/exported exp correction tables depend on the supplied reference CPU arithmetic/libm. Never claim the original CPU equality merely because exporters run on another platform. Set library loader paths as required by your platform.
5. Put generated inputs in the stage layout: root `inputs/` contains model (or configured model path), controls, residual medians, GELU table and prompt sequences; stage `inputs/` contains decoder pieces, RoPE and exp tables. Run S2's engine validation, benchmark, full screen and all-position validation. Keep its original 40/40 top1 and ≥9/10 top10-overlap gates. The included source contains its downstream postprocessing and independent metrics.
6. Run S3 CPU/GPU generation validation (≥9/10 exact reference sequences, plus batch and zero equality), then `s3/run_arms.py`. S3b follows `push/validate.py`, `build_directions.py`, `run.py`. The reference register-vector file is not included; its arms must fail clearly if it is absent. To reproduce the historical grid exactly, an authorized copy is needed. Omitting those arms is a reduced/new experiment, not complete historical reproduction.
7. S6 uses the authored build/validation split, fixed prompt lines and held-out scenarios. Regenerate `tokenized_inputs.json`, token groups, `freeze.json`, engine validation and rule hashes for a **new** run using the included redacted rule. The public rule hash differs from the original by disclosed redaction. Update `RULE_HASH` in the new work copy to the new rule hash and freeze all input hashes before computation. Do not bypass freeze checks or label a redacted hash as the original. Then run the copied `S6/tools/b_compute.py` with its original S2 cached-screen path. The full grid precedes selection; no held-out reselection.

The frozen RULE copies and original/adapted hashes explain exact historical decisions. The prose above identifies reconstruction work that requires the supplied model and numerical dependencies; it is not a claim of tested one-command reproduction. The release decisions are recorded in `DECISIONS.md`. Model-dependent portability remains untested.


## Round-2 aggregate addition

`build_page.py` also reads `data/round2_sensors.json` for a short Korean/English sensor panel. Its build archive stays outside the staged public tree. The existing inference tools and `check_offline.py` cover round 1; no round-2 inference, probe training, private adapters, packages or app source are added. The new JSON files contain aggregates with source fingerprints; they do not enable end-to-end round-2 reconstruction. See `data/ROUND2.md` and `METHODS.md` for units, missing-threshold semantics and the frozen protocol copies.

## Round 3 static map

The one authoritative large data copy is `docs/map/data/round3/`, with map outputs under `stats/` and licensed excerpts under `snippets/open/`. `data/round3/` contains compact aggregate and metadata copies only. Run `python3 code/round3/build_page.py` to rebuild assets and synchronize compact metadata; it never duplicates the large shards. Serve `docs/` with a static server or GitHub Pages. The browser uses same-origin fetch and native gzip DecompressionStream, with no external font, analytics, host or judge calls.

Checks: `python3 code/round3/check_offline.py` verifies all 337,920 rows, snippet/reference geometry, single-copy layout and sizes; `node code/round3/check_cards.js` resolves the known cards plus an open-source snippet; `node code/round3/check_page.js` exercises the actual app with a read-only local DOM harness, including the enabled on-disk AI-label config and a disabled branch in memory. No snippet text is printed. The existing `python3 code/check_offline.py` retains round-1 tests. The parent separately reviews desktop/375px rendering in a browser.

`code/round3/map-config.json` has `show_ai_labels:true`. To remove label data too, run `python3 code/round3/build_page.py --exclude-ai-labels`; it preserves both small label copies outside the public tree. Labels describe 1,400 real neurons and 420 typed controls, without rationale text. An enabled label always carries the input-only AI-judge caveat.

Each card is a 580-byte little-endian row in a 256-neuron gzip shard. `docs/map/data/round3/stats/schema.json` specifies fields. Exact-piece search covers displayed firing-sample top 10 and lens top 10 using NFKC/trim/lowercase. Means retain float64; integer counts and float32 maxima are preserved; histogram midpoint/lens display values are float32. Quantiles are estimates and token shares are reservoir samples.

All 20 contexts are eligible for text after open-source, whole-chunk, boundary and snippet privacy checks. Windows are 15 + 1 + 16 tokens, clipped to 32; each clipped snippet is stored once, with per-position marks in reference shards. A fixed lexical list conceals matches until a click. The page loads five contexts first and the remaining 15 on expansion. Full JSON pointers have a closed details/copy view; visible token marks are ␣, ↵ and ⇥. No rehydration tool is included; see METHODS.md for the exact reconstruction boundary.

The source manifest had one inherited stale release digest for `code/s2/postprocess.py`. Round-3 staging corrects that metadata to the unchanged read-only snapshot bytes, preserves the previous digest as `inherited_release_sha256`, and claims no new inference validation.

### Whole-map browsing

`code/round3/build_browse.py` derives the one overview file and 35 layer files from public catalogs/cards, without corpus text, a model or extra Python dependencies. Rebuild it before `code/round3/build_page.py`. `code/round3/check_browse.py` verifies all 337,920 overview rows and all top-3 IDs, and independently sorts the L0 catalog. `check_offline.py` includes this check and the overview/grid/list loading bounds. `check_page.js` exercises real JS handlers with a DOM/canvas geometry harness; it does not claim a real browser viewport measurement.

Overview rows are 6 bytes: script, frequency-bin, maximum-bin, AI code, frequency brightness and log-maximum brightness. Layer browse rows are 25 bytes: six uint32 token IDs (top-20-context count top 3, lens top 3), then one AI code. Catalogs retain exact counts and float32 maxima. `browse/metadata.json` defines bins, dimensions and code meanings; these estimates are display encodings, not new measurements. History routes use hash navigation; card rendering does not replace history entries.

The compact vocabulary candidate was larger on every measured 50-row page and is excluded. Lists load the existing vocabulary shards, reusing cached strings. Canvas colors use dedicated sequential and script tokens; resizing or theme changes redraw the views. Phone list rows stack into four lines below 600 px; desktop keeps the table. A 64-column canvas, minimum 288 px wide and maximum 768 px on desktop, keeps cells at least 4 px wide.

The default browse mode is maximum activation. Seven ordered sequential CSS tokens match legend and canvas exactly. Script mode omits brightness mixing to keep muted Hangul distinct from vivid Latin and other scripts. AI keeps frequency brightness; encoded brightness bytes remain unchanged. Phone lists use four-line blocks below 600 px. Color preference uses guarded localStorage; unavailable storage falls back to maximum. Native touch clicks navigate without hover.

List sorting reads its controls at startup and on every table render. A deferred pageshow refresh also reads browser-restored values. The initial list default is number/ascending; the separate color-mode default stays maximum activation. The JS regression harness parses actual HTML option defaults and checks fresh and restored lists.

### Visitor-facing map fields

Rebuild in order: `python3 code/round3/build_browse.py`, `python3 code/round3/build_reader.py`, then `python3 code/round3/build_page.py`. No corpus text or model is read. `reader.js` contains paired Korean/English strings and deterministic token-count/rank helpers. `build_reader.py` verifies the global percentile edges, adds all distinct top-20-context token pieces to the existing index, and selects the eight starting points from public records and snippet-reference availability only. The selection rule is saved in `starting.json`; seven examples receive no semantic titles, and L27 N1939 alone uses its published observation. The list records remain 25 bytes, but their first three IDs now mean strongest-context counts, with best-context-rank ties. The 580-byte card rows and their reservoir fields are unchanged. Gzip compression choices affect transport bytes only.

Research numbers, raw lens values, quantile caveats, positive-reservoir counts and raw record IDs remain in card disclosures. Source pointers and full licenses are in source disclosures; text before the marked firing token is muted, the token is bold/underlined, and following text was not used for that position. Snippet files, references, filters and concealment flags are unchanged. The first-visit guide fold uses guarded localStorage. Unknown and unreadable AI results share the plain undecided label/color while their stored codes remain distinct. Browser form-restore sorting tests and touch checks remain in `check_page.js`; tests never print evidence text.

`build_page.py` also mirrors the unchanged aggregate Round 2/3 RESULTS.md sections into `docs/map/results.html`, keeping guide/result links within the Pages root. It excludes the Round 1 example section. No sentence evidence is added.

### Token visibility and starting points (M5h)

All token labels use `Round3Reader.visible`/`piece`: CR is ␍, existing space/LF/tab marks are kept, other C0 controls and DEL use Unicode control pictures, and C1 controls, Unicode format/default-ignorable characters, unusual spaces, surrogate fragments and blank braille use explicit U+ codes. An empty piece uses ⟦∅⟧. These marks do not change the vocabulary, snippet strings or search normalization. The node test `node code/round3/check_tokens.js` checks 114 fixtures, every starting point and 1,000 seeded random neurons' summary/list/lens token labels without printing token or context strings.

`python3 code/round3/build_reader.py --starting-only` updates starting-point metadata without rewriting the token index. Seven examples retain the majority-a/open-3-of-5 requirements, and additionally require a Hangul/Latin/Han letter in the rank-1 context token and at least 10 of 20 context positions. Masked/undecodable pieces do not count as letters. Each five-layer band takes its highest-maximum unused candidate; an exhausted band advances to the next band cyclically. The published L27 N1939 is kept separately. Selected/replacement bands and letter counts are recorded in starting.json. Script summaries still count all positive positions, independently of strongest-context tokens. Compact legend ranges keep full bounds/inclusivity in their title attributes.
