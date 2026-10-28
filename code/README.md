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
