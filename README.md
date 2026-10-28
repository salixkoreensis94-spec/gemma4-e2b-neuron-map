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

Round 2 tests residual readouts in the same base model with four private per-character LoRA adapters, using separate synthetic data and two independent AI judges. R2 was descriptive only. See [RESULTS.md](RESULTS.md#round-2-residual-sensors) for the negative result and the optional input-side follow-up; round-2 data are aggregate only, with no character names or LoRA replies.

**Automatic pass did not establish meaning preservation.** Direct nonblind readings found invented words, changed referents and factual errors even when script, repetition and length bars passed. Fixed prompt-only style lines produced 20/20 hasipsio and 17/20 haera dev replies, compared with 18/20 for the selected hasipsio knob and 13/20 for the best haera near-miss. These tests support register effects in specific settings, not a reliable character-control system or superiority over prompting/LoRA.

Browse the map: [docs/index.html](docs/index.html) — Korean first, with an English toggle.

Online map (GitHub Pages): https://salixkoreensis94-spec.github.io/gemma4-e2b-neuron-map/

The map can be opened locally. It is self contained and makes no runtime network requests. `RESULTS.md` gives tables and selected original replies; `METHODS.md` distinguishes measurements, proxies, selection and reading. `data/` contains compact JSON with all condition summaries and fixed illustrative reply samples, without vocabulary token IDs or checkpoints. S5a appears only as aggregate counts; no reviewer words or individual marks are included.

Reproduce without a model: `python3 code/check_offline.py` verifies Python syntax, reruns the frozen classifier on all S6 validation pairs, and checks regression probes. `python3 code/build_page.py` rebuilds both page formats from data and the single strings object. See `code/README.md` for model-dependent reconstruction, configurable inputs and numerical dependencies. The user supplies the GGUF and compatible dependencies; weights are excluded. Selected steering vectors are included in `vectors/`; see its [README](vectors/README.md) and manifest for exact scaling and provenance. The app's register vector is excluded; results quoted for the reference register vector come from stored outputs. Inference-dependent portability has not been tested in this release step.

Limits: round 1 uses one quantized base model, synthetic short Korean dialogs, greedy decoding, 96-token cap, heuristic ending classification and no measured semantic-accuracy rate. Shared templates are not independent examples. S2 searches many neurons, and empirical random-control quantiles are not significance tests. CPU agreement is established only for the stored validation scope, not arbitrary long/pushed generation. Different methods use different injection norms. Round 1 did not compare LoRA steering. Round 2 uses four private adapters, synthetic short dialogs and greedy plus sampled replies, with agreed AI labels rather than human-validated truth. Shared families, small positive-family counts and the R1 distribution shift limit transfer; it establishes no reliable sensor, self-knowledge, corrective intervention or live-app benefit. Round-2 model inference is not reproducible from this aggregate-only addition alone.

## License

Code in `code/` is licensed under Apache-2.0; the complete text is in [LICENSE](LICENSE). Data in `data/`, documentation (including the map page and preregistration copies), vectors in `vectors/`, and generated replies are licensed under CC BY 4.0; the complete text is in [LICENSE-DATA](LICENSE-DATA). See [NOTICE](NOTICE) for Gemma provenance and trademark information. Third-party dependencies retain their own licenses.

[DECISIONS.md](DECISIONS.md) records the 2026-10-27 release decisions. See [SCRUB_REPORT.md](SCRUB_REPORT.md) for the privacy scan.
