# Methods and scope of evidence

Model identity: base Gemma 4 E2B Q4_K_M, GGUF SHA256 `740185b21d22ceb83a11c3aa62ad5842ef32c70f6096d756bbee85a1e4ec34b8`. 35 layers, residual width 1536, vocabulary 262144. FFN widths are 6144 at L0–14 and 12288 at L15–34. Output embedding is tied. No weights or model vocabulary IDs are distributed in data.

## Static lens (S1b, replacing S1)

For write column v, compute `a = output_norm.weight * v / sqrt(mean(v²)+epsilon)`, `x=E a`, and population `z[t]=(x[t]-mean(x))/std(x)`. Group score is the mean z over its explicit whole-token allowlist, including exact leading-space-marker variants. All vocabulary IDs, including special/unused, enter mean/SD. The GGUF norm already contains the model's offset; do not add one again. Actual inference retains softcap `30*tanh(x/30)`; lens scoring/ranking uses pre-softcap x. Original post-softcap S1 register interpretations were withdrawn.

S1b face validity requires ≥10 of a group's top20 neurons to have ≥3 group-token IDs in their top20 tokens. Casual 0/20 and polite 4/20 fail. S6 uses the same lens statistic for four new frozen groups, with a full-vocabulary covariance identity and direct-logit spot checks. A lexical lens association alone is not a causal style effect.

Map columns casual/polite and S6 styles count full-group mean z >1.5, an explicitly exploratory S4 threshold. S1b retained only top/bottom50, group means and full-vocabulary moments. Address groups (explicit S4 multilingual informal-you and formal-you allowlists, including 너/你/ты/তুমি/तुम and 당신/您/usted/आप/আপনি respectively) were not original averaged groups. We therefore use the maximum observed address-token z and certify exhaustive counts only above the largest top50 boundary over all neurons: **z >21.5**, giving two informal and two formal neurons across all layers. Any omitted token is bounded by its neuron's last stored top50 logit. Observed address top-k lists are restricted to retained top50 evidence and are not certified global rankings. Small counts at this conservative threshold must not be read as no address association. Whole-token surfaces can be plural, homographic or used outside second-person address; this map does not resolve context. Exact lower-threshold full-group address maps cannot be recovered from these stored outputs; no new model computation was performed. Group membership strings and cutoff provenance are in `data/lens_map.json`.

## Contextual screen (S2 and S6)

Twelve fixed frames use six paired stems in casual/polite contexts. S2 statistic is `R=logsumexp(polite ending logits)-logsumexp(casual ending logits)`, shift `R(push)-R(None)`, averaged over frames. Screen last prompt position only, reusing exact unpushed prefix. Dose ±.5 qualifies; ±.8 is descriptive. Each layer has 512 original random directions. Original S2 criteria require strict control extrema, sign agreement ≥10/12 and Korean top1 proxy ≥11/12. The 39 narrower candidates additionally clear both-dose control extrema and expected-ending top1 in ≥11/12. S4 computes descriptive p05/p95 by linear interpolation at q*(N−1); these quantiles do not replace the original criteria. Signed ΔR, both tails and extrema are retained on cards.

S6 selects the top40 lens neurons per style, screens target-minus-partner ending groups at +.5, requires mean shift strictly above same-layer 512-control p95, then admits up to two per style. This yields eight admitted neurons; screen admission is not generation success. The S6 screen deliberately retains S2's validated cached numerical path; extraction/generation uses S3b ExactEngine(fma=True). A failed attempt to apply exact-generation operations to the cached screen was corrected before selection. The frozen rule's broad engine wording must be read with this implementation distinction.

## Push engine and numerical validation

`h_l += d * median_residual_norm_l * unit(v_l)` at l_out, after post-FFN RMSNorm, FFN and PLE residuals and layer output scaling. S3/S3b/S6 push all prompt/generated positions. Greedy argmax, lowest-ID ties, ≤96 new tokens, EOG excluded from reply length. Each condition resets state. Bundles use d/sqrt(k) per component; correlated components mean actual vector norms need not be equal. The reference register vector instead uses its raw L4–14 vectors × dose. Its dose is not comparable to residual-normalized neuron doses.

The custom GGUF engine dequantizes supplied weights and reproduces reference CPU activation rounding, quantized integer blocks, fp16 KV/attention, RoPE and libm compatibility tables. S2 versus stored llama.cpp CPU: top1 40/40, top10 mean overlap 9.45/10, max logit difference 2.15581131; S2's 12-frame unpushed top1 is 11/12. Generation validation S3/S3b/S6 matches original ten short CPU references 10/10 through EOG or cap; batched/single and zero-push comparisons also match. Four of the ten S3 references hit the cap. Reference prompts are 20–27 tokens whereas dev prompts are 81–90. Pushed complete generations and arbitrary longer contexts are not established CPU equivalent.

S2 all-position validation has high pooled correlation, but L14–34 equality is structural because of shared KV; early-layer evidence is only six neurons, Pearson .666 and sign agreement 76.39%. Do not generalize pooled correlations to early layers.

## Classifiers, proxies and pre-registration

S3/S3b split on sentence punctuation/newlines, inspect final Hangul words and count fixed suffixes. Unknown fragments stay in denominators. Dominance is strict >.5. A flip requires None source dominance and pushed target dominance. The S3b friend-largest condition uses a strict comparison with the other recognized ending families, plus flip; it requires seven replies, not 70% of fewer flips. Bounded second-person occurrences miss embedded malformed words, so a substring sensitivity count is also retained in source summaries.

S6 s6-endings-1.1.0 precedence: hasipsio, aegyo, haeyo, eumseum, haera, friend, other_casual. Unknown cannot win. Nominal homographs, quotations, malformed or truncated replies remain ambiguous. A build-only change removed 봄 from noun exclusions and added 림/짐/움 before the first validation evaluation. Validation is 126/128: friend 30/32, haeyo 32/32, four styles 16/16 each; other_casual unsupported. The two friend errors repeat one shared content template. Classifier validation pairs do not build directions or select conditions. Pure -다 does not identify social hierarchy; haera_core is a later descriptive count, never a qualification criterion.

Damage measures: Hangul fraction among Unicode letters; distinct character 4-gram fraction after whitespace collapse; generated-token length relative to matched None. Also track invalid bytes, replacements, truncation, repetition and cheap content-word overlap. None is unsteered but not a semantic gold answer. No damage proxy or overlap value is a human meaning-preservation score.

S3 bar: ≥8 flips/10; Hangul≥.95, D4≥.9, length ratio .5–2 each in ≥9/10; paired random ≤2 flips. S3b adds second-person delta≤.5 and friend+flip≥7/10 for casual targets. Polite-target flip ceiling is six due to None. S6 dev: dominance≥14/20; None/random≤3/20; each damage count≥18/20; pronoun delta≤.5. S6 selection is lexicographic by dominance, weakest quality count, lower dose, earlier layer and name. Full grid completed before no-knob reporting. Held-out twelve scenarios are descriptive, with no reselection or new pass rule. Prompt-only is one fixed line per style, not optimized prompting.

## Included frozen rules and hashes

Every local RULE.md, including the superseded S2 token-audit rule, is represented below. Privacy/environment references in the included copies are neutralized. **Original hashes identify the source; they do not hash the redacted copy.** Both hashes are provided, so a reader can verify the distributed text without mistaking it for byte-identical preregistration. Decision thresholds and scientific definitions are preserved. S1b's protocol JSON is included separately. Later S3b reading notes, S6 parent readings and S5a aggregates never change frozen decisions.


| Original source | Original SHA256 | Included copy SHA256 |
|---|---|---|
| S2/RULE.md | `cf5e6e6a2e559711536177fe53f0f947621b9308fdeea61b42993e3639e4a853` | `cbee0277b2a9460131d07d91580c7c162aed3f12d953dcf42b7cf0252b792ba1` |
| S2/_superseded/token_audit_v1/RULE.md | `7dce1d22b6528f78d98472ba62a6451153fc784d732ce65b3b6701f148a29da4` | `c1b8c6868c520c1fc7bf79f371894d9d17e8849baf36e4fc84192916b07a760e` |
| S3/RULE.md | `5b8875c2b0f5eb0ec8bfcba1f3a2ce3260feb6177a01713508ac09b6452665d8` | `ec4cf6dc2d69df5f37e0a023875f19c8aed59bda7aaa915d33ded1ea4815aaae` |
| S3b/RULE.md | `195e3aa31af79c5e0ee413751da42a0faaaa45d03f3578e4011517fe49fc688f` | `1ef44d07e84ade1f81c7ffc279cd17b359273ae7f32480e33e313834c58ac8ac` |
| S6/RULE.md | `320df4ff3a461b03899e167a2d8035a8f6e6ac9876c4d8c19a1ac76853fc9852` | `4a4741a0bc78276196c5afdbec006443d2c04d7ab8396e017f5203eab28c3eac` |

## Release transformations and current verification

Full stage condition summaries are retained. Generated examples are fixed dev dinner/gift contexts and held-out museum/cinema/chess, chosen after the experiments for S4 illustration and held constant across conditions/doses; this is not a preregistered or representative quality sample, nor all generated replies. Full scenario contexts are included. Vocabulary IDs, raw byte hex, checkpoints and model weights are omitted. Selected S3b and all S6 vectors are distributed with exact provenance and application instructions in [vectors/README.md](vectors/README.md) and [vectors/manifest.json](vectors/manifest.json). The app's register vector and its controls are excluded; results quoted for the reference register vector come from stored outputs. Token surfaces matching private patterns are neutralized; 11 identifier-shaped matches in generated zero-pattern text are also masked, marked with privacy_redactions, with metrics retained from the original reply; no rating notes are quoted. S5a aggregate counts describe one reviewer only. Numerical source files retain their arithmetic; paths become explicit work/model configuration, tokenizer CLIs take a model argument, and optional reference-vector loading permits an absent vector while reference-vector arms still require one. Source and adapted hashes are in `code/source_manifest.json`.

This release preparation runs syntax and stored classifier validation only. Model-dependent reconstruction has not been executed; supplying compatible CPU sources/numerical tables remains necessary. `code/README.md` lists that boundary explicitly. No statistical significance, semantic-accuracy or general style-control claim follows from these measurements.
