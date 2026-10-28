# Round-2 aggregate data

All new round-2 files contain aggregates only. Character labels A–D are consistent opaque bench labels; no private package descriptions or generated replies are included.

| File | Contents |
|---|---|
| `round2_sensors.json` | All 12 category×timing×capture-variant summaries; selected layer/feature/C; word/character/surprise baselines; primary shuffle maxima; operating precision/recall, CIs and bars; aggregate character/arm and R4 subtype results |
| `round2_counts.json` | Raw N0b category/split counts and AI κ; raw N0c counts including kind×arm; deduplicated reply and greedy-prompt sensor counts |
| `round2_owner.json` | Owner-probe aggregate baselines/readouts and separate counterfactual summaries, with no cases |
| `round2_namesakes.json` | Optional input-note follow-up aggregate table, unknown-label sensitivity and frozen bars |
| `round2_sources.json` | Source-file SHA256 fingerprints for the aggregate extraction |

Raw reply counts include repeated conditions/decodes; reply sensors collapse exact character/full-context/reply duplicates. Pre-generation units use the agreed greedy label per character×scenario×arm prompt. Disagreement is unknown, never a negative. `threshold: null` means +infinity because no dev threshold qualified; it is not a missing or retunable threshold. Zero detections have reported precision 0 and fail. CIs use family bootstrap; the shuffle maxima are dev OOF, not held-out. LoRA is primary, base auxiliary. R2 is descriptive only. N0's mixed historical labels and all individual human marks/words are excluded from this addition.

No activations, token IDs, adapters, checkpoints, private package files, app code, note text or round-2 sample replies are present. The inherited round-1 data and authorized steering vectors retain their original scope. Protocol redactions and both source/copy hashes are recorded in `preregistration/round2_manifest.json`; source hashes do not hash the distributed redacted copies.
