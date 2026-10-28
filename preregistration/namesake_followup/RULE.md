[Release copy: private identities, package references and environment details redacted; original and copy hashes are recorded separately.]

# Who-is-who namesakes: frozen evaluation, 2026-10-28

## Parent decision (Resume 4), before generation

Section 3 is green for this change: related 145 / skipped 5 / failures 0; full 3,395 / skipped 19 / 21 failed
assertions in the same 13 pre-existing failing tests (new=[], noLongerFailing=[], exactlySame=true); Release succeeds.
This is not an entirely passing full suite. App sources are frozen; no further app edits are authorized here.

The original all-160 parity gate stopped: 54 app-off notes differ from N0c arm a. Preserve that failure and its
outputs unchanged. Parent inspection of N0c build.py established that N0c arm a uses handwritten family user_things,
the collision filter, and N0b's Python who_note: an idealized ledger, not the shipping app reader. Fifty differences
are unrelated reader-coverage gaps, out of scope. Four friend_same_pet rows carry a scripted third-party pet relation, which the
app never produces. This mismatch is the reason for the revised design, not a reason to repair or tune the reader.

App notes change under the switch in 50 scenarios. Main set: the 46 changed scenarios where app off equals recorded
N0c arm-a note byte-for-byte. Reuse their 138 recorded arm-a replies as baseline. Generate only a′ from arm-a prompt
with the app on note substituted. Verify prefix/suffix and all non-note text remain identical. Side set: the four
friend_same_pet scenarios, generating BOTH app off and app on, three decodes each. Side results never enter bars.
Unchanged scenarios and the 50 unrelated off differences are not generated. No expansion based on observed labels.

Reproduce sets without inference: import _tools/prepare_eval.py and call sets(), which selects only from
eval/app_notes.json. eval/changed_scenarios.json contains main/side arrays. Failed parity artifacts remain immutable.

## Frozen sets

Main: 46 collision scenarios. Side: four scenarios, excluded from bars. Private scenario IDs are withheld.

## Model, renderer and decoding

Same N0c Q4_K_M base and four package adapters, scales and exact-engine implementation. Model manifests below are
copied, not resolved from a provider. Prompt rendering is the already frozen N0c stable-head/Gemma text with ONLY
the who note replaced, so template, first-sitting note, history and current user text are fixed. Mac tokenization
uses existing llama_reference tokenize in vocab_only mode (skipping tensors), never Mac model inference.

Compute machine only: [local-workspace], existing N0 LoRAEngine and N0b dynamic attention kernels. No [local-workspace]
reads/writes or venv use. New files only under [local-workspace] Check GPU before starting and stop if a
GPU workload exists. N0c guard checks before every 64-token prefill chunk and every 16 decode tokens, pauses 30 s
while external memory (total minus own torch reservation) >3072 MiB. A game therefore pauses the run. Do not kill
another process or release/acquire the parent's lock.

Before generating a′, replay one preselected N0c arm-a prompt per character, greedy AND both samples (12 replies),
requiring identical token IDs, raw bytes and decoded text against N0c. Any difference stops generation. Immutable
KV prefix reuse is allowed only at complete 64-token boundaries, preserving per-prompt numerical schedules.

Every generated prompt: greedy temperature 0 seed null; s0 temperature .8 seed 20261027; s1 temperature .8 seed
20261028. Fresh CPU torch.Generator per sample, full-vocabulary multinomial; no top-k/top-p/min-p/repetition penalty.
GPU logits, cap 96 tokens including EOG; stop IDs 1/106. Record tokens, raw hex, truncation, hashes, decode timings,
prompt IDs and model/adapter/driver versions. Main new replies 138; side new replies 24; total new 162.
Baseline 138 replies are re-judged, so the blind judge universe is 300 replies (before exact duplicate collapsing).

## Independent judges, blind to arm

A=gpt-6.1-sol, B=gpt-6.1-terra. Fresh independent sessions, fork_turns=none; as in N0c, up to three disjoint shards
per model. Each reads full canon plus complete context/reply. Never read old labels, other judge's decisions, this
RULE, mapping, generation metadata or parent outcome annotations. No network/delegation by judges. Exact duplicate
(character, full context, reply) units may share labels with explicit mapping; never clone across different contexts.
Shuffle opaque IDs with seed 20261028, assign complete scenario blocks containing both arms and all decodes to a
packet, then independently shuffle A/B orders (seed 20261029/20261030 + packet index). Main baseline and a′ are
re-judged in the SAME packets/sessions. Side off/on is judged in the same protocol and reported separately.

R1 is ownership misattribution only. Preserve every owner-bearing reference span, true_owner/surface_owner
(user|character|third_party|shared|unknown), evidence, slip bool and decisive exact reply span. Unsupported ordinary
canon/self-life alone is not R1. Same spelling can mean different entities. Ambiguous ownership stays unknown,
with no forced positive. No heuristics, proxy labeling, cross-judge adjudication or old-label reuse.

Additional question per reply: note_echo {yes: bool, span: exact reply span, reason: string}. Does it repeat the
note's wording or speak unnaturally about names or the note? Natural remarks that two people share a name are NOT echoes. Each model judges this independently; preserve disagreements and spans. No automatic echo word matching.

Only agreed slip/echo yes/no labels count; disagreements remain unknown, never negatives. Preserve both complete
annotations and report agreement coverage per arm. Scenario score is the count of agreed R1 slips among its three
decodes (0..3), with unknown counts reported beside it. Also report unknown-label sensitivity, without replacing
the preregistered bars. These are independent AI judgments, not human-verified truth.

## Bars, MAIN SET ONLY (all must hold)

Unit: scenario = character × family. Compare the agreed-slip scores (3 decodes). Among scenarios with unequal
scores, a′ must improve more often: one-sided exact sign-test p=sum_{k=better..better+worse} C(n,k)/2^n <= .10,
and net better = better-worse >=5. Ties excluded from sign test; if no unequal scenarios, p=1.
Total agreed slips a′ <= .7 × a. Agreed note echoes <=3% of the 138 main a′ replies. Unknown echoes are reported
separately and never treated as agreed no. Side rows are excluded from all three bars.

Report per character and per original N0c kind (the selected main rows may all be collision). List EVERY scenario
whose a′ score is worse, with both arms' complete three replies and independent labels. Report the four side
scenarios separately. List all 50 unrelated reader-gap rows by family, with at least one actual setup example per
family. Do not fix those gaps. Record measured counts separately from code reading ([잰 값]/[읽기]).

No app/default/iPad/ops/AGENTS changes, downloads, deletion or extra networks. Preserve prior records by moving to
_superseded/. Freeze this RULE and input SHA-256 manifest before replay or generation. Rules cannot change after
outcomes; any further parent decision must be separate and preserve this version.

## Exact model manifest

[Private source paths and adapter metadata withheld. The base model identity is in METHODS.md.]
