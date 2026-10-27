# S6 frozen rule — Phase A / version 1

Scope: Phase A preparation and no-push validation only. STOP after the Phase A report.
Phase B requires the parent's explicit resumption after reviewing pairs, classifier and this rule.
No S6 contrastive residual extraction, direction construction, steer-lens computation,
neuron screening, prompt-only generation or nonzero steered generation precedes this freeze.
The Phase A engine validation reuses S3b's original ten stored Mac CPU references.
The freeze UTC and this file's SHA256 are recorded in freeze.json and RULE.md.sha256.
They must be checked on Phase B entry and before resuming any checkpoint.

## Model, engine, intervention and runtime

Use the existing [work directory] engine on ssh [compute host], the existing laptop venv and model
gemma-4-E2B-it-Q4_K_M.gguf, SHA256
740185b21d22ceb83a11c3aa62ad5842ef32c70f6096d756bbee85a1e4ec34b8.
Use S3b ExactEngine (fma=True), the same numeric tables, token-piece decoder,
greedy <=96 new tokens and EOG IDs 1/106. EOG is stored in token_ids and excluded
from generated reply length, exactly as S3b. Use S3b's l_out insertion after
post-FFN norm/residual, per-layer embedding branch/residual and layer output scale.
Directions are added at every prompt and generated position, as in S3b.
All layer indices below are zero based.

Gate: original ten prompts, complete token-for-token equality including EOG on >=9/10;
also require ten batched replies to equal individual generation and exactly-zero
push to equal no push. Engine gate must pass before Phase B. Phase A measured 10/10,
batch 10/10, zero equal. This does not establish pushed or long-context CPU equivalence.
No new Mac CPU inference or tokenization was run in Phase A. Any later Mac CPU step
must be coordinated with the parent holding the Mac heavy lock; do not acquire or
assume release of that lock. Prefer existing laptop tools when equivalent and validated.

Check GPU allocation before model loading, each extraction/screen chunk and every
generation condition; use configured GPU status executable. Pause if external use exceeds
3072 MiB. Under WSL, total used minus this process's torch reserved memory is the
conservative proxy. Poll <=50s per sleep, log all checks and remain in turn.
Phase B aim <=10800 seconds total GPU-job wall time, hard stop at 12600 seconds,
including loading, lens, extraction, screening, validation and generation. Keep a
cumulative ledger across resumed processes. Benchmark first and reserve enough
time for the complete dev and held-out grids. A budget stop with missing coverage
is incomplete evidence, never a completed scientific 'no knob' result.
Checkpoint each condition with full replies, token IDs, measures, input hashes and
completion status. Write [work directory]/S6/logs/DONE_A when Phase A is complete and
DONE_B when the Phase B laptop job ends (include complete/incomplete and reason).

Never delete; move superseded versions. Do not touch [unrelated application directory]. No git, model/package
downloads or network other than ssh/scp. PyPI into the existing own laptop venv is
allowed only if truly necessary and explicitly reported. Phase A added no packages.

## Pairs and directions

pairs.json has 48 authored same-content style/partner pairs per style (192 total).
haera ↔ friend; hasipsio ↔ haeyo; aegyo ↔ haeyo; eumseum ↔ friend.
All four families share 48 semantic topics. Python random.Random(20261026).shuffle
of indices 0..47 fixes first32 as build and last16 as classifier validation.
The same semantic index has the same split in all four families; split.json is
authoritative. No S3 scenario topic or held-out scenario supplies a training pair.
Validation replies never contribute residuals, direction means, lens or neuron screening.

For each style and layer in {8,14,20,27,31}, teacher-force prefix + complete authored
reply with no push. Capture the l_out residual at the final reply token, before EOG
or template suffix. Compute mean of the 32 (style residual - partner residual)
differences, with equal weight per pair. Unit-normalize and multiply by that layer's
existing S1 token-weighted median residual norm from S1/residual_norms.json.
Record exact raw means, scaled vectors, norms and hashes. No vector from the 16
classifier-validation pairs or 12 held-out scenarios is permitted.

## Style neurons and screen

Use the S1 steer lens with four new explicit whole-token groups in token_group_plan.json.
Curated ending pieces are lexical allowlists, including exact space-marker variants;
never substring matches. Unavailable forms remain recorded as missing, without
adaptive replacements. The token inventory is the existing S1 vocabulary; no downloads.
Use S1b's pre-softcap score: (group mean logit - full-vocabulary mean) /
full-vocabulary population SD over all 262144 IDs, including special/unused IDs.
Evaluate write directions of all layers/neuron indices, including layer0. Rank each
style by descending score, ties earlier layer then neuron index; top40 distinct
neurons per style enter the next-token screen. No outcome-dependent token edits.

Screen at positive d=0.5 with S2's twelve stored frames and last-prompt-position-only
cached-prefix insertion. Use logsumexp(target-group logits) minus
logsumexp(partner-group logits), pushed minus None, averaged over twelve frames.
Groups use the no-leading-space token IDs from the explicit plan; the target/partner
groups must be nonempty and disjoint. Report both groups and the shifts per frame,
KL(push||None), entropy delta and S2 Korean-continuation proxy. These are ending-piece
statistics, not grammatical probability or human style ratings.

Compare each candidate to the original 512 S1 same-layer random directions at +0.5,
normalized and scaled to the same median residual norm. The screen threshold is
strictly greater than their empirical p95 (linear interpolation at q*(N-1)).
Take up to2 qualifying neurons per style, ranked by descending mean target-minus-
partner shift; ties earlier layer then neuron index. No added sign search or adaptive
doses. No qualifying neuron means an empty neuron arm set, not a failed direction test.

## Dev grid and controls

Use unchanged twenty S3 scenarios in scenarios_dev.json: ten polite-context and ten
banmal-context. Every constructed direction and every admitted style neuron is
generated at each d in {0.15,0.25,0.4,0.5,0.65}, always positive toward the target.
This is five direction layers × five doses per style, plus <=two neurons × five doses.
Use all20 scenarios for every condition, regardless of context register.
Also run shared None and four prompt-only conditions; prompt-only cannot be selected.
Prompt lines are fixed in prompt_lines.json, appended with one ASCII space to the
unchanged scenario system line. No in-context examples or alternative wordings.

Each direction/neuron condition has an exact-norm, same-layer random control at the
same dose, using S1 control index317 (zero based) at its layer. Renormalize to the
actual stored target vector norm before multiplication by d; record float tolerance.
An identical numerical control may be reused with all named pairings recorded.
Do not substitute a different random direction based on outcomes. Neuron screen p95
uses all512 controls; the generation grid's paired control is this one fixed vector.

## Classifier and measures

Classifier version s6-endings-1.1.0 in tools/style_classifier.py; source SHA256 and
complete suffix lists, precedence and noun exclusions are in classifier/spec_v1_1_0.json.
Sentence splitting and trimming are identical to S3/S3b: split on [.!?…\n]+; omit only
initially whitespace-empty segments; strip spaces then trailing Unicode punctuation /
symbols and ~^ㅋㅎ; inspect the final contiguous Hangul word. Segments that become
empty or have unknown endings remain in the denominator. No quote/code filtering.

Labels: friend, haeyo, haera, hasipsio, aegyo, eumseum, other_casual, unknown.
Precedence: hasipsio, aegyo, haeyo, eumseum, haera, friend, other_casual, else unknown.
A dominant STYLE must have count>0.5 × all nonempty segments and be strictly largest
among style counts. Unknown is not a style and can never win. Ties / no majority /
empty replies are unknown. Shares always use all segments including unknown.
Broad homographs and nominalized/noun ambiguity remain documented limitations.
Known S3b misses 돼/어때/줄래/-ㄹ게 are covered. Shared -자 is friend in S6.

Build-only revision v1.0→v1.1 removed 봄 from noun exclusions and added 림/짐/움
after observing build eumseum 24/32. No validation was inspected before this edit.
First and final validation: friend30/32 (93.75%); haeyo32/32 and each target label
16/16 (100%). All supported labels clear >=90%. The two friend errors are one
shared reply with 넘겨, counted in two families; no post-validation tuning.
other_casual has no gold support in the requested four families and is NOT validated.
These held-out authored replies do not validate arbitrary generated-reply accuracy.

Retain all S3b measures from its measures.py, measures_extra.py and diagnose_checks.py:
old register/flip and app checker as descriptive companions, Hangul fraction among
Unicode letters, character distinct-4 and repetition, max word/char4 repetition,
generated-token length ratio to matched None, other scripts, replacement chars,
broken UTF-8 bytes, truncation, content-word Jaccard/recall against None and user.
Content overlap is a proxy and never a semantic-accuracy score or pass criterion.
Keep S3b's orthographically bounded second-person occurrences of
너/네가/니가/너의/너는/너도 with optional particles; also report its substring sensitivity.
Second-person delta = mean(condition count - matched None count) over all20 dev replies.
Add all seven style counts/shares, unknown share and S6 dominant style.
Do not substitute S6 counts into old S3b/S3 findings. The separate S3b comparison
report covers only stored None, friend_last+.5/+.65 and N1939+.5; no generation/recount.

## Dev bar, selection and near-miss

For EACH condition over the same twenty dev scenarios, all of the following must hold:

* Target style dominant >=14/20.
* Target style dominant in None <=3/20 AND paired random <=3/20.
* Hangul >=.95 in >=18/20, distinct-4 >=.9 in >=18/20, token length ratio in [.5,2]
  in >=18/20, assessed separately.
* Mean second-person delta <=.5 occurrences per reply.

Selection among passing direction/neuron conditions of each style is lexicographic:
highest target-dominant count, then highest min(Hangul-pass,distinct4-pass,length-pass),
then lower dose, earlier layer, lexicographically smaller canonical condition name.
Here 'smallest of the three damage counts' means the weakest of the three reported
quality-pass counts; larger is better. Canonical names: style.dir.L08 etc and
style.neuron.L27.N01939 (fixed zero padding). No human read, prompt-only performance,
content proxy or held-out outcome changes selection.
Best near-miss = nonpassing selectable condition ordered by the SAME dominance,
weakest-quality, dose, layer, name key. Report all failed checks; near-miss is not a pass.
If no condition passes after the complete dev grid, report 'no knob' plus best
near-miss and do not create that style's held-out knob, prompt+knob or knob-control arms.
If no nonpassing condition exists, label near-miss unavailable. Missing grid coverage
is 'incomplete', not 'no knob'. None/prompt/random conditions are never selectable.

## Held-out design and reporting

scenarios_heldout.json fixes twelve new scenarios: six polite-context and six
banmal-context, same base system line and S3 chat template, one preceding user and
assistant exchange (two earlier turns) plus final user turn. Topics differ from S3
and the48 pair topics. No held-out scenario generates residuals or screens neurons.
For each style run None, fixed prompt-only, selected knob, prompt+knob and the knob's
paired exact-norm random control. None may be shared; a no-knob style still gets
None/prompt-only. Prompt+knob uses the SAME layer/dose/vector with only the fixed
system addition. Random control has the base system line. All lengths/proxies and
second-person deltas compare to the same unprompted, unsteered matched None.
Report the same measures over all12 and by six-scenario context; no new selection
or qualification cutoff on held-out data. Show all12 in review_heldout.md.

review_selection.json freezes the later review page's six scenarios: museum,
bicycle, audio (polite), cinema, tent, microphone (banmal), three per context.
The page is produced only after the parent's Phase B reading. No Phase A review page.
review_dev.md shows all20 replies of each selected condition and its best near-miss,
next to matched None and prompt-only, with every failure and measure disclosed.

Phase B maps: map/style_layer.tsv (style × layer × dose, target dominance and H/D/L
counts, exact-norm control counts, second-person deltas and all bar checks);
map/style_neurons.tsv (lens, full screen, qualifying/admitted neurons and dev results);
map/cosines.json (all style pairwise cosines at matching layers, plus N1939 at27,
N6893 at26, S3b friend_last at27 and original app vector at matching layers4..14).
No matching layer is explicit null/not applicable, never silently compared across layers.
Persist generations.tsv.gz, summary.md, review_dev.md, review_heldout.md, tools/,
logs/, SHA256SUMS.txt and plain Korean README_ko.md with [잰 값]/[읽기], deviations,
incomplete coverage and limitations. Mirror final artifacts to the Mac S6/.

Freeze manifest covers rule, pairs/split, scenarios/review choice, prompt lines,
classifier/spec/revisions, token-group plan and referenced engine/measurement inputs.
Phase A checksum manifest excludes its own manifest, caches and superseded versions;
preserve superseded files outside that canonical manifest. Phase B writes a new
version of the overall manifest by moving the old one, without changing this rule.
