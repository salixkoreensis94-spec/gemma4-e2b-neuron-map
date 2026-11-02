# M2a design freeze — 2026-10-29

This file is frozen before any M2 sample or vector is drawn. PLAN_v3_ko.md §M2
is the source of the four categories and decision thresholds. Main judging
requires the parent's review. No judging in M2a; no main or pilot packets are
written until the capture is complete. The three-neuron demonstration is a
local, unjudged software fixture, excluded from both experimental samples.

## Population, samples, seeds

35 layers, widths 6144 (L0–14) and 12288 (L15–34). Main: 40 uniformly sampled
neurons per layer = 1400, 6 Gaussian directions per layer = 210, and 6 mixed
items per layer = 210. Pilot: 4 neurons, 1 direction, 1 mixed item per layer =
210 items; do NOT trim. Pilot never enters a main statistic.

NumPy Generator(PCG64), seed 2026102901, ascending layers: draw one permutation
of each layer's neuron indices. First 40 are main, next 4 pilot; next 12,
paired consecutively, are main mixed donors; next 2 are pilot mixed donors.
Thus no sample neuron or donor overlaps any other neuron/donor in that layer,
including across pilot/main. A direction is not a donor neuron: its Gaussian
support necessarily includes the full layer. Independent direction columns
0–5 are main, column 6 pilot. Vector RNG is separate PCG64 seed 2026102902,
ascending layers, one float64 standard_normal([width,7]) draw per layer.
Save exact float32 vectors, population stds and sample JSON with SHA256 before
capture. Opaque packet identifiers use PCG64 seed 2026102903. Judge shuffle
seeds are 2026102911 Claude, 2026102912 GPT, 2026102913 Gemini.

## Direction formula and numerical checks

Let A[t,j] be the signed GELU(gate)*up activation before ffn_down, excluding
BOS, and s[j] = population standard deviation (ddof=0), evaluated in float64
over M1's fixed 8192 full activation rows. Let valid[j] = s[j] >= 1e-6.
For independent g[j,k] ~ N(0,1), set g=0 on invalid dimensions and define

    V[j,k] = float32(g[j,k] / (s[j] * sqrt(sum_valid g[:,k]^2)))
    Y[t,k] = FP32_matmul(A[t,:], V[:,k])

Invalid coefficients are zero. The norm/division is evaluated float64 before
one float32 cast. No activation centering, PCA, optimization, direction
selection, post-hoc sign flip or unit-variance rescaling of Y. Frozen vectors
are the definition; sign is the drawn sign. Positive means Y>0.
This controls large marginal activation scales, as specified in the planning brief; it does not
whiten correlated neurons, so correlated groups can still dominate. Whitening
would be a different control and is NOT substituted.

Same unchanged M1 d_fast_graph_batch1 engine, FP32 matmul/attention, TF32 off,
autocast off, independent chunks with one BOS=2 and positions reset. Project
all seven directions together per layer. On four deterministic chunks (first
two ko and first two en in dev order), independently recompute full rows and
project again: abs error <= 2e-5 + 2e-5*abs(reference), all coefficients.
Also compare against float64 A@float64(V), with forward-error bound
gamma_width * sum(abs(A*V)) + 2e-5, gamma_n=n*eps32/(1-n*eps32).
Same tolerance for M1 reservoir-row projection versus recomputed activation
row projection (GEMM row shape can change reduction rounding). Nonfinite
values, any tolerance violation or changed frozen input: STOP before full pass.

## Evidence packets (same for every kind)

One packet: 20 top windows and FOUR empirical activation-rank bands, SIX
windows each (44 total). Six instead of M1's five permits exactly 3+3 donors
in EVERY mixed band; all real neurons/directions also use six, drawn afresh
from their unchanged 8192-position reservoir, never a new model sample.
Bands are [0,.5), [.5,.9), [.9,.99), [.99,1], using stable sort by activation
then global position. Within each band, candidates are a fixed PCG64
permutation seeded with SeedSequence([2026102904,layer,local_item_id,band]).
Top candidates: exact distinct-chunk maxima, descending activation, smaller
chunk ID for ties, earliest content offset within chunk. After the recorded
20, fallback candidates are reservoir positions descending activation, then
smaller chunk ID/offset. Fallback is explicitly audited internally; it must
not be described as an exact stream top-20 replacement. Bands use the band
permutation to replace dropped candidates. Exclude duplicate chunks across
all 44 windows. If insufficient, mark packet INCOMPLETE, do not silently
change filters or show it to judges. Report completeness by kind before judging.

Mixed: donor0/donor1 alternate, starting donor0 in each top list and each
band: 10+10 top, 3+3 per band. Each donor's candidate ranking and band
definition are its own; no fabricated scalar activation. If candidate dropped,
advance only that donor's list. Donors do not appear in any other item.

Decode full content chunk using GGUF vocabulary, run unchanged M0 chunk-wide
PII patterns (phone, resident registration, email, accounts, addresses) BEFORE
extracting any window. Reject the whole chunk if any pattern matches. Window
is ±16 content tokens, firing token marked ⟦...⟧, clipped at chunk ends.
Template rule applied identically to all kinds and across all evidence slots:
NFKC + lowercase window text, replace each digit run with #, collapse all
whitespace, remove punctuation; form sets of character 5-grams (whole normalized
string if shorter). Jaccard >= .70 to an existing family's first accepted
representative joins that family; otherwise starts a family. At most TWO
accepted windows per family per packet. Process top then ascending bands,
candidate order as above, deterministic first-family tie rule. Record drops,
fallbacks, source/license/generator provenance locally, hidden from judge.
This reduces, but does not prove elimination of, semantic/template dependence.

Judge sees only opaque ID, top/band slot headings and marked windows. No item
kind, layer/neuron, activation/frequency, source statistics, output lens or
automatic concept label. Source/license provenance stays local. Pilot sends
only packet text to the three authorized judges. Corpus/packets stay local
otherwise. No publication.

## Frozen decision statistics and thresholds

Categories: a=one concept; b=multiple meanings with a common concept (name it);
c=multiple unrelated meanings; u=unreadable. Each judge chooses exactly one,
with a short concept name (empty for c/u) and a short explanation.
Three judges: majority >=2/3 identical category is the item outcome; a 1/1/1
split, missing or invalid judge result is unknown. Unanimity sensitivity uses
only 3/3 category agreement. Two judges, if panel changes after pilot: only
2/2 category agreement counts; otherwise unknown. Unknowns stay visible and
never become c or leave the denominator of the mixed validity bar.

Primary p=b/(b+c) among agreed neuron items, Wilson score 95% interval,
z=1.959963984540054. Zero b+c => undefined => inconclusive.
p lower>0.5 supports H only if BOTH control conditions below pass; p upper<0.5
rejects H only with valid controls; otherwise inconclusive. Random-direction
condition: p_neuron-p_direction has Newcombe independent-Wilson 95% interval
with lower>0. Specifically lower=d-sqrt((pN-LN)^2+(UD-pD)^2),
upper=d+sqrt((UN-pN)^2+(pD-LD)^2), clipped [-1,1]. A zero denominator in either
arm is inconclusive. Mixed validity: agreed c / ALL planned mixed items >=.80
(main210, pilot35); missing, unknown, a/b/u are failures for this bar. Also
report Wilson 95% interval, but the bar is on the observed fraction, not its
lower bound. Pilot and main are separate tables, NEVER pooled.

Report Fleiss' kappa for three complete valid judge ratings, nominal four
categories, and all three pairwise Cohen kappas; with two judges Cohen kappa.
Report denominators, missingness, raw agreement, majority/unanimity coverage;
degenerate zero expected-disagreement => undefined, not perfect agreement.
Kappa has no additional invented pass threshold. Pilot is feasibility/usage/
agreement information: parent and the reviewer choose whether main judging proceeds.
Layer trends descriptive only. After pilot ONLY panel, judge prompt and packet
format may change (version and disclose); samples, controls, statistics and
thresholds remain frozen. No outcome-based replacement or new sample draw.

## Capture, guard, durability, storage and stopping

Same 49438284-token,263910-chunk dev schedule/order as M1; heldout excluded.
Keep per direction the M1 hist[1024] int32, total float64, positive int64,
maximum float32, language/script positive counts, top20 float32/int32/int16,
count int64. Uniform reservoir positions are M1's seed20261028 positions;
store projection[8192,7] float32/layer, compare against A_res@V at completion.
Checkpoint before a whole chunk exceeds a 1M-token gap, exact 1M, SIGTERM,
guard pause and completion. COMMIT-last + SHA256 + committed reservoir prefix
hashes, cumulative snapshots, next_row cursor; no adding checkpoints, no double
count. Retain older files in _superseded. Pid identity includes boot/start ticks;
exclusive locks and DONE.json, no pgrep. Relaunch frozen launch.py after reboot.

Worker polls at chunk boundaries at least once/sec; external=max(0,totalGPUused
MiB-ownTorchReservedMiB). At external>3072, commit at safe chunk boundary and
EXIT with dedicated status75, releasing every CUDA allocation/context. Supervisor
is CPU-only; after pause/crash/initial launch, require total GPU used <2500 MiB
continuously for300 seconds, sampled every10 sec (any high/error resets timer),
before starting worker. Query error while running => checkpoint+exit, fail closed.
During checkpoint no next chunk launches; CPU synthetic tests use shortened
timings/fake telemetry with real worker exits and supervisor restarts. Duplicate
supervisors excluded, uncommitted tail replayed. Guard/SIGTERM process exit
guarantees VRAM release, rather than holding allocations while sleeping.

C: must remain >=15000000000 bytes. Before setup/start/checkpoints check actual
Windows C free space. Bound all M2 capture input/checkpoint/history/reservoir/
logs to a conservative 1GB normal-run budget; logs/checkpoint disk gate leaves
at least100MB safety above15GB, exits with durable BLOCKED marker if insufficient.
No cleanup/deletion. Excess restart history may stop capture for parent review.

M2a stops after first verified checkpoint and reports PID/start/ETA/status.
Packet generation follows capture in a later resume. Judge runner is prepared,
batched (default6 items/call), one shared prompt and strict JSON schema, separate
fixed judge shuffles, empty working directory, no tools, read-only execution,
per-call raw usage logs and explicit missing-usage failures. No judge calls now.
Main pilot/model identity must be verified by parent before paid execution:
Claude Opus5.5, codex gpt-6.1-sol/high, Gemini gemini-4-argon-high. No substitution.
