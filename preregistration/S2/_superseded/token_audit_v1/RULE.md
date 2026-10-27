# S2 pre-registered decision rule

Frozen UTC: 2026-10-25T19:25:30.114069+00:00

Model: app GGUF gemma-4-E2B-it-Q4_K_M.gguf, SHA256
740185b21d22ceb83a11c3aa62ad5842ef32c70f6096d756bbee85a1e4ec34b8.
No Hub configuration or weights. No generated responses or sampling.

Engine gate: compare full post-softcap next-token logits against the S1
llama.cpp CPU runtime on all 40 saved S1 token sequences. Report maximum
absolute logit difference, top-1 equality count and mean top-10 intersection.
Do not screen unless top-1 = 40/40 and mean top-10 intersection >= 9/10.
Additional checks: zero push, restoration, cached-prefix vs full forward,
and representative pushed logits against the S1 CPU control-vector path.

Push: h_l += d * median_S1_l * W_down[:,n] / ||W_down[:,n]||_2,
d in [-0.8,-0.5,0.5,0.8]. Apply at l_out: AFTER post-FFN RMSNorm,
FFN residual addition, per-layer embedding branch and layer output scaling,
immediately before the next layer (or final output RMSNorm). This is S1's
sanity-check insertion site. Include layer 0 if the implementation supports it.
Screen the last prompt position only; cache the exact unpushed prefix.
Use the original 512 S1 control vectors per layer, normalized to unit length.

Frames: 12, comprising six paired sentence stems in a casual friend context
and a polite acquaintance context. No outcome-dependent frame or token edits.
Explicit candidate endings are fixed in frame_candidates.json. Before any
forward evaluation, the actual GGUF tokenizer checks that appending an ending
preserves the prompt token prefix and adds exactly one ordinary token.
Only such continuations enter frames.json, with excluded candidates recorded.
Each frame must retain nonempty disjoint polite and casual ID sets, or stop.

Per-frame register statistic R = logsumexp(log p[polite IDs]) -
logsumexp(log p[banmal IDs]); shift = R(push)-R(base).
The common log partition cancels, so the implementation may use logits.
Damage: KL(p_push || p_base) over all 262144 vocabulary IDs, entropy(push)-
entropy(base), top-1 ID and surface, automatic Korean-continuation flag.
The automatic flag requires an ordinary token with at least one Hangul
syllable and only Hangul syllables/Jamo, whitespace and .!?~… punctuation.
This script check is a proxy, not a judgment of grammatical/contextual fitness.

S3 PASS requires at least 10 DISTINCT neurons. For at least one tested dose
d in [-0.5,+0.5], let M be the arithmetic mean of the 12 frame shifts.
A positive candidate must have M > max(M of all 512 same-layer controls at
that same dose), with shift > 0 in at least 10/12 frames. A negative candidate
must have M < min(controls at that same dose), with shift < 0 in >=10/12.
Top-1 must pass the pre-fixed Korean-continuation proxy in >=11/12 frames.
Strict comparisons; zero shifts do not agree. Deduplicate neurons across doses
and directions. Rank each direction by absolute mean shift, tie by layer,
neuron, then dose. The +/-0.8 doses are descriptive and cannot qualify for S3.
If coverage or engine validation is incomplete, S3 is NOT ESTABLISHED rather
than a scientific FAIL. Report measured FAIL only after the complete screen.

Always report L26 N6893, L27 N1939, L28 N3794 and L34 N6703 at all four doses.
Top report: top 30 per direction at |d|=0.5 (one best entry per neuron),
including per-frame shift and top-1 token; separately mark qualifying neurons.

All-position validation: 50 strongest distinct neurons by maximum absolute
mean shift at |d|=0.5, plus 50 uniformly random distinct neurons outside that
set (NumPy PCG64 seed 20261026). Evaluate all four doses and all 12 frames,
pushing all prompt positions. Report pooled Pearson and Spearman correlations
of last-only and all-position shifts, separately by subset and dose, and
sign agreement. Record exact selection and per-frame paired observations.

Benchmark before screening; estimate complete neuron+control cost from early,
middle and late layer batches, with output-head and metrics included. Aim at
<=8 GPU hours including validation. If estimate exceeds this, optimize and
re-benchmark before committing. Checkpoint complete layers with hashes.
Resume must verify RULE, frames, model, engine and control input hashes.
Before each layer query configured GPU status executable. If external GPU allocation
exceeds 3072 MiB, pause and poll every 180 seconds; record all checks/pauses.
In WSL, missing per-process memory accounting uses total GPU memory minus
this process's reserved memory as a conservative external-allocation proxy.

Never delete files. Move superseded versions to _superseded/. No git.
Network only ssh/scp to [compute host] and PyPI installation on that laptop.
No changes to voice, app, other research, or chat/bot data.
