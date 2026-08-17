# llama.cpp — Metal tuning for Qwen3.8-27B on an M4 Max

A fork carrying Metal-backend changes aimed at one model on one machine: **Qwen3.8-27B (Q4_K_M) on
a binned 32-core M4 Max, 369 GB/s measured, 36 GB unified**. Branch `shipped` is the tuned stack; `master`
tracks upstream `ggml-org/llama.cpp` and is kept in sync so the diff stays readable. Synced through
**`89e0aa6fd`**.

**This file is called `QWEN36-METAL.md` and the model is Qwen3.8.** The work started on Qwen3.6-27B
and moved to Qwen3.8-27B on 2026-08-14. The filename, the parent directory and the older records
still say 3.6 on purpose, because that is what they measured. Every figure below names the model it
came from.

**Nothing here is a general llama.cpp speedup claim.** It is one model on one chip, and some of it is
tuned to both. Five things here are worth upstreaming on their own anyway, and are called out
below: the `dequantize_q4_K` scale, the `test-backend-ops` coverage block, `nr1_k(5) = 3`, the
unguarded `n_max` clamp, and one line of CMake.

## Where it stands

Measured 2026-08-16 on Qwen3.8-27B Q4_K_M, read off **one session** on the tuned build, at the
shipped operating point, with the workload proven identical to the Qwen3.6 reference.

| | this fork, Qwen3.8 | Qwen3.6, earlier stack |
|---|---|---|
| prefill (pp512) | **203.43 tok/s** | 203.37 |
| decode, speculation on | **33.137 tok/s** | 30.262 |
| decode, speculation off | 18.350 | 18.513 |
| speculation gain | **1.806×** | 1.635× |

Decode is speculative (MTP, **depth 3**, `p_min` 0.6): acceptance **0.8658**, **3.03 tokens per
forward pass**. Acceptance on Qwen3.8 is 5.97 pp below Qwen3.6 and that is real, but the draft policy
compensates by drafting more, so tokens per forward end up *above* Qwen3.6's 2.938.

**Read the second column carefully.** It is the same fork on the older model, but it is not the same
stack: it predates `L1-l16p`, `MM-pipeline` and an armed `J-frspec`, and it was measured on the
mis-built binary described below. It is here as history, not as a model-to-model delta.

**Prefill moves about 2 % with thermal state.** 203.43 is the headline session; the best-conditioned
prefill measured the same night was 206.9 on a cooler machine, on the same stack. Take prefill first
on a cold machine or do not take it. This is why `MM-pipeline`'s +1.36 % below was established
*within one binary* rather than by comparing two sessions.

**What ships.** Upstream `89e0aa6fd` plus, in the order they landed: `B-nr1` (§0), `E-gdnfusion`
(§0), `J-frspec` (§7), `N_R0_Q5_K_R1=2`, `A5-q6nr0` (§6), `A10-q4l16` (§1), `T-q4ksubnormal` (§5),
`MM-tile32` (§3), `L1-l16p` (§2) and `MM-pipeline` (§4). Operating point `--spec-draft-n-max 3
--spec-draft-p-min 0.6`. Kill switches: `GGML_METAL_NO_Q4_L16`, `GGML_METAL_NO_Q6_NR0`,
`GGML_METAL_NO_L16P`, `GGML_METAL_NO_MM_PIPELINE`. `MM-tile32` has none and needs an `ne11` gate
before it goes upstream.

**Losslessness is 7/10** — greedy speculative output is byte-identical to non-speculative output on
seven of the ten frozen prompts. That is a property of the design, not a regression; "Known limits"
item 2 explains the mechanism. It was 9/10 on Qwen3.6 at depths 2 and 3.

**The physics, and both roofs came in under their derivations.**

- **The bus is 369.3 GB/s measured, against 409.6 derived** (`8533 × 384 / 8`). Every decode ceiling
  in this fork's history divided by the derived figure.
- **A decode pass is not pure streaming.** It is **43.15 ms of weight reading** — fixed at any batch
  width, and 99 % of the measured roof — plus **11.99 ms per verify column**, which runs at 97 % of a
  dequantising arithmetic floor of 5.30 TFLOP/s. So one token per pass costs 53.94 ms and the wall is
  **18.54 tok/s**, not the 25.48 an earlier version of this file reported. **Unspeculated decode
  measures 18.350 — 99 % of its wall, not 72 % of an imagined one.**
- **The closed form that follows** is the most useful number here:
  `ms/token = 43.15 / tokens_per_forward + 11.99`. It predicts unspeculated decode at 18.14 against
  18.350 measured **without ever being fitted to it**, and puts the ceiling at infinite verify width
  at **83.4 tok/s** — of which today's 33.137 is 40 %.
- **On prefill there is no matrix unit on M4** (`matmul2d` lowers onto the ordinary shader path), so
  the ceiling is the scalar roof, **measured** at 12.100 TFLOP/s pure FP32. Against a probe built to
  match this kernel in every binding ratio, prefill sits at **95 %**.

**So the remaining distance is tokens per forward — the draft model's acceptance — and not a Metal
kernel.** One more draft depth costs a draft step plus a verify column, about 16 ms, and pays back one
token only if accepted: break-even is 53 % marginal acceptance against a measured 15 %. That is why
three separate mechanisms here each bought more tokens per pass and each lost throughput.

**On MLX.** The nearest comparable engine read 148.2 tok/s prefill and 21.4 tok/s unspeculated decode
on this machine — but that was measured on **2026-08-08, on Qwen3.6, on the mis-built binary
described below, and it has never been re-taken**. No current delta against MLX is quoted here.
The standing caveats still hold: this fork streams **1.1161× MLX's bytes** per decode pass (Q4_K_M
promotes `ffn_down`, `attn_qkv`, `attn_v` and `output` to Q6_K and `ssm_out` to Q5_K, against MLX's
uniform 4.5 bpw — a quality difference, not a speed one), MLX's prefill figure is inflated by
applying the 5120×248320 LM head at all 512 positions where `llama-bench` applies it once, and the
two engines cannot be co-resident on 36 GB, so no drift-controlled head-to-head has ever run.

**The upstream sync was verified rather than assumed.** The merge was clean — upstream barely touched
the Metal mat-vec files this fork changes. After merging: the Metal source compiles, a full build
succeeds, `test-backend-ops` passes **219 q4_K and 135 q6_K MUL_MAT cases with 0 failures**, the gated
kernels are still selected at verify widths 3 and 4, and MTP acceptance is unchanged to three
decimals. The figures either side of the sync were 31.30 / 18.64 before and 30.26 / 18.51 after, with
acceptance and tokens per forward **identical on all ten frozen prompts** — the same verify passes at
the same widths, so only wall-clock differed. That pair of runs was read as an A/A null putting
cross-session variation at 15 %. **Treat that 15 % as an upper bound on the machine, not a property
of it** — it was taken on the mis-built binary described next, and correctly built, two arms of the
same comparison agree to 0.04 %.

## The build defect underneath every older number in this file

**This fork's own build was compiled without `-O` for months.** The whole flag string had been glued
into `CMAKE_BUILD_TYPE`:

```
CMAKE_BUILD_TYPE:STRING=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DLLAMA_CURL=OFF
```

CMake compares that against `Release`/`Debug`/`RelWithDebInfo`/`MinSizeRel`, matches none, and
applies **no** `CMAKE_CXX_FLAGS_<CONFIG>` at all. No error, no warning. The compile line then carries
no `-O` and no `-DNDEBUG`. `libllama.dylib` came out at **9,001,552 bytes** where the correct build
is **2,872,128**. The only signal was `common_init()`'s own `warning: debug build, performance may be
affected`, printed into every server log for months and never correlated with the cache.

**The speed effect on the frozen GPU benchmark is unresolved and is not quoted.** The `-O0` arm's own
two readings differ by more than the effect. **The variance effect is not ambiguous:** two
correctly-built ABBA arms agree to **0.04 %** where two `-O0` arms differ by **3.90 %**. A large part
of what this fork called a 15 % cross-session noise floor was its own build. On the host-heavy
path the defect cost far more than the benchmark showed — server prompt processing went **104.7 →
141.7 tok/s, +35.3 %**, because that path is host C++ while `llama-bench` is nearly pure GPU.

**What this means for sections 0, 1, 3, 5 and 6 below.** They were measured in wave 1, on Qwen3.6,
on the mis-built binary. Both arms of every comparison carried the same defect, so the ratios are
internally valid and stand. The absolutes belong to their own sessions and do not compose with
today's. If anything those ratios understate the kernels: an `-O0` host dilutes a GPU-side win, and
the one kernel re-measured on a correct build gained — `L1-l16p` read +3.8 % on the old host and
**+5.45 %** rebuilt. That last sentence is an inference from one case, not a measurement.

The CMake half of this is written up for upstream; see "Upstream-worthy bugs" below.

## The changes

### 0. A column dimension for the K-quant mat-vec — everything else builds on it

*Wave 1, Qwen3.6.* Upstream's dedicated K-quant mat-vec handles one `src1` column per threadgroup.
Speculative verification presents two to four. This adds compile-time column variants (`_r1_2`,
`_r1_3`, `_r1_4`) for q4_K/q5_K/q6_K, so the expensive 6-bit scale decode is paid **once** and
amortised across the columns. Register pressure sets the shape: `nr0*nr1` accumulators plus `nr1`
float4s of live activations, so the multi-column variants carry their own `N_R0_*_R1` constants, all
2.

**1.431× on speculative decode**, and 1.43× again with the arm order reversed. Acceptance and
tokens-per-forward both moved *down* (0.7549 → 0.7507, 3.253 → 3.242), so the gain provably is not
coming from drafting. Sections 1, 2 and 6 tune this kernel further.

One more thing rides on the branch: **the gated-delta-net state write is fused into the recurrent
state cache**, removing one `cpy` per layer. Upstream's own version of this fusion (PR #25788),
ported verbatim, fires **0 times out of 192** here: `ggml_metal_graph_optimize_reorder()` hoists a
node in between the `GATED_DELTA_NET` and its `CPY`, so upstream's "the CPY is the next node" test
never holds. Matching the snapshot view by identity and pinning the pair gets 192/192.

### 1. Sixteen lanes per super-block for the Q4_K multi-column mat-vec — decode +6.7 %

*Wave 1, Qwen3.6.* Q4_K is 65 % of what decode reads, so its multi-column kernel is the one that
matters. It put **eight** lanes on a super-block, each reading one `float4` at a stride of eight
floats — so four lanes cover 64 bytes of a 128-byte line and the rest is discarded. Sixteen
contiguous lanes cover the line exactly.

**Neither half of the fix works alone** (kernel time, lower is faster). Four rows per simdgroup
instead of two is 0.98×; sixteen lanes at the old two rows is 1.083×. Together, **0.86× at verify
width 3**, on all four real Q4_K shapes within half a percent of each other. Halving the requested
*bytes* had never helped because it did not reduce the line *transactions*. Q6_K already used sixteen
lanes, which is the real reason it had taken the wider tile and Q4_K had not.

End to end **+6.7 %**, reproduced twice to within 0.02 points against two A/A nulls bracketing 1.000,
and collapsing to 1.001 at `n_max = 1` where the kernel cannot fire. `GGML_METAL_NO_Q4_L16=1` restores
the old behaviour.

### 2. `L1-l16p` — packed q4_K scales, dispatched at verify width 4 only — decode +5.45 %

*2026-08-16, Qwen3.8, correctly-built binary.* Section 1's kernel hoists the q4_K scales into
registers, and at width 4 it holds **32 registers** of them. `l16p` packs the same values into
**12**. It is dispatched **at width 4 only**: the same packing is 0.919 at width 4 and 1.069 — a loss
— at width 3, so `l16` still serves width 3 and `l16p` serves width 4. One kernel used to serve both
widths, and that threw the win away.

**+5.45 %** end to end. ABBA inside one binary, env-switched, order A B B A, every arm
assertion-verified to have run the kernel it claims *and not the other two*:

| arm | kernel | tok/s | tokens per forward |
|---|---|---|---|
| a1 | control `l16` | 30.724 | 3.0373 |
| b1 | **`l16p`** | **32.513** | 3.0373 |
| b2 | **`l16p`** | **31.454** | 3.0373 |
| a2 | control `l16` | 29.936 | 3.0373 |

Control 30.330 → 31.983. Largest order gap 3.31 %, so the effect exceeds it. 10/10 prompts positive,
median +5.36 %.

**The mechanism gate is the strong part.** Tokens per forward is identical on 10/10 prompts across
both arms — that quantity is deterministic under greedy decoding, so the arms provably did the same
work and the gain cannot be drafting. And the kernel is **bit-identical to `l16` over 16,384
outputs**, against a non-vacuous control: the 8-lane kernel at the same shape differs in the last
mantissa bits, so the comparison can fail.

**It is worth this much because of where the passes land.** 69.9 % of verify passes run at width 4 —
see "The width mix" below. Nobody had counted that before this kernel was built.

Kill switch `GGML_METAL_NO_L16P`. **`GGML_METAL_NO_L16P=` — set but empty — reads as ON**, so a
treatment arm must `env -u` it. **No register count is obtainable on this machine at all**, so any
register budget quoted here is a model; what is falsifiable is the source-level live set, 32 → 12.

### 3. A 32×32 accumulator tile for the quantised `mul_mm` — prefill +3.9 %

*Wave 1, Qwen3.6.* Each simdgroup held a **32×16** accumulator, so the threadgroup tile was 64×32.
Every well-tuned Apple GEMM converges on 32×32 per simdgroup — metal-flash-attention's Apple9 config
is `32x32x8`, MLX's Steel GEMM is 64×64×16 at `wm=wn=2`. Raising it takes fragment loads per
multiply-accumulate from 6:8 to 8:16 at the same 128 threads and the same four simdgroups.

**195.8 → 203.4 tok/s.** Bit-exact, and tested rather than argued: greedy output byte-identical
including a 7.5 k-token prefill, and perplexity identical to four decimals over 24 k tokens.

*That 203.4 is a coincidence and not the headline at the top of this file.* This pair is Qwen3.6 on
the `-O0` binary in wave 1; today's 203.43 is Qwen3.8 on `-O3` with two more prefill changes since.
Prefill has sat near this value throughout, which makes the two easy to confuse. Quote the **+3.9 %**,
which was taken within one session, not either absolute.

**Known limitation, not fixed here:** below `ne11 = 64` the wider tile is **1.6× slower** (1.64 at
n=9, 1.62 at n=32, 1.19 at n=63, 0.965 at n=512). Unreachable on this fork's operating point —
prefill runs `ne11 = 512` and speculative verify uses the mat-vec — but routine with `--parallel`
batching and on a prompt's last partial ubatch. **It needs an `ne11` gate before it is proposed
upstream.** The differentiator is `ne11`, not `src0` type: at n=9, f16 and q4_K lose identically.

### 4. `MM-pipeline` — hoist the `mul_mm` B-tile device loads above the WAR barrier — prefill +1.36 %

*2026-08-16, Qwen3.8, correctly-built binary.* The shipped kernel loaded the B tile from device
**and** stored it to threadgroup *between* the two barriers, where no `simdgroup_multiply_accumulate`
is in flight — so that load latency and its float→half converts were covered by nothing. Hoisting the
device loads above the WAR barrier makes B symmetric with A, and moves one device-load latency per
k-slice into the barrier-free region that already holds the previous slice's 64 MMAs.

**Measured within one binary**, which is the part that matters on an axis that drifts 2 % with
temperature:

| arm | kernel | pp512 |
|---|---|---|
| a1 | OFF (control) | 203.640 ± 0.014 |
| b1 | **ON** | **206.914 ± 0.083** |
| b2 | **ON** | **205.942 ± 1.398** |
| a2 | OFF (control) | 203.688 ± 0.102 |

**203.664 → 206.428 = +1.36 %, largest order gap 0.47 %.** The two control arms agree to 0.02 %.
An earlier cross-build pair read +1.92 % and is **superseded** — the difference between the two
figures is roughly what comparing two separately-built binaries was worth.

Decode is untouched; `mul_mm` is not on the decode path. Output is **byte-identical to the previous
kernel on 10/10 frozen prompts**, all non-empty on both sides, and `test-backend-ops` runs 144 OK /
0 FAIL. The change adds no arithmetic — same values, same MMA order — so greedy-output identity plus
`test-backend-ops` is the correct gate and is stricter than perplexity.

**Gated behind a Metal function constant**, `FC_mul_mm_pipe [[function_constant(FC_MUL_MM + 6)]]`,
the same mechanism `FC_mul_mm_bc_inp` already uses, so each pipeline compiles exactly one form and
the unused one costs nothing. Switch: `GGML_METAL_NO_MM_PIPELINE`. **The switch must appear in the
pipeline NAME** (`_pipe=%d`) — pipelines are cached by name, so without it the first arm compiled is
handed to the second and the A/B silently compares one kernel with itself. Proof of firing comes from
`llama-server -lv 5`, which prints `..._r3=1_pipe=1` against `..._r3=1_pipe=0`; `llama-bench` cannot
prove it, because it silences `GGML_LOG_*` after the device is created.

It also carries a critic's fix: `int64_t` where the original had `short`, which wrapped at
`k >= 32768`.

### 5. `dequantize_q4_K` divided the super-block scale in half precision — **upstreamable as-is**

*Wave 1, Qwen3.6, and it applies to every Q4_K model.* `ggml-metal.metal` computed the second-half
Q4_K scale as `xb->d / 16.h` — a **half** division. In a K-quant `d` is a scale of scales, ~100×
smaller than a first-order quant's: the median over all 78,960,640 Q4_K super-blocks of this model is
**6.354e-05**. Divide that by 16 in half precision and the quotient is 3.97e-06, below half's
smallest normal 6.104e-05 — **subnormal for 99.9998 % of this model's blocks**, losing ~3 mantissa
bits. `dequantize_q5_K` twenty lines below already wrote `16.f`.

This is on the prefill path (`mul_mm` at batch > 8, plus `get_rows`), so it affects **every Q4_K model
on Metal**, not just this one. Perplexity 6.6534 → 6.6475 on the frozen wikitext split — lower in 23
of 32 chunks and in the last 17 consecutively. One character, and it makes the model slightly more
accurate rather than faster.

### 6. A wider row tile for the Q6_K multi-column mat-vec at verify widths ≥ 3 — decode +4.1 %

*Wave 1, Qwen3.6.* `nr0 = 4` instead of 2 when `nr1 >= 3 && ne01 >= 4096`. Bit-exact. Gated on width
because at width 2 it is 1.01–1.06×, and on rows because a 1024-row matrix would drop to 128
threadgroups across 32 cores. `GGML_METAL_NO_Q6_NR0=1` restores the old behaviour.

Mechanism: a structural dump of the decode graph counts **0** gated nodes at verify width 2, **57** at
width 3 and **56** at width 4, exactly as the gate specifies, and the change covers **27.4 %** of the
decode weight stream — which sizes the measured +4.1 % without a residual.

### 7. `J-frspec` — an FR-Spec draft vocabulary — decode +2.93 %

*2026-08-16, Qwen3.8.* The draft head is trimmed to the **65,536 most frequent vocabulary rows plus
every non-normal token**, 65,806 of 248,320 — **994.6 → 263.6 MiB** (arXiv:2502.14856). The *target*
head stays at full vocabulary, so verification is unchanged.

| | unarmed | **armed** |
|---|---|---|
| decode, speculation on | 31.202 | **32.117 (+2.93 %)** |
| acceptance | 0.8656 | 0.8658 |
| draft steps | 9534 | 9524 |
| losslessness vs speculation off | 7/10 | **7/10** |

**It had never been armed in this fork's history.** It needs two environment variables and nothing
set either, so every MTP number this fork ever published used the full 248,320-row head:

```
LLAMA_MTP_VOCAB_N=65536  LLAMA_MTP_VOCAB_FILE=/path/to/frspec_rank.txt
```

`LLAMA_MTP_VOCAB_N` is how many ranked rows to keep; the file is one token id per line, most frequent
first, with `#` comments ignored. The subset is materialised as a contiguous copy in the same
quantisation as the full head, not gathered per step — a gather would touch and dequantise the same
rows anyway, which is the cost being removed.

**`-lv 4` is mandatory to see the proof line at all** — the trim logs at `LLAMA_LOG_INFO`, which
`common_get_verbosity()` maps to level 4 against a default threshold of 3. Without it, an unarmed run
is indistinguishable from an armed one in the log.

It is lossless by construction: the draft only proposes, and the target scores every proposal with
its own untouched full-vocabulary head. It does not change the losslessness *rate* either — armed and
unarmed are both 7/10 against speculation off, on two of the same three prompts, with one near-tie
moving between two prompts. The ranking comes from a 54 M-token mixed corpus, is **not** fitted to
the benchmark, and a fitted one would differ by at most 43 rows of 65,536.

## The width mix, measured

Counted per dispatch for the first time on 2026-08-16 — 18,894 graphs, 0 dropped dispatches, over
`blk.0.ffn_down.weight` as the pass counter, at the shipped operating point on Qwen3.8:

| verify width `ne11` | passes | share |
|---|---|---|
| 1 | 489 | 12.1 % |
| 2 | 397 | 9.8 % |
| 3 | 335 | 8.3 % |
| **4** | **2833** | **69.9 %** |

**Seven verify passes in ten land at width 4.** That single fact is why §2 was built and why it paid
+5.45 %. It also **retires an argument this fork used to close several experiments** — "it wins at a
width the operating point never reaches" was wrong for width 4, and everything closed on that
sentence is worth re-pricing.

**Do not read pass share as time share.** Width 1 is 12.1 % of passes but **27.9 % of mat-vec bytes**,
because draft steps read the 1.04 GB head at width 1.

Before re-opening a width-gated idea, check the byte share too. `l16p` worked because q4_K is 65.2 %
of the decode read. `A12-q5l16` looked similar and was not: its tensor `ssm_out` is 6.5 % of the
verify read, the kernel saves 23.7 % of that, and it measured +0.19 % at 0.42 σ. **Width was never its
problem; its ceiling was.**

## What binds the width-4 mat-vec — a register-allocation cliff

This is the most useful thing a future contributor can know about these kernels, and all three rows
were measured on 2026-08-16 at width 4, op-level, on the correctly-built binary:

| hypothesis | test | verdict |
|---|---|---|
| register **pressure** | `l16q`, four registers wider than `l16p` | **indistinguishable** — adding registers costs nothing |
| **instruction** count | `l16x`, `extract_bits` moved into the hot loop, −0.79 % trip-weighted | **1.013 — free.** Cutting instructions buys nothing |
| register **allocation** | `l16y`, **two fewer integer instructions in the PROLOGUE**, no float touched, no load width changed, no array resized | **1.353 — 35 % SLOWER** |

**Width 4 is neither instruction-bound nor register-pressure-bound. It sits on a register-allocation
cliff that is reached by editing the prologue**, where `sq`/`mq` are built. `l16w`, which also changes
a load width there, reads **2.375** — the spill signature, and the same shape as `nr0 = 8`'s 2.4×
collapse.

**`l16p` is on the good side of that cliff and nothing in its source says so.** Part of its +5.45 % is
an accident of how its prologue happens to compile, and any future edit near it can fall off. Several
earlier attacks on this axis probably did exactly that without knowing.

**Static instruction counts rank these changes wrongly, and so does a trip-weighted count.** Blocks
run 1, 4, 16 and 64 times per super-block, so a prologue instruction and a dot-loop instruction score
the same statically and differ 16× in reality — yet the *free* change (`l16x`) is the one inside the
64×-weighted loop, and the *catastrophic* one (`l16y`) is in the 1×-weighted prologue. A trip-weighted
count was built and predicted nothing either. **Price width-4 ideas by measuring the op, not by
counting AIR.**

All six variants are bit-identical over 16,384 outputs against a non-vacuous control, and all are
opt-in behind `GGML_METAL_L16P_VARIANT`. **None is proposed. `l16p` remains the best kernel at width
4.**

The earlier model in this fork's notes — "width 3 is instruction-bound, width 4 is register-bound" —
drove every attack on this axis, and the second half of it has expired. Note that `l16p` itself
*costs* 5.4 % more instructions (565 AIR against 536) and won anyway.

## Draft depth 3, and why the depth decision had to be re-taken

*Wave 1, Qwen3.6.* One change to this fork is **not a kernel**. It moves the operating point from
`--spec-draft-n-max 2` to `3`, worth **+6.01 %** — 20 order-balanced pairs on one warm server, 9.9 σ,
every pair and every category positive, order gap 0.12 pp.

Depth 3 had been measured earlier on this fork as a **loss**, and written down as closed. That
measurement was correct when it was taken. It was taken **before** the Q6_K wide-verify tile (§6) and
the Q4_K sixteen-lane mapping (§1), both of which are gated on verify width and both of which are
worth far more at width 4 than at width 3 — §6 alone is **0.58× at width 4** against 0.74 at width 3.
Depth 3 is the configuration that produces width-4 verify passes.

> A configuration decision is a measurement against a particular kernel stack. It expires when the
> stack changes. This fork re-derived kernel numbers constantly and had never re-derived a policy
> number.

**Method note.** The A/B was made possible by giving each draft step its own threshold and setting
step 2's above 1.0, which disables it — so a depth-3 server expresses depth 2 exactly and both depths
are comparable inside one model load, seconds apart, instead of two servers ~12 minutes apart. That
instrument is not part of this branch; only the operating point changed.

**Do not go to depth 4.** It is the worst setting available on this backend, for a reason that has
nothing to do with acceptance — see the `nr1_k(5)` item below.

## Test coverage worth upstreaming on its own

`test-backend-ops` runs green on things it cannot see, and it keeps biting this fork.

**It has no quantised K-quant mat-vec case above `ne01 = 16`, one `mul_mm` perf shape, and almost no
partial-tile coverage.** Any backend that switches kernels on row count or tile geometry is untested
there. Section 6 passed 1143/1143 with its gate **on and off** — both arms ran identical code and the
test could not have failed. Section 3 encodes its tile in six independent places and **four were
wrong** in the first version; the last, a `short` holding a 64-bit row stride that wraps past
`k = 32768` and reads ~4 MB before the buffer, needed cases nobody had written.

**Unfiltered, it aborts before it reaches any of it.** `test-backend-ops test -b MTL0 -o MUL_MAT`
aborts at **case 54 on `GGML_TYPE_TQ2_0`**, which is *before* the first `m = 4096` case — and
`m = 4096` is the only shape the `l16`/`l16p` width gate accepts. So the run prints 54 OK, exits
non-zero on an unrelated abort, and **never once dispatches the kernel under test**. That abort is
pre-existing upstream, not ours, but it sits in front of everything this fork needs to test, and a
healthy-looking case count is exactly what makes it dangerous.

```
test-backend-ops test -b MTL0 -o MUL_MAT -p 'type_a=q[456]_K,type_b=f32,m=4096'
   -> 36 cases OK, 0 FAIL, rc = 0
```

The filter is required, not optional. Mode is `test` and backend is `MTL0`; `Metal` matches nothing
and exits green.

**Watch the shapes as well as the count.** Checking §4's `mul_mm` gate with `test-backend-ops`
reported 26 OK / 0 FAIL on both arms while compiling **only `kernel_mul_mv` pipelines** — the suite's
K-quant MUL_MAT cases run `n = 1..9`, and `mul_mm` never ran.

**It also never calls `graph_optimize`.** A graph-level fusion can therefore pass every unit test and
still fire zero times in production, which is exactly what upstream's gated-delta-net fusion does here
(§0).

This fork adds `ne01` ∈ {4088, 4095, 4096, **4100**, 4104} × `n` ∈ {2,3,4} for q4_K/q5_K/q6_K,
`mul_mm` partial tiles in both dimensions across both staging paths, and the large-stride shapes.
**That block is independent of every Metal change here and is the piece most worth taking.**

Check a kernel-selection change by pipeline count, not by the pass:

```
                          tests          gated pipelines compiled
default                   14581/14581    4
GGML_METAL_NO_Q4_L16=1    14581/14581    0
```

`strings` on the dylib does **not** work as that check: with `GGML_METAL_EMBED_LIBRARY` the
`host_name`s come from the Metal preprocessor at `ggml_metal_init`, so a name built from a macro
argument exists nowhere in the binary. The Metal *source* is embedded as text, so source-level changes
can be diffed that way — kernel names cannot.

## Upstream-worthy bugs, independent of anything here

**1. `ggml_metal_mul_mv_nr1_k(5) = 3` makes a five-column mat-vec stream the weights twice.** In
`ggml-metal-device.cpp`:

```c
case 5:  return 3; // 2 threadgroups, 1 slot wasted
```

The comment accounts for the wasted column *slot* and not for the wasted *bandwidth*. The dispatch
grid is `ceil(ne11 / nr1)` threadgroups in the column direction, and **each threadgroup reads its own
copy of the weight rows it covers**. So `ne11 = 5` at `nr1 = 3` issues two passes over the entire
16.08 GB weight set to produce five columns, where `ne11 = 4` at `nr1 = 4` issues one to produce four.
Measured on Qwen3.8-27B Q4_K_M at the frozen operating point:

| draft depth | verify width | verify cycle |
|---|---|---|
| 2 | 3 | 87.6 ms |
| 3 | 4 | 109.4 ms |
| **4** | **5** | **151.9 ms** |

The 3 → 4 step costs +21.8 ms; the 4 → 5 step costs **+42.5 ms**, roughly double, on one extra column.
**Draft depth 4 is therefore the worst setting available** — worse than 2, 3 and even 5, since
`nr1_k(6) = 3` is two exact threadgroups. A user tuning depth upward hits a hole at exactly one value
and has no way to see why. Fix either by instantiating `nr1 = 5` (cheap — the kernels are already
templated over `nr1 ∈ {2,3,4}`) or by making the comment tell the truth and documenting which widths
pay an extra pass. It is a performance cliff hidden behind a comment that counts the wrong resource,
not a correctness bug.

**2. `--spec-draft-n-max` is unguarded for single-head MTP models.** In `common/speculative.cpp` the
clamp

```cpp
this->params.n_max = std::min(this->params.n_max, n_mtp_layers);
```

sits **inside** the `if (chain_heads)` branch, and `chain_heads` is
`n_mtp_layers > 1 && !is_mem_shared`. A model with exactly one prediction head — this one has only
`blk.64.nextn.*` — therefore never reaches the clamp and will accept any depth the user asks for.
Depth 3 happens to work here. Nothing stops someone asking for 8.

**3. CMake accepts a malformed `CMAKE_BUILD_TYPE` silently.** Described above. It is a user error, but
the failure is silent, the resulting binary runs correctly and only a few percent slower on a
GPU-bound benchmark, and **the misconfigured build is the one people benchmark with**. Five lines in
the top-level `CMakeLists.txt` would catch it:

```cmake
set(LLAMA_KNOWN_BUILD_TYPES Debug Release RelWithDebInfo MinSizeRel)
if (NOT CMAKE_CONFIGURATION_TYPES AND CMAKE_BUILD_TYPE AND
    NOT CMAKE_BUILD_TYPE IN_LIST LLAMA_KNOWN_BUILD_TYPES)
    message(WARNING "CMAKE_BUILD_TYPE='${CMAKE_BUILD_TYPE}' is not a known build type; "
                    "no optimisation flags will be applied.")
endif()
```

**4. The `dequantize_q4_K` half-precision scale** (§5) — one character, affects every Q4_K model on
Metal.

## Closed with numbers — do not re-propose

Each of these was built or derived and refuted. They are here so nobody spends the campaign again.

**`MM-tile64` — a 64×32 accumulator tile for `mul_mm`. 5.6× SLOWER.** Built, gated,
`test-backend-ops` clean at 144 cases, firing asserted in both directions (3 selections with the gate
on, 0 with it off).

| arm | tile | pp512 |
|---|---|---|
| a1, a2 | 32×32 (control) | **201.07** |
| b1, b2 | **64×32** | **36.00 / 36.01** |

Reproduced to 0.03 %. The register arithmetic explains it: counting fragments and dequant staging as
well as the accumulator, it is **111 registers per lane against `MM-tile32`'s 67**, i.e. 15.0
simdgroups per core against 24.8, and at that point it spills. The trade was +33 % MAC per fragment
load against −40 % occupancy. **The accumulator alone now fills the budget that all of `MM-tile32`
fits inside.**

**Fusing `ffn_gate` and `ffn_up` — rejected on registers, with zero GPU time, and it closes a
family.** The premise checks out: both are 5120×17408 Q4_K at all 65 blocks and together they are
46.9 % of the prefill GEMM budget. The kernel is not: the fused kernel is `MM-tile64` wearing a
different name — 113 registers per lane against 111 — equal in every term but where the second A
stream is read from. **The bound is worth more than the rejection.** Per simdgroup per 8-wide k-step,
`a` A-fragments and `b` B-fragments give `a·b` MACs from `a+b` loads, so by AM-GM

> **MAC per load = `ab/(a+b)` ≤ `√N / 2`**, N = accumulator fragments, equality iff `a = b`.

`MM-tile32` is `a = b = 4`, N = 16, ratio **2.00 = √16/2 — already exactly on the bound.** Fusion does
not appear in the bound; it only relabels where some A-fragments were loaded from. **So anything
beating 2.00 must raise N — and N = 32 is `MM-tile64`, which spills. Do not re-propose any tiling that
raises the accumulator fragment count above 16**, by widening one tensor's tile or by fusing two.
§4 survives this bound because prefetching changes *when* loads issue, not `a`, `b` or N.

**An entropy-gated draft policy (`Z-adaedl`) — −0.40 %.** Swept at tau ∈ {0.15, 0.25, 0.35} and
confirmed ABBA at the best tau on the full prompt set: 31.013 → 30.889 tok/s, with tokens per forward
*up* 4.0 % (3.0373 → 3.1581) and acceptance down (0.8656 → 0.8107). **The gate really does yield 4 %
more tokens per verify pass, and it costs exactly what it earns.** Price extra columns marginally,
never on pooled tokens per forward: the 2nd column costs **+8.7 ms** and the 3rd **+23.1 ms** against
a token worth 33 ms, so column 3 needs marginal acceptance ≈ **0.70** to break even. The promising
direction is drafting *less* and better, not more.

**Also closed, with numbers in this fork's records:** the `simdgroup_matrix`/BM=16 family for decode
widths (f16 caps it at 214 GB/s against the mat-vec's 337), split-K in either direction, `nr0 = 8`
(2.4× collapse from spill), offline weight repacking, `simd_sum` reduction cost (2.3 % of lane slots),
sharing the K-quant scales by `simd_shuffle` (473 instructions against 456 — one
`air.simd_shuffle.v4f32` is four hardware lane operations), threadgroup staging of activations,
lowering `ne11_mm_min`, the tensor/`matmul2d` API (no matrix unit, ~0 %), and trimming the **verify**
head — greedy verification needs the true argmax over all 248,320 rows, so it changes which token
wins. That last one is why §7 trims the draft head only.

## Known limits

**1. A pre-existing out-of-bounds read gets wider.** These mat-vec kernels guard the *store*
(`first_row + row < args.ne0`) but not the *load*, so the last threadgroup reads up to `nr0*nsg - 1`
rows past `ne01` — 3 rows before these changes, 7 after. Latent on this model because every affected
tensor has `ne01` divisible by 8; `test-backend-ops` cannot see it because the results are still
correct. Not fixed here.

**2. Speculative decoding is not bit-exact against non-speculative decoding, at any depth.** The
verify pass selects its mat-vec kernel by batch width — `nr1_k(3) = 3`, `nr1_k(4) = 4` — and those
instantiations partition the `simd_sum` reduction differently from the width-1 path, so the last bits
differ. Wherever two tokens are near-tied the argmax flips; the first divergence traced sat
mid-JSON-schema with both continuations valid. The rate is **7/10 on the shipped Qwen3.8 stack**,
armed or unarmed, and was 9/10 on Qwen3.6 at both depth 2 and depth 3, a different prompt each time.
Lower acceptance means more rejection points, and every rejection point is a place where a near-tie
can flip. Byte-identity is judged against the **same binary with speculation off**. Do not quote
10/10.

**3. The thresholds are tuned to this chip.** `ne01 >= 4096` is really a threadgroups-per-core
quantity — 16 per core on a 32-core part — written as a row count because that is what the dispatch
has to hand. Another part should re-derive it rather than inherit it.

**4. What limits the decode mat-vec differs by width, so a win at one width is not a win.** §2 is
0.919 at width 4 and 1.069 at width 3, from one kernel and one packing. This is why the selection is
width-aware and why any new variant must be measured at each width it can reach.

**5. The chat template is part of the workload.** Not a kernel limit, but it invalidated a whole
model-to-model comparison here. Qwen3.8's template ships inside the GGUF and injects a 42-token
reasoning-effort preamble that Qwen3.6 never emitted, so every frozen prompt tokenised 42 tokens
longer while the prompt-file hash still matched. Diff `prompt_n` per prompt before comparing any two
runs. `--chat-template-kwargs '{"reasoning_effort":"medium"}'` reproduces Qwen3.6's token counts
exactly.

## Build

```
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
cmake --build build -j
```

**Those are separate arguments and it matters.** Quote them into one and CMake takes the whole string
as the build type, matches no known config, and applies no optimisation — silently. Assert it after
configuring:

```
grep CXX_FLAGS build/ggml/src/CMakeFiles/ggml-base.dir/flags.make | grep -- -O
```

To arm the FR-Spec draft vocabulary (§7), set both variables and run the server at `-lv 4` to see the
proof line:

```
LLAMA_MTP_VOCAB_N=65536 LLAMA_MTP_VOCAB_FILE=/path/to/frspec_rank.txt
```

## Licence

MIT, as upstream. See [LICENSE](LICENSE).
