# llama.cpp — Metal tuning for Qwen3.6-27B on an M4 Max

A fork carrying Metal-backend changes aimed at one model on one machine: **Qwen3.6-27B (Q4_K_M) on
a 32-core M4 Max, 410 GB/s, 36 GB**. Branch `shipped` is the tuned stack; `master` is the upstream
commit it forks from (`3653e6d6d`), kept so the diff stays readable.

**Nothing here is a general llama.cpp speedup claim.** It is one model on one chip, and some of it
is tuned to both. Two pieces are worth upstreaming anyway and are called out below.

| | this fork | MLX 4-bit | vs MLX |
|---|---|---|---|
| prefill (pp512) | **203.4 tok/s** | 148.2 | **+37 %** |
| decode, speculation on | **29.0 tok/s** | 21.4 | **+36 %** |
| decode, speculation off | 18.4 | 21.4 | −14 % |

Decode is speculative (MTP, depth 2, `p_min` 0.6): acceptance 0.943, 2.49 tokens per forward pass,
output byte-identical to unspeculated greedy on 9 of 10 frozen prompts (the tenth is a long-context
near-tie that flips either way).

**Both figures were read off a run, not derived** — prefill over 9 order-balanced arms with a
control that reproduced to 0.05 %, decode rested alongside its own unspeculated arm in the same
session. Cross-session absolutes on this machine carry ~8 % of drift, more than most individual
results, so derived numbers are not quoted.

The −14 % unspeculated is mostly not a kernel deficit: **11.6 points of MLX streaming fewer bytes**
(uniform 4.5 bpw against Q4_K_M's Q6_K/Q5_K promotions, a quality difference) × **6.3 points of
rate**. Both engines sit inside the 73–85 % of peak normal for an M4-generation streaming read.

## The changes

### 1. Sixteen lanes per super-block for the Q4_K multi-column mat-vec — decode +6.7 %

Speculative decoding verifies several tokens per pass, so the kernel that matters is the
multi-column mat-vec, not the one-column one. Q4_K is 65 % of what decode reads.

That kernel put **eight** lanes on a super-block, each reading one `float4` at a stride of eight
floats — so four lanes cover 64 bytes of a 128-byte line and the rest is discarded. Sixteen
contiguous lanes cover the line exactly.

**Neither half of the fix works alone.** Four rows per simdgroup instead of two is 0.98×; sixteen
lanes at the old two rows is 1.083×. Together, **0.86× at verify width 3**, on all four real Q4_K
shapes within half a percent of each other. Halving the requested *bytes* had never helped because
it did not reduce the line *transactions*. Q6_K already used sixteen lanes, which is the real reason
it had taken the wider tile and Q4_K had not.

End to end **+6.7 %**, reproduced twice by a reviewer to within 0.02 points against two A/A nulls
bracketing 1.000, and collapsing to 1.001 at `n_max = 1` where the kernel cannot fire.
`GGML_METAL_NO_Q4_L16=1` restores the old behaviour.

### 2. A 32×32 accumulator tile for the quantised `mul_mm` — prefill +3.9 %

Each simdgroup held a **32×16** accumulator, so the threadgroup tile was 64×32. Every well-tuned
Apple GEMM converges on 32×32 per simdgroup — metal-flash-attention's Apple9 config is `32x32x8`,
MLX's Steel GEMM is 64×64×16 at `wm=wn=2`. Raising it takes fragment loads per multiply-accumulate
from 6:8 to 8:16 at the same 128 threads and the same four simdgroups.

**195.8 → 203.4 tok/s.** Bit-exact, and tested rather than argued: greedy output byte-identical
including a 7.5 k-token prefill, and perplexity identical to four decimals over 24 k tokens.

**Known limitation, not fixed here:** below `ne11 = 64` the wider tile is **1.6× slower**
(1.64 at n=9, 1.62 at n=32, 1.19 at n=63, 0.965 at n=512). Unreachable on this project's operating
point — prefill runs `ne11 = 512` and speculative verify uses the mat-vec — but routine with
`--parallel` batching and on a prompt's last partial ubatch. **It needs an `ne11` gate before it is
proposed upstream.** The differentiator is `ne11`, not `src0` type: at n=9, f16 and q4_K lose
identically.

### 3. `dequantize_q4_K` divided the super-block scale in half precision — **upstreamable as-is**

`ggml-metal.metal` computed the second-half Q4_K scale as `xb->d / 16.h` — a **half** division. In a
K-quant `d` is a scale of scales, ~100× smaller than a first-order quant's: the median over all
78,960,640 Q4_K super-blocks of this model is **6.354e-05**. Divide that by 16 in half precision and
the quotient is 3.97e-06, below half's smallest normal 6.104e-05 — **subnormal for 99.9998 % of this
model's blocks**, losing ~3 mantissa bits. `dequantize_q5_K` twenty lines below already wrote `16.f`.

This is on the prefill path (`mul_mm` at batch > 8, plus `get_rows`), so it affects **every Q4_K
model on Metal**, not just this one. Perplexity 6.6534 → 6.6475 on the frozen wikitext split — lower
in 23 of 32 chunks and in the last 17 consecutively. One character, and it makes the model slightly
more accurate rather than faster.

### 4. A wider row tile for the Q6_K multi-column mat-vec at verify widths ≥ 3 — decode +4.1 %

`nr0 = 4` instead of 2 when `nr1 >= 3 && ne01 >= 4096`. Bit-exact. Gated on width because at width 2
it is 1.01–1.06×, and on rows because a 1024-row matrix would drop to 128 threadgroups across 32
cores. `GGML_METAL_NO_Q6_NR0=1` restores the old behaviour.

Its mechanism was open when this file was first written and is now settled: a structural dump of the
decode graph counts **0** gated nodes at verify width 2, **57** at width 3 and **56** at width 4,
exactly as the gate specifies, and the change covers **27.4 %** of the decode weight stream — which
sizes the measured +4.1 % without a residual.

## Test coverage worth upstreaming on its own

Upstream `test-backend-ops` had **one** `mul_mm` perf shape and almost no partial-tile coverage, and
no quantised K-quant mat-vec case above **`ne01 = 16`**. Any backend that switches kernels on row
count or tile geometry is untested there, and a green suite says nothing about it.

That bit this fork twice. Change 4 passed 1143/1143 with its gate **on and off** — both arms ran
identical code and the test could not have failed. Change 2 encodes its tile in six independent
places and **four were wrong** in the first version; the last, a `short` holding a 64-bit row stride
that wraps past `k = 32768` and reads ~4 MB before the buffer, needed cases nobody had written.

This fork adds `ne01` ∈ {4088, 4095, 4096, **4100**, 4104} × `n` ∈ {2,3,4} for q4_K/q5_K/q6_K,
`mul_mm` partial tiles in both dimensions across both staging paths, and the large-stride shapes.
**That block is independent of every Metal change here and is the piece most worth taking.**

Verification looks like this — the pipeline count, not the pass, is the evidence:

```
                          tests          gated pipelines compiled
default                   14581/14581    4
GGML_METAL_NO_Q4_L16=1    14581/14581    0
```

`strings` on the dylib does **not** work as a check for kernel selection: with
`GGML_METAL_EMBED_LIBRARY` the `host_name`s are produced by the Metal preprocessor at
`ggml_metal_init`, so a name built from a macro argument exists nowhere in the binary. (The Metal
*source* is embedded as text, so source-level changes can be diffed that way — kernel names cannot.)

## What is not established

**A pre-existing out-of-bounds read gets wider.** These mat-vec kernels guard the *store*
(`first_row + row < args.ne0`) but not the *load*, so the last threadgroup reads up to
`nr0*nsg - 1` rows past `ne01` — 3 rows before these changes, 7 after. Latent on this model because
every affected tensor has `ne01` divisible by 8; `test-backend-ops` cannot see it because the
results are still correct. Not fixed here.

**The thresholds are tuned to this chip.** `ne01 >= 4096` is really a threadgroups-per-core quantity
— 16 per core on a 32-core part — written as a row count because that is what the dispatch has to
hand. Another part should re-derive it rather than inherit it.

**What limits the decode mat-vec is now known to differ by width**: verify width 3 is
instruction-bound and width 4 is register-bound, and the same change can help one and hurt the
other. A packed-scale variant that saves 18 registers is 8 % *faster* at width 4 and 7 % *slower* at
width 3, so it is not shipped.

## Build

```
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
cmake --build build -j
```

## Licence

MIT, as upstream. See [LICENSE](LICENSE).
