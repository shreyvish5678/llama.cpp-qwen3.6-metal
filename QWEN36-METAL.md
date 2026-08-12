# llama.cpp — Metal tuning for Qwen3.6-27B on an M4 Max

A fork carrying Metal-backend changes aimed at one model on one machine: **Qwen3.6-27B (Q4_K_M) on
a 32-core M4 Max, 410 GB/s, 36 GB**. Branch `shipped` is the tuned stack; `master` tracks upstream
`ggml-org/llama.cpp` and is kept in sync so the diff stays readable.

**Synced with upstream through `89e0aa6fd`.** The merge was clean — upstream barely touched the
Metal mat-vec files this fork changes. Verified after merging: the Metal source compiles, a full
build succeeds, `test-backend-ops` passes **219 q4_K and 135 q6_K MUL_MAT cases with 0 failures**,
the gated kernels are still selected at verify widths 3 and 4, and MTP speculation runs with
acceptance unchanged to three decimals.

**The figures below were re-taken after that sync**, so they describe the merged tree. The sync
changed nothing measurable: pre-sync 31.30 / 18.64, post-sync 30.26 / 18.51, with acceptance and
tokens-per-forward **identical on all ten frozen prompts** — the same verify passes at the same
widths, so only wall-clock differed. That pair of runs doubles as an A/A null on provably identical
work and puts cross-session variation on this machine at **15 %**, which is why a 3 % gap is not
readable as a regression.

**Nothing here is a general llama.cpp speedup claim.** It is one model on one chip, and some of it
is tuned to both. Two pieces are worth upstreaming anyway and are called out below.

| | this fork | MLX 4-bit | vs MLX |
|---|---|---|---|
| prefill (pp512) | **203.4 tok/s** | 148.2 | **+37 %** |
| decode, speculation on | **30.3 tok/s** | 21.4 | **+41 %** |
| decode, speculation off | 18.5 | 21.4 | −13 % |

Decode is speculative (MTP, **depth 3**, `p_min` 0.6): acceptance 0.925, **2.94 tokens per forward
pass**. Output matches unspeculated greedy on 9 of 10 frozen prompts.

**On that 9/10 — it is a property of the design, not a regression, and it is the same at every
depth.** Measured in one session against the same unspeculated texts, depth 2 differs on one prompt
and depth 3 differs on a different one. The verify pass selects its mat-vec kernel by batch width,
and those instantiations partition the `simd_sum` reduction differently from the width-1 path, so
the last bits differ. Wherever two tokens are near-tied the argmax flips; the first divergence we
traced sits mid-JSON-schema with both continuations valid. **Speculative decoding on this backend is
not bit-exact against non-speculative decoding at any depth.**

**Both figures were read off a run, not derived** — prefill over 9 order-balanced arms with a
control that reproduced to 0.05 %, decode alongside its own unspeculated arm in the same session
with the texts diffed. Absolutes on this machine drift ~8 % between sessions, so nothing here composes numbers
taken on different days.

MLX is the nearest comparable engine, not the target; the hardware is. Prefill's `mul_mm` runs at
~77 % of this chip's 12.93 TFLOP/s scalar roof — **there is no matrix unit on M4**, `matmul2d`
lowers onto the ordinary shader path. Unspeculated decode at one token per forward pass cannot
exceed **25.5 tok/s**: 410 GB/s over a 16.1 GB read. All remaining decode headroom is speculative.

The −14 % unspeculated is mostly not a kernel deficit: **11.6 points of MLX streaming fewer bytes**
(uniform 4.5 bpw against Q4_K_M's Q6_K/Q5_K promotions, a quality difference) × **6.3 points of
rate**. Both engines sit inside the 73–85 % of peak normal for an M4-generation streaming read.

## The changes

### 0. A column dimension for the K-quant mat-vec — everything else builds on it

Upstream's dedicated K-quant mat-vec handles one `src1` column per threadgroup. Speculative
verification presents two to four. This adds compile-time column variants (`_r1_2`, `_r1_3`,
`_r1_4`) for q4_K/q5_K/q6_K, so the expensive 6-bit scale decode is paid **once** and amortised
across the columns. Register pressure sets the shape: `nr0*nr1` accumulators plus `nr1` float4s of
live activations, so the multi-column variants carry their own `N_R0_*_R1` constants, all 2.

**1.431× on speculative decode**, and 1.43× again with the arm order reversed. Acceptance and
tokens-per-forward both moved *down* (0.7549 → 0.7507, 3.253 → 3.242), so the gain provably is not
coming from drafting. Sections 1 and 4 tune this kernel further.

Two more things ride on the branch:

- **The gated-delta-net state write is fused into the recurrent state cache**, removing one `cpy`
  per layer. Upstream's own version of this fusion (PR #25788), ported verbatim, fires **0 times out
  of 192** here: `ggml_metal_graph_optimize_reorder()` hoists a node in between the
  `GATED_DELTA_NET` and its `CPY`, so upstream's "the CPY is the next node" test never holds.
  Matching the snapshot view by identity and pinning the pair gets 192/192.
- **An FR-Spec draft-vocabulary trim** (arXiv:2502.14856), inert unless `LLAMA_MTP_VOCAB_N` and
  `LLAMA_MTP_VOCAB_FILE` are set. It restricts the *draft* head to a frequency-ranked row subset;
  the target head stays at full vocabulary, so verification is unchanged. Measured +2.8 % on a
  single non-interleaved pair, ~1.6 σ. **Not confirmed, and not in the 29.0 above.**

### 1. Sixteen lanes per super-block for the Q4_K multi-column mat-vec — decode +6.7 %

Q4_K is 65 % of what decode reads, so its multi-column kernel is the one that matters. It put
**eight** lanes on a super-block, each reading one `float4` at a stride of eight floats — so four
lanes cover 64 bytes of a 128-byte line and the rest is discarded. Sixteen contiguous lanes cover
the line exactly.

**Neither half of the fix works alone** (kernel time, lower is faster). Four rows per simdgroup
instead of two is 0.98×; sixteen lanes at the old two rows is 1.083×. Together, **0.86× at verify
width 3**, on all four real Q4_K shapes within half a percent of each other. Halving the requested
*bytes* had never helped because it did not reduce the line *transactions*. Q6_K already used
sixteen lanes, which is the real reason it had taken the wider tile and Q4_K had not.

End to end **+6.7 %**, reproduced twice to within 0.02 points against two A/A nulls bracketing
1.000, and collapsing to 1.001 at `n_max = 1` where the kernel cannot fire.
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

Mechanism: a structural dump of the decode graph counts **0** gated nodes at verify width 2, **57**
at width 3 and **56** at width 4, exactly as the gate specifies, and the change covers **27.4 %** of
the decode weight stream — which sizes the measured +4.1 % without a residual.

## Test coverage worth upstreaming on its own

`test-backend-ops` runs green on things it cannot see, and that bit this fork twice.

**It has no quantised K-quant mat-vec case above `ne01 = 16`, one `mul_mm` perf shape, and almost no
partial-tile coverage.** Any backend that switches kernels on row count or tile geometry is untested
there. Change 4 passed 1143/1143 with its gate **on and off** — both arms ran identical code and the
test could not have failed. Change 2 encodes its tile in six independent places and **four were
wrong** in the first version; the last, a `short` holding a 64-bit row stride that wraps past
`k = 32768` and reads ~4 MB before the buffer, needed cases nobody had written.

**It also never calls `graph_optimize`.** A graph-level fusion can therefore pass every unit test and
still fire zero times in production, which is exactly what upstream's gated-delta-net fusion does
here (section 0).

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
argument exists nowhere in the binary. The Metal *source* is embedded as text, so source-level
changes can be diffed that way — kernel names cannot.

## Known limits, and one thing not to re-propose

**A pre-existing out-of-bounds read gets wider.** These mat-vec kernels guard the *store*
(`first_row + row < args.ne0`) but not the *load*, so the last threadgroup reads up to
`nr0*nsg - 1` rows past `ne01` — 3 rows before these changes, 7 after. Latent on this model because
every affected tensor has `ne01` divisible by 8; `test-backend-ops` cannot see it because the
results are still correct. Not fixed here.

**The thresholds are tuned to this chip.** `ne01 >= 4096` is really a threadgroups-per-core quantity
— 16 per core on a 32-core part — written as a row count because that is what the dispatch has to
hand. Another part should re-derive it rather than inherit it.

**What limits the decode mat-vec differs by width**, so a win at one width is not a win. Verify
width 3 is instruction-bound and width 4 is register-bound: a packed-scale variant that saves 18
registers is 8 % *faster* at width 4 and 7 % *slower* at width 3, so it is not shipped. This model's
operating point never reaches width 4, which makes instruction count the axis that matters.

## Build

```
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
cmake --build build -j
```

## Licence

MIT, as upstream. See [LICENSE](LICENSE).


## Draft depth 3, and why the depth decision had to be re-taken

The last change to this fork is **not a kernel**. It moves the operating point from
`--spec-draft-n-max 2` to `3`, worth **+6.01 %** — 20 order-balanced pairs on one warm server,
9.9 σ, every pair and every category positive, order gap 0.12 pp.

Depth 3 had been measured earlier on this fork as a **loss**, and written down as closed. That
measurement was correct when it was taken. It was taken **before** the Q6_K wide-verify tile
(`A5-q6nr0`) and the Q4_K sixteen-lane mapping (`A10-q4l16`), both of which are gated on verify
width and both of which are worth far more at width 4 than at width 3 — A5 alone is **0.58× at
width 4** against 0.74 at width 3. Depth 3 is the configuration that produces width-4 verify passes.

> A configuration decision is a measurement against a particular kernel stack. It expires when the
> stack changes. This fork re-derived kernel numbers constantly and had never re-derived a policy
> number.

**Method note.** The A/B was made possible by giving each draft step its own threshold and setting
step 2's above 1.0, which disables it — so a depth-3 server expresses depth 2 exactly and both
depths are comparable inside one model load, seconds apart, instead of two servers ~12 minutes
apart. That instrument is not part of this branch; only the operating point changed.

## Upstream-worthy bug, independent of anything here

**`--spec-draft-n-max` is unguarded for single-head MTP models.** In `common/speculative.cpp` the
clamp

```cpp
this->params.n_max = std::min(this->params.n_max, n_mtp_layers);
```

sits **inside** the `if (chain_heads)` branch, and `chain_heads` is
`n_mtp_layers > 1 && !is_mem_shared`. A model with exactly one prediction head — this one has only
`blk.64.nextn.*` — therefore never reaches the clamp and will accept any depth the user asks for.
Depth 3 happens to work here. Nothing stops someone asking for 8.
