# llama.cpp — Metal tuning for Qwen3.6-27B on an M4 Max

A fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) carrying Metal-backend changes aimed
at one model on one machine: **Qwen3.6-27B-A3B (Q4_K_M) on a 32-core M4 Max, 410 GB/s, 36 GB**.

Branch **`shipped`** is the tuned stack. `master` is the upstream commit it forks from
(`3653e6d6d`), kept so the diff is readable.

Everything below was measured on that machine with speculative decoding (MTP) at depth 2. Numbers
are from a frozen 10-prompt set across code generation, code regeneration, prose, long context and
JSON. **Nothing here is a general llama.cpp speedup claim** — it is one model, one chip, and some
of it is tuned to both.

| | this fork | MLX 4-bit | vs MLX |
|---|---|---|---|
| decode, speculation on | **29.3 tok/s** | 21.4 | **+37 %** |
| decode, speculation off | 18.2 | 21.4 | −15 % |
| prefill (pp512) | **193.4 tok/s** | 148.2 | **+30 %** |

The −15 % on unspeculated decode is mostly not a kernel deficit: it factors into **11.6 points of
MLX streaming fewer bytes** (uniform 4.5 bpw against Q4_K_M's Q6_K/Q5_K promotions — a quality
difference, not a speed one) times **6.3 points of rate**. Both engines sit inside the 73–85 % of
peak that is normal for an M4-generation Metal streaming read.

## The change on top of the earlier stack

**A wider row tile for the Q6_K multi-column mat-vec at speculative verify widths ≥ 3.**

Speculative decoding verifies several tokens per forward pass, so the kernel that matters is not
the one-column mat-vec but the multi-column one. On this model that kernel family is **~56 % of the
whole speculative decode cycle** — measured by forcing a knob with a known op-level cost (one row
per simdgroup, a 1.5× slowdown) and observing 22 % end-to-end, then solving for the share.

It is bound by **activation-read traffic, not the weight stream.** One threadgroup requests
`(ne01/nr0)·nr1·ne00·4` bytes of `src1`. Adding weights and dividing by measured time puts widths
2, 3 and 4 on a single ceiling near 2.3 TB/s, while width 1 sits well under it, DRAM-bound:

| q4_K m=4096 k=14336 | activation requests | + weights | µs | rate |
|---|---|---|---|---|
| width 1 | 117 MB | 150 MB | 89.3 | 1.68 TB/s |
| width 2 | 235 | 268 | 118.2 | 2.27 |
| width 3 | 352 | 385 | 166.7 | 2.31 |
| width 4 | 470 | 503 | 222.4 | 2.26 |

`nr0` — rows per simdgroup — is the only knob that divides that traffic. Halving it to 1 costs
**1.51×** pooled over 26 shapes. Doubling it to 4 pays, **but only for Q6_K**, and the source says
why: `kernel_mul_mv_q6_K_f32_nr1_impl` reads its int8 sub-block scales inline, where the q4_K and
q5_K versions hoist all four into `ds[nr0][4] + dm[nr0][4]` — `8·nr0` registers held across the
whole super-block — and have no room left.

Measured on this model's real Q6_K tensors, both order halves, closing control within 0.1 %:

| tensor | k | m | width 3 | width 4 |
|---|---|---|---|---|
| `ffn_down` | 17408 | 5120 | **0.74×** | **0.58×** |
| `attn_qkv` | 5120 | 10240 | **0.74×** | **0.63×** |
| `output` | 5120 | 248320 | **0.73×** | **0.71×** |

End to end: **+4.1 %** over 20 interleaved pairs on one warm server (order gap 0.12 %) and
**+4.5 %** on an independent four-arm ABBA with a restart per arm, run by a separate reviewer.
Draft acceptance and tokens-per-forward-pass were identical to four decimals and generated text was
byte-identical in every pair, so this is the same speculation running faster rather than more of it.

Gated on width because at width 2 it is 1.01–1.06× — the extra rows only pay once the activation
stream is the binding constraint. Gated on row count because `nr0 = 4` leaves `ne01/(nr0·nsg)`
threadgroups, and a 1024-row matrix drops to 128 across 32 cores where it loses 1.5×.

`GGML_METAL_NO_Q6_NR0=1` restores the previous behaviour in the same binary.

## What is not established

**Why it helps as much as it does.** The change only fires when the verify pass is ≥ 3 tokens wide.
Under this configuration the mean width is 2.575, and the share of wide passes ranges from ~5 % to
~87 % across the prompt set — so the gain should scale with that share and vanish where there are
none. It does not: regressing per-prompt gain on it gives slope +0.0001, r = +0.002, and the gain
extrapolates to +4.5 % where the kernel should never run.

Either the batch width in the verify graph is not what it appears, or part of the gain has another
source. **The effect itself is not in doubt** — two measurement designs with different failure modes
agree, and at arm level three patched arms across a 40-minute window read 27.54 / 27.65 / 27.70
against unpatched arms at 26.62 and 26.34, with no overlap, which is a step on the switch rather
than a trend in time. But the mechanism is unexplained and stated here rather than smoothed over.

**A pre-existing out-of-bounds read gets wider.** These mat-vec kernels guard the *store*
(`first_row + row < args.ne0`) but not the *load*, so the last threadgroup reads up to
`nr0*nsg - 1` rows past `ne01` — 3 rows before this change, 7 after. It is latent on this model
because every affected tensor has `ne01` divisible by 8, and `test-backend-ops` cannot see it
because the results are still correct. It is not fixed here.

**The thresholds are tuned to this chip.** `ne01 >= 4096` is really a threadgroups-per-core
quantity — 16 per core on a 32-core part — written as a row count because that is what the dispatch
has to hand. Another part should re-derive it rather than inherit it.

## Test coverage worth having regardless

Nothing in upstream `test-backend-ops` exercises a K-quant mat-vec above **`ne01 = 16`**. Every
quantised `MUL_MAT` eval case is 16 rows wide, so any backend that switches kernels on row count is
untested there and a green suite says nothing about it.

This bit me directly: the change above is gated on `ne01 >= 4096` and passed 1143/1143 with it
**on and off**, because both arms ran identical code and the test could not have failed.

This fork adds `ne01` ∈ {4088, 4095, 4096, **4100**, 5120} × `n` ∈ {1,2,3,4,8} for q4_K/q5_K/q6_K,
straddling the boundary in both dimensions, with 4100 as the partial-tile case where
`first_row + row` runs past the end of the matrix. That block is independent of the Metal change
and is the piece most worth upstreaming.

Verification now looks like this — note that the count, not just the pass, is the evidence:

```
                          tests        _r0_4 pipelines compiled
patch on                  1218/1218    4
GGML_METAL_NO_Q6_NR0=1    1218/1218    0
```

`strings` on the dylib does **not** work as a check: with `GGML_METAL_EMBED_LIBRARY` the Metal
source is embedded as text and kernel `host_name`s are produced by the Metal preprocessor at
`ggml_metal_init`, so a name built from a macro argument exists nowhere in the binary.

## Build

```
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
cmake --build build -j
```

## Licence

MIT, as upstream. See [LICENSE](LICENSE).
