# Qwen3.8-Flash-Next Investigation

Status: initial primary-source reconstruction and correctness slice, 2026-08-26.

## Source pins

- Official model revision: `Qwen/Qwen3.8-Flash-Next@f5d08274bafd880402bd16f5e3e6c514136ec06c`
- Transformers: `huggingface/transformers@36bc98ef9dd009569366f5e253ec1876ecafd925`
- mlx-lm: `ml-explore/mlx-lm@74e7cf9` (current upstream HEAD inspected)
- mlx-vlm: `Blaizzy/mlx-vlm@4857a6b0` (includes merged Qwen4-Exp PR #2032)
- llama.cpp: `ggml-org/llama.cpp@4d19b287691e8f47fc303be420f630c40ec45684`
- Unsloth: `unslothai/unsloth@60d2a636ba0332e3faac78cdcc091f815ca72c6f`
- Unsloth GGUF artifact:
  `unsloth/Qwen3.8-Flash-Next-GGUF@d3bc75ee6ccef3efc1e228ec00a6cc2cdb1e2249`
- Qwen technical report: <https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf>

The model card, config, weight index, and all 131 safetensors headers were
inspected. No weight payload was downloaded.

## Verified architecture

The artifact is `Qwen4ExpForConditionalGeneration`, model type `qwen4_exp`, and
is a causal language model with a 27-layer vision encoder. The language model
has 48 layers with a repeating three Gated DeltaNet layers followed by one QSA
layer (36 linear-attention and 12 QSA layers). Hidden size is 2,560.

The MoE has 512 routed experts, 10 selected per token, one shared expert, and
an intermediate size of 640. The official 125B-main / 6B-active description is
consistent with the released configuration and headers.

QSA uses 24 query heads, two KV heads, head dimension 256, partial RoPE
dimension 64, and a separate 4-query/1-key indexer with dimension 128. The
indexer compresses four tokens per block and has a 2,048-token (512-block)
budget.

Gated residuals use four streams and rank 320. One MTP layer is present and its
released tensors contain 2,607,150,848 parameters (4.856 GiB BF16).

The PLE n-gram table is injected at one-indexed layer 2. It has 16 independently
hashed heads: eight bigram heads and eight trigram heads. Each head returns 160
elements, so one token selects 16 rows and yields exactly 2,560 BF16 values, or
5 KiB. Hashes are deterministic from the current token and at most two prior
tokens, and history resets at EOS.

The released table contains 320,001,536 rows × 160 BF16 values =
51,200,245,760 parameters = 102,400,491,520 bytes (95.368 GiB). It is stored as
128 equal tensors of 2,500,012 rows. These are storage splits of one logical
table, not 128 simultaneously accessed heads.

Native context is 262,144 tokens. The model card describes extension to one
million with RoPE scaling; that extension is not native-equivalent evidence.

## Exact BF16 tensor map

Counts below come from tensor shapes in the released safetensors headers.

| Family | Parameters | BF16/source GiB |
|---|---:|---:|
| Routed MoE experts | 120,795,955,200 | 225.000 |
| N-gram embedding | 51,200,245,760 | 95.368 |
| MTP | 2,607,150,848 | 4.856 |
| Gated DeltaNet | 2,086,510,464 | 3.886 |
| Gated residual | 640,624,640 | 1.193 |
| Token embedding | 635,699,200 | 1.184 |
| LM head | 635,699,200 | 1.184 |
| QSA attention/indexers | 617,358,336 | 1.150 |
| Vision encoder | 448,931,056 | 0.836 |
| Shared experts | 235,929,600 | 0.439 |
| Routers | 63,037,440 | 0.117 |
| Other PLE tensors | 32,839,715 | 0.061 |
| **Total** | **179,999,981,459** | **335.276** |

### Affine quantization envelope

The following is a storage lower-bound using MLX affine groups of 64 and one
FP16 scale plus bias per group: effective bits/weight are nominal bits + 0.5.
It is not a quality recommendation, and not every tensor is necessarily
quantizable by the current module predicates.

| Family | Q8 GiB | Q6 GiB | Q5 GiB | Q4 GiB | Q3 GiB |
|---|---:|---:|---:|---:|---:|
| Routed experts | 119.53 | 91.41 | 77.34 | 63.28 | 49.22 |
| N-gram table | 50.66 | 38.74 | 32.78 | 26.82 | 20.86 |
| MTP | 2.58 | 1.97 | 1.67 | 1.37 | 1.06 |
| Gated DeltaNet | 2.06 | 1.58 | 1.34 | 1.09 | 0.85 |
| All remaining families | 3.28 | 2.50 | 2.12 | 1.73 | 1.35 |
| **Total theoretical** | **178.12** | **136.21** | **115.25** | **94.30** | **73.34** |

The n-gram row width is 160, which is not divisible by mlx-lm's default affine
group size 64. The current generic `quantize_model` predicate would therefore
leave the 95.368 GiB table in BF16. A Q4 conversion of everything else with an
unquantized table would be about **162.85 GiB** and cannot fit. A future model
implementation can explicitly choose group size 32 for `QuantizedEmbedding`;
that makes Q4 effectively 5 bits/weight for this table (29.80 GiB) and raises
the whole-model Q4 estimate from the theoretical 94.30 GiB to about **97.28
GiB**.

Properly configured all-resident Q4 is therefore about 97 GiB before allocator
overhead, temporary dequantized operands, recurrent state, KV/cache,
tokenizer/processor state, and macOS reserve. This machine reports
137,438,953,472 bytes, exactly 128 GiB. A 97.28 GiB model leaves 30.72 GiB for
macOS, runtime allocations, cache/state, and temporary buffers. That is a
credible all-resident configuration, not evidence of an inherently tight or
unusable fit. Whether the margin is comfortable at a chosen context length
requires measured high-water data.

Representative strategy envelopes before runtime/context memory are:

| Strategy | Approximate resident weights | Interpretation |
|---|---:|---|
| Generic current conversion (table remains BF16) | 162.85 GiB | Does not fit |
| All-resident Q4, PLE explicitly Q4/group-32 | 97.28 GiB | Credible 128 GiB baseline; measure peak at target context |
| Experts Q4, PLE Q3/group-32, other tensors Q8 | ~90 GiB | Plausible but quality unsupported |
| External PLE, experts Q5, other tensors Q8 | ~83 GiB plus PLE cache | Most interesting higher-quality hypothesis |
| External PLE, experts Q6, other tensors Q8 | ~97 GiB plus PLE cache | Potentially feasible; similar weight envelope to all-resident Q4 |

For the 12 QSA layers, unquantized K+V alone is approximately 24 KiB/token if
the implementation retains full keys and values: about 6 GiB at 262,144 tokens.
The exact MLX cache design must be measured because QSA index state and sparse
selection can change this accounting.

## Support state

- **Transformers:** released implementation is present. It defines Gated
  DeltaNet, micro-block QSA, gated residuals, PLE indexing, multimodal merging,
  and cache state. The n-gram weight is explicitly excluded from ordinary
  device placement, allowing a different device, but this is not NVMe caching.
- **vLLM/SGLang:** named by the official model card as supported. Their GPU
  strategies and kernels are reference implementation sources, not directly
  portable to Metal.
- **mlx-lm:** no `qwen4_exp` implementation or active source reference at the
  inspected HEAD. Its loader calls `mx.load` for every shard, constructs one
  weight dictionary, and evaluates all parameters unless `lazy=True`.
- **mlx-vlm:** Qwen4-Exp support merged in PR #2032 while this investigation
  was active. It implements config parsing, the Qwen3 vision encoder, gated
  residuals, DeltaNet, QSA and its auxiliary cache, MoE, sharded PLE lookup,
  text/multimodal generation, and checkpoint sanitization. Its documented
  boundary excludes MTP and continuous batching for QSA. Static intake found
  two conversion defects in the merged implementation: the official FP8
  artifact's expanded per-expert tensors and per-tensor FP8 PLE scale were not
  handled, and a model-owned PLE group-32 override was evaluated after the
  converter's default group-64 divisibility rejection. Both are corrected and
  covered offline on local isolated mlx-vlm branch
  `604/qwen38-flash-next-upstream-intake` at `127d943e`; no upstream PR has
  been opened.
- **MLX:** `mx.load` is lazy, but there is no public external-row-backed
  embedding abstraction. Deferring evaluation is not proof that a gather can
  avoid materializing the complete source array.
- **llama.cpp:** the inspected HEAD has no Qwen4-Exp architecture registration,
  tensor mapping, or PLE table implementation. Existing `ngram-*` flags concern
  speculative decoding and are unrelated to the model's PLE table.
- **Unsloth:** the Day-0 Dynamic 3.0 GGUF repository appeared during this
  investigation and remains explicitly marked WIP. At the pinned revision it
  publishes only `UD-IQ1_S`, split across three files totaling 67.56 GiB. The
  10.4 MiB first shard holds common metadata; range reads of the other two GGUF
  headers expose all 1,224 language-model tensor descriptors without
  downloading their payloads. The artifact contains neither vision nor MTP
  tensors, so it is not a complete conversion of the released conditional-
  generation checkpoint.

### Verified Unsloth Day-0 tensor policy

This is post-training GGUF quantization using a 45-chunk, 926-entry importance
matrix. It is not evidence of an Unsloth-specific NVMe cache or external PLE
backend. Generic llama.cpp mmap remains distinct from the quantization policy.

| Tensor family | Observed GGUF types | Approximate GGUF payload |
|---|---|---:|
| PLE n-gram table | IQ4_NL | 26.822 GiB |
| Routed expert projections | IQ1_S, IQ2_XXS, IQ4_NL | 37.109 GiB |
| Routers | F32 | 0.234 GiB |
| QSA/indexer | Q5_K, Q6_K, F32 | 1.355 GiB |
| Gated DeltaNet | Q6_K, F32 | 0.471 GiB |
| Gated residual | Q8_0, F32 | 0.647 GiB |
| Shared experts | Q5_K, Q6_K, Q8_0, F32 | 0.179 GiB |
| Token embedding and head | Q4_K | 0.666 GiB |
| Vision and MTP | absent | 0 GiB |

The policy is materially sensitivity-aware: it retains routers in F32, uses
Q8 for the low-rank gated-residual projections, Q5/Q6 for attention and
recurrent matrices, and concentrates the most aggressive IQ1/IQ2 formats in
the routed experts. The enormous PLE table is ordinary resident IQ4_NL in the
GGUF; there is no special sparse-storage representation. MLX Q4/group-32 has
the same 26.82 GiB table payload estimate, but it is not numerically or
operationally equivalent to IQ4_NL and requires independent quality evidence.

## Recommended integration boundary

Implement the multimodal architecture in mlx-vlm and share the language
components with mlx-lm if upstream maintainers choose that structure.
vllm-mlx should consume those APIs through its existing SimpleEngine and
BatchedEngine paths. A vllm-mlx compatibility model would duplicate a large,
rapidly evolving architecture and is not justified before upstream direction
is known.

The first local slice is deliberately narrower:
`vllm_mlx.utils.qwen4_exp_ngram` reconstructs the released logical table,
generates the exact bigram/trigram row IDs, resets history at EOS, and maps a
global row to its 128-way storage split. Tests bind it to the released header
shape and compare randomized sequences against an independent transcription of
Transformers' vectorized shift/gather formulation. It changes no serving
behavior and provides a parity primitive for an upstream model and a cold-table
prototype.

The original isolated `mlx-vlm` branch `604/qwen38-flash-next` is preserved as
an early prototype. Upstream subsequently merged its own complete base Qwen4-Exp
architecture in PR #2032. Active intake therefore moved to
`604/qwen38-flash-next-upstream-intake`, based on upstream commit `4857a6b0`,
instead of duplicating the merged model.

Static checkpoint inspection found two conversion defects in that upstream
implementation. First, the official FP8 artifact stores each routed expert as
separate per-expert gate/up/down tensors with inverse scales, while the model
expects packed expert tensors. Second, the generic quantizer checked the global
group size before the model-specific predicate, so the 160-wide PLE rows were
silently left unquantized when the requested global group size was 64. Local
commit `127d943e` restores and packs the official FP8 layout and permits the
model predicate to select Q4/group-32 for PLE while retaining Q4/group-64 for
ordinary tensors. Four focused synthetic tests pass; this is conversion-path
evidence, not full-model or generation qualification.

Local commit `9c538470` adds the first model-owned read-only mmap PLE storage
experiment. Resident MLX and mmap row lookups return identical synthetic rows.
The mmap backend validates its versioned manifest and exact file size, confines
the data file beside the manifest, supports a bounded row LRU, and reports
lookups, hits, misses, bytes read, and elapsed lookup time. It is deliberately
not integrated into model loading and cannot affect normal serving. Two focused
storage tests pass.

## Cold n-gram feasibility

Decode needs 16 rows/token × 160 BF16 values = 5 KiB/token. With 4 KiB VM pages,
the pessimistic cold-read amplification is 64 KiB/token. At 20 tokens/s this is
only 1.25 MiB/s but requires up to 320 random page reads/s. Bandwidth is not the
primary decode risk; page-fault latency, synchronization, and cache miss rate
are.

The indices are known as soon as the token is selected and its two-token
history is available. Current-token rows cannot be prefetched before sampling,
but the gather can overlap other per-token work only after selection. Batches
increase the number of independent rows nearly linearly unless prompts share
token n-grams.

Prefill is less favorable: sequence length `S` requires `16S` row lookups. At
262K tokens that is 4.19 million logical row reads and up to 16 GiB of page
traffic without locality. The hash functions intentionally spread rows across
the table, so a small LRU should not be assumed effective. Real token/n-gram
repetition may still provide useful locality and must be measured.

An mmap-backed row store plus CPU gather and a 5 KiB/token transfer to MLX is
technically plausible. Current MLX embeddings cannot directly express this
backing. A custom Metal kernel cannot make NVMe memory directly addressable and
should not be attempted before a CPU/mmap prototype measures page faults,
latency, and hit rate. macOS page cache should be measured as the baseline
before adding an application LRU.

If the table is kept outside resident memory, it saves 26.82 GiB versus Q4 or
32.78 GiB versus Q5. That can make a higher-quality base configuration possible,
but only if sparse-read latency and quality of the separately quantized table
pass measurement.

## Current decision

Evidence does not rule out **A**. Once PLE uses Q4 with group size 32, plain
all-resident MLX Q4 is the required baseline and may fit comfortably on this
exact 128 GiB machine. Special handling is justified only if measured peak
memory, context envelope, throughput, or quality shows a material advantage.
B and C remain hypotheses until those measurements are available. The generic
converter's prior behavior of silently leaving the PLE n-gram tables in BF16
was a separate conversion defect; local intake commit `127d943e` addresses it.

## Offline conversion result

The pinned official FP8 snapshot at revision
`bcd9f01ddc9cff2316eb84281bebcd5b058bddce` was downloaded to Lexar and
verified byte-for-byte: 144 files, 185,563,783,127 bytes, and 133 LFS SHA-256
objects with no mismatches. All 152,089 indexed source tensors reconcile with
the 131 safetensors shard headers.

Using mlx-vlm intake commit `9c538470`, the verified source converted to an
all-resident affine MLX artifact with Q4/group-64 as the global policy,
Q4/group-32 for all 128 PLE shards, and Q8/group-64 for all 96 router gates.
The converter reported 4.675 effective bits per weight. The output contains 20
weight shards with 103,664,015,352 indexed bytes (approximately 96.5 GiB), and
all 3,767 indexed output tensors reconcile with their shard headers.

A strict lazy load binds all 3,767 parameter leaves to
`mlx_vlm.models.qwen4_exp.qwen4_exp.Model`. This validates architecture
construction, quantized-module reconstruction, and weight-key accounting only;
it does not evaluate weights or run a forward pass. The artifact is published
at `/Volumes/Lexar/qwen38-flash-next/mlx/Qwen3.8-Flash-Next-Q4-MLX`, with
machine-readable evidence in `CONVERSION_VERIFICATION.json` and file hashes in
`MANIFEST.sha256`. MTP extraction remains disabled because no Qwen4-specific
draft splitter exists upstream.

The initial Q4 text inference gate now passes in both required Qwen modes. An
instruct probe using the vendor sampling policy returned exactly `READY` in two
generated tokens. A thinking probe using the vendor thinking policy emitted a
complete reasoning block, closed `</think>`, and returned `READY` in 35 tokens.
Measured MLX peaks were 103.87 GB and 103.99 GB respectively; peak process
footprints were 104.59 GB and 104.81 GB. macOS used transient compression and
swap during prefill, but neither run reported throttled pages, an OOM, or a
child-process swap, and memory recovered after exit. This is a tight but
working 128 GiB envelope, not yet a comfortable concurrent-service envelope.

Single-request streaming, native tool generation, and a minimal multimodal
probe also pass. Streaming emitted two incremental events and reconstructed
`READY` with a normal stop. A native `get_weather` call produced one complete
Qwen XML call with `city=Chicago`; the existing vllm-mlx `qwen3_xml` parser
decoded it without residual content or duplication. The vision path correctly
counted two cats in the 640×480 mlx-vlm fixture at a 448×448 resize. Peak MLX
memory across these probes ranged from 103.87 GB to 104.74 GB.

Long-context growth, sustained throughput, and model-quality comparison remain
unqualified. MTP remains unsupported by the current upstream model
implementation, and upstream documents that the QSA cache is not wired into
continuous batching. Neither feature is claimed from the single-request
results.

## Next falsifiable experiments

1. Finish and hash-verify the pinned official FP8 snapshot on Lexar.
2. Convert one complete all-resident MLX baseline with ordinary tensors at
   Q4/group-64 and PLE at Q4/group-32; retain complete weight-load accounting.
3. Perform config, header, and load-only validation without generation.
4. After the inference server is released, compare component outputs and short
   deterministic generation against Transformers at declared tolerances.
5. Measure resident baseline peak memory before deciding whether external PLE
   storage is necessary.
6. If external PLE remains justified, measure cold/warm decode, prefill, page
   faults, bytes/token, and batch scaling for page-cache-only and bounded-LRU
   mmap backends.
7. Add MTP only through a Qwen4-specific draft splitter and architecture; the
   current upstream implementation explicitly excludes MTP, so no support is
   claimed from the presence of MTP tensors in the official checkpoint.
