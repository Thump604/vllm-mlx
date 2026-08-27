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

Long-context growth beyond the measured envelope, sustained throughput, and
model-quality comparison remain unqualified. The QSA cache is not wired into
continuous batching, so that combination remains fail-closed rather than being
claimed from single-request results.

An exclusive-process, unquantized-KV context sweep on the 128 GiB Mac now
measures the all-resident Q4 envelope through 16,384 input tokens. Each level
used chunked prefill (`prefill_step_size=2048`), the vendor instruct sampling
tuple, and one generated token. The evaluated warm model occupied 96.56 GiB of
MLX active memory before a request.

| Input tokens | MLX peak | Increment over warm model | Prompt throughput |
|---:|---:|---:|---:|
| 128 | 97.07 GiB | 0.51 GiB | 177.4 tok/s |
| 1,024 | 98.78 GiB | 2.22 GiB | 411.0 tok/s |
| 4,096 | 103.30 GiB | 6.74 GiB | 435.7 tok/s |
| 8,192 | 107.63 GiB | 11.07 GiB | 364.5 tok/s |
| 16,384 | 116.36 GiB | 19.80 GiB | 299.7 tok/s |

The sweep recorded zero throttled pages and no process-attributed swap. System
memory pressure ended at 17 percent free. A 32K run was deliberately not
attempted: the measured growth projects beyond physical memory and would test
macOS compression/OOM behavior rather than establish a safe operating
contract. Consequently, 16K is a measured very-tight single-request point,
not a recommended service maximum, and the model's native 262K configuration
is not a claim that this artifact reaches 262K on this machine. Raw evidence is
`/Volumes/Lexar/qwen38-flash-next/mlx/Qwen3.8-Flash-Next-Q4-MLX/MEMORY_CONTEXT_SWEEP.jsonl`.

### Day-one Unsloth/llama.cpp comparison (2026-08-26)

The implementations are not at complete feature parity. Both implement the
base language architecture, PLE n-gram addressing, QSA, text generation, and
quantized loading. This MLX path has direct single-request text, streaming,
native-tool, thinking-separation, vision, and native MTP evidence on the local
Q4. The released Unsloth GGUF omits vision and MTP tensors. Its Qwen4-Exp
llama.cpp fork requires one server slot and F16 KV for this architecture. The
current MLX QSA/MTP state is likewise not wired into continuous batching and
fails closed rather than silently disabling MTP.

Unsloth now publishes seven Dynamic 3.0 GGUF policies from 67.56 GiB
(`UD-IQ1_S`) through 103.69 GiB (`UD-Q4_K_XL`). MLX currently has one measured
96.54 GiB mixed-Q4 policy. Nominal Q4 is not a parity statement: the block
formats and family-specific assignments differ, and comparative quality has
not been measured. The llama.cpp PR reports reference comparisons for its
architecture path, including WikiText-2 perplexity and QSA selection checks;
those are not transferable quality evidence for the MLX Q4 artifact.

### External-PLE Q4 qualification (2026-08-26)

mlx-vlm commit `fbc1449e` replaces the resident 29.80 GiB PLE parameter with a
read-only interleaved affine-Q4 row store. The store has 320,001,536 rows, a
160-element dequantized width, and one 100-byte packed record per row. It is on
the internal PCIe SSD at
`/Users/David/ai-models/ple/Qwen3.8-Flash-Next-Q4-MLX/ple-q4.rows`; the compact
ordinary-weight artifact is 67 GiB at
`/Users/David/ai-models/mlx/Qwen3.8-Flash-Next-Q4-MLX-External-PLE`. Thirty-two
sampled rows, including storage boundaries, matched the original resident Q4
checkpoint exactly after dequantization.

The warm model uses 71,677,702,240 MLX-active bytes. A short text generation
read 1,336 unique PLE rows (133,600 logical bytes), and total PLE lookup plus
dequantization time was 0.297 seconds. A 4,096-token repeated-token prefill ran
at 442.5 tok/s because only 40 rows missed the bounded cache. A deterministic
varied-token 4,096-token prefill produced 65,545 misses, read 6,554,500 logical
bytes, spent 9.683 seconds in PLE lookup/dequantization, and ran end-to-end at
217.8 tok/s. These are measured best- and pessimistic-locality bounds, not
representative-corpus quality evidence.

Reducing `prefill_step_size` from 2,048 to 512 materially improves the memory
envelope. The following repeated-token sweep generated one token at each level
using the vendor instruct sampling tuple:

| Input tokens | MLX peak | Prompt throughput | Wall time |
|---:|---:|---:|---:|
| 4,096 | 74.08 GB | 418.0 tok/s | 12.4 s |
| 16,384 | 78.05 GB | 342.7 tok/s | 50.4 s |
| 32,768 | 83.28 GB | 266.5 tok/s | 125.6 s |
| 65,536 | 93.63 GB | 179.4 tok/s | 368.2 s |
| 131,072 | 114.47 GB | 89.31 tok/s | 1,470.8 s |

The 128K run completed with no throttled pages and no observed swap-out growth;
the lowest sampled system memory availability was 11 percent. It proves the
128K context/memory envelope on this 128 GiB Mac. The repeated prompt is a
best-case PLE-locality workload and does not establish representative 128K
prefill speed or model quality.

The same artifact plus the Q4 MTP sidecar peaked at 73,707,907,026 MLX bytes.
Thinking and instruct generation completed at 16.95 and 18.83 tok/s,
respectively. vllm-mlx SimpleEngine returned exact instruct content with 8/6
MTP drafts accepted and streamed thinking into `reasoning_content` with no
protocol leakage. BatchedEngine rejected startup with the explicit
`does not support continuous batching` error, preserving the known boundary.

`mlx-vlm` PR #2045 now gathers distinct external-PLE rows in one storage
operation and runs one batched MLX Q4 dequantization before restoring request
order. A 65,536-row high-entropy microbenchmark on the internal SSD measured
8.47 seconds from cold pages versus 9.0–9.8 seconds for the row-at-a-time path.
With the pages resident, the batched path completed in 0.14 seconds. This is a
lookup microbenchmark, not an end-to-end throughput claim.

`mlx-vlm` PR #2046 makes a requested Qwen4-Exp MTP block size an adaptive
ceiling. The controller starts at the checkpoint-native depth, expands only
after eight rounds with at least 65% native-prefix acceptance, and backs down
when acceptance falls. The serving candidate should request block size 4 so
the measured 83.3% instruct profile can expand while the 52.9% thinking profile
normally stays at the native depth.

The existing vllm-mlx SSD tier stores evicted KV-prefix entries behind the
memory-aware continuous-batching cache. It does not tier model weights or
experts, and it is not available to this model's currently qualified serialized
route. It can reduce repeat-request prefill after continuous-batching QSA/MTP
support lands; it cannot reduce the current 71.68 GB resident model footprint.

For a text-only engine comparison, Unsloth `UD-Q4_K_XL` (111,323,630,080 GGUF
bytes) was served from Lexar by the isolated Qwen4-Exp llama.cpp fork at
`ef9fa1ba`. With one 4K slot, PLE forced to CPU/mmap, Metal for the remaining
layers, F16 KV, warmup disabled, and fit disabled, it loaded in 47.9 seconds,
used 107,350,880 KiB RSS, and left 14 percent system memory available. It
generated the exact instruct response at 25.07 tok/s and the exact thinking
response at 25.05 tok/s with separated reasoning and no protocol leakage.
This is faster short decode than the current MLX+MTP path, but uses roughly
43 GiB more process RSS, loads about 5.5 times more slowly from Lexar, provides
no vision or MTP from this GGUF, and is restricted to one slot. Storage location
is a comparison caveat: the compact MLX artifact is internal while the GGUF is
on the slower directly attached Lexar volume.

The local MLX intake now goes beyond the initial upstream boundary with a
Qwen4-Exp-specific native MTP drafter. It consumes the target's 10,240-wide
pre-mixer residual, preserves the four-stream residual between draft rounds,
uses the checkpoint's QSA/MoE/hyper-connection layer, and owns serialized
recurrent-state verification and rollback. This does not imply continuous-
batching support; the drafter advertises that boundary as false and the batched
engine rejects it before scheduling.

The standalone Q4/group-64 MTP artifact is 1,468,479,046 weight bytes at
`/Volumes/Lexar/qwen38-flash-next/mlx/Qwen3.8-Flash-Next-MTP-Q4-MLX`.
Direct generation returned `READY` with 2/2 drafts accepted at a combined
target-plus-drafter MLX peak of 105,336,489,248 bytes. vllm-mlx SimpleEngine
then passed an instruct response, a native `get_weather(city=Chicago)` call,
tool-result continuation, and streaming thinking/final separation. The tool
turn exercised rejection and rollback (24 drafted / 15 accepted); continuation
used the returned weather result (18 / 6); streaming thinking ended normally
with final `READY` and 61/31/92 usage. No duplicate call, parser failure,
protocol leakage, or lost tool result was observed.

The implementation commits are mlx-vlm `374aa186` and vllm-mlx `d0d76e3`.
The drafter manifest pins config SHA-256
`53f8553e12f86b225dcb84ebc5d79f7e038cc15ff2dc84bb6126a6efc33dd104`
and weight SHA-256
`49ec25f890c9c13b75e07b5043bf2b3349e297cb309a8292f5a22e35ca5f0e54`.

The first vllm-mlx SimpleEngine attempt exposed a serving-specific correctness
bug: `qwen4_exp_text` was not a registered extracted-text family, so
`text_model_from_vlm` silently constructed the generic Qwen3.5 TextModel. The
weights loaded mechanically, but the route emitted garbled text, reported zero
prompt tokens, and stopped by length. Local routing now fails closed for the
Qwen4-Exp family and keeps text requests on the correct loaded mlx-vlm model.

With that fix, SimpleEngine single-request serving passes streaming instruct,
streaming native tools, and thinking separation. Instruct returned `READY`
with correct 17/3/20 usage. The tool request returned one `get_weather` call
with `city=Chicago`, a `tool_calls` finish, and correct 287/27/314 usage. The
thinking response separated reasoning content and returned final content
`READY` with a normal stop. Warm-server generation measured 10.4 tok/s for the
tool call and 15.3 tok/s for the thinking probe. Continuous batching and MTP
were disabled and remain unqualified.

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
7. Extend Qwen4 MTP to continuous batching only after QSA auxiliary caches and
   four-stream drafter state have batch-aware merge/extract/filter/rollback
   evidence. The current serialized path must remain the default until then.
