# Model Onboarding Acceptance Fixture: Laguna S 2.1

## Purpose

Laguna S 2.1 is the reference acceptance fixture for the ModelProfile and
lifecycle product. It represents the current owner workflow for adding a large
MLX model: acquire an exact source revision, convert it on a specific machine,
verify the resulting artifact, inspect the installed implementation, write a
manual serving contract, and defer runtime qualification until a large-model
window is available.

The fixture is deliberately split into proven facts and qualification work. A
successful conversion must not be reported as a qualified model, and a
qualified model must not automatically become a Jobs, resident, default, or
Open WebUI model.

## Fixture Identity

| Field | Value |
|---|---|
| Model | `poolside/Laguna-S-2.1` |
| Source revision | `a50e85e7e0aae7b0a504d156bd36a616ec9fea38` |
| Source artifact | `/Volumes/Lexar/source_models/Laguna-S-2.1-BF16` |
| Converted artifact | `/Volumes/Lexar/mlx_models/Laguna-S-2.1-4bit` |
| Conversion manifest | `/Volumes/Lexar/mlx_models/Laguna-S-2.1-4bit/conversion-manifest.json` |
| Checksums | `/Volumes/Lexar/mlx_models/Laguna-S-2.1-4bit/SHA256SUMS` |
| Format | MLX safetensors, affine 4-bit, group size 64 |
| Router | Affine 8-bit with 47 recorded overrides |
| Effective bits per weight | `4.501` |
| Indexed tensors / weights | `1,962` / `66,147,556,864` bytes |
| Source architecture | 48 layers, 256 routed experts, one shared expert |
| Attention layout | 12 global layers, 36 sliding layers, window 512 |
| Initial local envelope | 32K context, 32K output ceiling, one request at a time |
| Initial posture | Artifact-only, unqualified, not registered or exposed |

The two local paths above are evidence locations for this owner's handoff, not
portable catalog fields. Any upstream fixture must replace them with repository
URIs, immutable revisions, and content hashes; local installation paths belong
only in mutable installation state.

## What Is Already Proven

The following claims are supported by the handoff artifacts and may be imported
as provider facts or artifact-integrity evidence in a ModelProfile. They do not
require a model-generation claim.

- The exact source revision was identified.
- A complete MLX 4-bit artifact was produced with the recorded conversion
  command and tool versions.
- All 13 MLX shards and all indexed tensors passed SHA-256 and non-empty
  checks.
- All 48 layers are represented.
- All 256-expert stacked matrices and 47 sparse router overrides are present.
- Index, config, tokenizer, and chat-template files passed checksum checks.
- Thinking-on and thinking-off chat-template rendering passed structural
  rendering checks.
- The installed `mlx-vlm` package contains a Laguna implementation covering
  the model configuration, mixed attention, rotating sliding-layer KV cache,
  routing, shared expert, expert stacking, router remapping, quantization
  predicate, correction bias, and weight sanitation.
- The conversion did not change the live serving slot, restart a service, load
  a model concurrently, or publish a registry/mode row.

These facts establish artifact integrity and implementation availability. They
do not establish that the installed implementation is correct for this exact
artifact under vllm-mlx.

## What Remains Unproven

The following remain later-stage qualification claims and must not be inferred
from the structural checks:

- Full model instantiation and successful health/readiness.
- Source-to-MLX logits agreement on a bounded prompt.
- Thinking-on and thinking-off generation with nonempty final content.
- `poolside_v1` reasoning and tool-call parsing.
- Reasoning preservation across assistant/tool turns.
- Tool response continuation and malformed-tool failure behavior.
- Memory envelope, queue behavior, cancellation, and recovery.
- 32K, 64K, or 128K context behavior beyond the initial policy ceiling.
- Coding quality, latency, throughput, or comparative value against Qwen 35B.
- Continuous batching, prefix cache, MTP, SpecPrefill, KV quantization, or any
  speculative backend.
- Jobs, resident/default, Open WebUI, or production-routing suitability.

The one-million-token advertised context is not a local qualification claim.
The initial profile must remain at 32K until exact memory and correctness
evidence supports a higher envelope.

## Acceptance Fixture

The fixture is accepted only when each stage has a durable artifact and the
stage status is explicit. A later stage may consume earlier artifacts but may
not rewrite an earlier fact into a stronger claim.

| Current manual step | Evidence or input | Product stage that automates it | Acceptance output |
|---|---|---|---|
| Select exact upstream model revision | Repository and revision in this document | Model intelligence/catalog | Provider-fact identity with immutable revision |
| Locate BF16 source and MLX destination | Source and destination paths | Lifecycle acquisition | Resumable acquisition record with ownership and completion state |
| Convert with hand-written command | Conversion command and tool versions | Lifecycle conversion | Versioned conversion request and conversion manifest |
| Verify index, tensors, shards, and checksums | `conversion-manifest.json`, `SHA256SUMS`, structural report | Artifact validation | Machine-readable integrity result bound to revision and artifact digest |
| Inspect installed family implementation | Installed `mlx-vlm` Laguna files | Family adapter/model intelligence | Adapter capability record and upstream implementation reference |
| Write sampling/template/parser policy | Model-card contract in handoff | Profile resolution | Profile defaults with provenance; no hidden runtime fallback |
| Choose initial context/output/KV envelope | 32K/32K conservative policy | Hardware fit and recommendation | Derived recommendation showing memory and safety margin |
| Keep model out of serving surfaces | Manual status flags | Lifecycle activation and exposure | Explicit artifact-only state; no accidental registration or routing |
| Run lazy load and health checks | Runtime test artifact | Lifecycle validation | Readiness result tied to the exact profile and artifact |
| Compare source and MLX logits | Bounded comparison fixture | Qualification harness | Reproducible numerical evidence and tolerance decision |
| Run generation and parser checks | Thinking, tool, and continuation fixtures | Qualification harness | Raw output, parsed output, safety result, and parser evidence |
| Measure memory, queue, cancellation, and recovery | Controlled one-request window | Lifecycle supervision | Measured hardware-fit and recovery evidence |
| Publish a selectable candidate mode | All preceding evidence | Profile registration/activation | Mode row with exact status, evidence, and allowed uses |
| Decide owner/Jobs/Open WebUI exposure | Owner decision plus mode evidence | Lifecycle control plane/application | Explicit exposure flags; never implied by registration |

## Initial Profile Boundary

The first profile may be represented as:

```text
laguna_s_2_1_4bit_baseline_thinking
status: artifact_only_unqualified
runtime_registered: false
ops_exposed: false
openwebui_exposed: false
jobs_allowed: false
```

The initial model-card-native generation contract is temperature `1.0`,
`top_p=1.0`, `top_k=20`, `min_p=0.0`, thinking enabled, a 32K output policy
ceiling, `poolside_v1` reasoning/tool parser names, and preservation of
`reasoning_content` across assistant history. Thinking-off is an explicit
request or separate mode, not an implicit fallback. No alternate sampling or
acceleration feature is part of the initial fixture.

## Qualification Order

Qualification must use a clean worktree and must not disturb an active Jobs
lease:

1. Lazy-load and structural model instantiation.
2. Bounded source-to-MLX logits comparison.
3. Minimal thinking-off generation.
4. Minimal thinking-on generation with nonempty final content.
5. Multi-turn reasoning preservation.
6. One tool-call round trip and tool-response continuation.
7. Parser coercion and malformed-output tests.
8. 32K context smoke, memory, queue, and cleanup checks.
9. Controlled owner coding comparison against Qwen 35B.

Only after those artifacts pass may a selectable candidate mode be considered.
Each later feature or exposure requires its own exact evidence; it does not
inherit qualification from the baseline.

## Acceptance Rule

Laguna S 2.1 is a successful product fixture when the owner no longer needs to
manually coordinate paths, conversion commands, checksums, profile defaults,
qualification order, and exposure boundaries, while every resulting claim
remains traceable to an immutable artifact. The fixture is not complete merely
because the model downloads, converts, or renders a prompt.
