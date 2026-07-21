# Model Fit Estimation Design

## Scope

P2.3 adds pure, deterministic calculations over inspected model facts and the
source-attributed `HardwareInventory`. It does not load a model, mutate the
registry, choose a serving mode, or infer architecture from a model name.

Every estimate must classify each input as a provider fact, derived assumption,
measured value, or unknown. Missing architecture inputs produce an explicit
unknown result with reasons rather than a fallback estimate.

## Generic Estimates

### Artifact Residency

Use exact recognized weight-file bytes when present. Parameter-count estimates
are not a substitute for an available artifact inventory. Quantization packing,
scales, and metadata overhead must be explicit inputs if a parameter-derived
estimate is ever added later.

### Dense/GQA KV Cache

The generic dense/GQA formula is:

```text
bytes_per_token = 2 * layer_count * kv_head_count * head_dimension * element_bytes
total_bytes = bytes_per_token * context_tokens * concurrency
```

The factor of two represents key and value tensors. The calculation requires
all named inputs and rejects non-positive values. KV quantization overhead must
be supplied explicitly; it is not guessed.

This formula does not apply to MLA, recurrent state, mixed or sliding-window
attention, multimodal caches, MTP state, or family-specific compressed layouts.
Those remain unknown until a family adapter supplies a compatible calculation.

### Context

Advertised context, selected serving context, output cap, and KV window are
separate values. A generic selected context may be the minimum of explicit
provider and policy limits, but `max_kv_size` is not automatically a model
context limit.

### Conversion Workspace

Conversion disk is an explicit sum:

```text
source artifact + expected output + temporary workspace + manifests + reserve
```

The current `total_bytes * 2.2` heuristic is not evidence and must not be used by
the new estimator.

### Memory Margin

```text
usable memory = total unified memory - explicit system reserve
estimated peak = weights + KV/cache + explicit runtime overhead + explicit temporary buffers
margin = usable memory - estimated peak
```

Reserve and overhead values are versioned caller inputs or measured evidence,
not hidden constants. A negative margin is a calculation result, not an
activation prohibition.

## Architecture Boundary

Generic P2.3 estimation supports only complete dense/GQA metadata. Family
adapters in P2.4 own MoE active/total parameter semantics, mixed attention,
sliding windows, recurrent state, multimodal encoders, MTP state, quantization
overrides, and family-specific KV layouts.

The Laguna fixture is an acceptance example: its exact tensor bytes and
effective bits per weight are artifact facts, while its hybrid attention and
recurrent layout require a family adapter for predictive KV memory.

## Verification

Tests must cover exact artifact bytes, dense/GQA KV math, explicit quantization
overhead, context limit separation, conversion workspace sums, positive and
negative memory margins, measured-value precedence, invalid inputs, and unknown
architecture handling. Tests do not load models or invoke serving code.
