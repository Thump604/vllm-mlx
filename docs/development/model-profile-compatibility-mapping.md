# ModelProfile Legacy Compatibility Mapping

Status: P1.3 implementation reference
Runtime wiring: none

## Boundary

`vllm_mlx.model_profile_compat` is a pure compatibility adapter. It accepts
already-loaded legacy records and returns a ModelProfile fragment plus explicit
issues. It does not read files, download artifacts, mutate a registry, resolve a
model name, start a server, or qualify a model.

Adapter inputs use `LegacySourceInput`, which contains the loaded payload plus
its location and SHA-256. Import-result `sources` are output descriptors and
intentionally omit payload bytes; an output envelope is an audit record, not a
new adapter input.

Only a result with `complete=true` and a profile that passes the canonical
schema and P1.4 semantic validation may become an activation input. Existing
commands continue to use their current manifests and defaults during P1.

## Source Mapping

| Legacy source | Profile destinations | Required cautions |
|---|---|---|
| Acquisition manifest | identity repository/requested/resolved revision, artifact source and size | Requested revision is not an immutable resolved revision |
| Conversion manifest | artifact format, dtype, quantization recipe, conversion provenance | A recipe is not proof that output bytes match it |
| Registration manifest | served name, alias, declared sampling, template kwargs, parser declarations | Registration is a handoff and is not consumed by current serving paths |
| Registry entry | engine feature declarations and memory estimate | Entry values inherit separate startup defaults; unresolved values remain issues |
| Explicit CLI/server record | route, engine, limits, template/parser policy, feature and request policy | Runtime fallback constants are maintainer policy, not provider facts |
| Qualification record | evidence only when artifact hash, subject digest, hardware, workload, and result are present | `production_ready` and a successful command return code are not qualification evidence |

## Conflict Rules

The adapter never collapses these concepts:

- provider repository ID;
- requested and resolved provider revisions;
- local installation location;
- immutable artifact ID;
- registry key;
- lifecycle key;
- served model name and aliases.

When explicit sources disagree about one destination pointer, the adapter keeps
the deterministic source-order candidate and emits a conflict issue naming all
sources and values. Conflicts between top-level and nested serving values in one
source are also reported; the explicit nested serving value is retained. P1.3
does not silently choose the newest file or a value that merely appears more
specific.

Registry fields without a lossless v1 destination, including preload policy,
GPU utilization, stream interval, generic MLLM selection, prefill step size, and
unbound memory estimates, remain explicit compatibility issues. They are not
dropped and are not forced into unrelated profile fields.

## Missing-Fact Rules

Current legacy records normally cannot prove all of the following:

- config, tokenizer, chat-template, and weights-manifest hashes;
- provider and profile sampling provenance;
- complete modality, tool, reasoning, streaming, and structured-output support;
- template source and hash;
- advertised context, selected serving context, output caps, and KV capacity as
  distinct values;
- a complete state for every v1 feature;
- activation and request policy;
- evidence bound to an exact subject digest.

Each missing required fact produces a stable issue code and JSON Pointer. The
adapter must not fill gaps from model-name heuristics, generic parser fallback,
environment variables, or a `production_ready` boolean.

## Installation State

Legacy artifact paths are source locations for import and future installation
records. They are not copied into immutable ModelProfile identity. This keeps a
profile portable when an artifact is relocated or installed on another Mac.

## Qualification

Qualification command manifests are execution handoffs. They become measured
evidence only when the supplied record includes:

- a passing result;
- evidence artifact location and SHA-256;
- hardware fingerprint;
- workload ID;
- the exact subject digest.

Even then, P1.3 only maps the record. P1.4 validates digest equality and the
full qualification invariants before `qualified` is permitted.

## Next Boundary

P1.4 owns canonical digest calculation, JSON Schema validation, semantic set and
limit checks, migration fixtures, and the complete/incomplete transition. P1.3
must remain a transformation layer so those checks can be tested independently
of serving.
