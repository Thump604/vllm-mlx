# ModelProfile Legacy Compatibility Mapping

Status: first reviewable import slice
Runtime wiring: none

## Boundary

`vllm_mlx.model_profile_compat` is a pure adapter for already-loaded legacy
records. This slice accepts acquisition, conversion, and registration inputs and
returns an incomplete ModelProfile v1 import-result envelope. It does not read
files, download artifacts, mutate runtime state, start a server, qualify a
model, or finalize a profile.

The implementation has three ownership modules:

- `vllm_mlx/model_profile_compat.py` is the stable public facade. It re-exports
  the three result/input dataclasses and owns the keyword-only dispatcher.
- `vllm_mlx/_model_profile_compat_types.py` owns `SourceKind` and the immutable
  dataclass implementations. Their supported import path remains
  `vllm_mlx.model_profile_compat`.
- `vllm_mlx/_model_profile_compat.py` is the private import engine. It owns
  source normalization, assignment and conflict tracking, source-specific
  mapping, provenance, and missing-fact collection.

Each input carries a payload, source location, and SHA-256. Output source
descriptors intentionally omit payloads. The result is an audit record and is
not an activation input: `complete` is always `false` in this slice.

## Source Mapping

| Legacy source | Profile destinations | Boundary |
|---|---|---|
| Acquisition | provider, repository ID, requested/resolved revision, artifact source and inspection facts | A requested revision is not an immutable resolved revision. |
| Conversion | MLX format, output inspection facts, quantization recipe | Failed or unverified conversion output contributes no artifact facts. |
| Registration | artifact ID, served name, alias, profile sampling defaults, template kwargs, parser declarations | Registration feature flags and `production_ready` are not v1 feature state or qualification evidence. |

The adapter copies imported values and never mutates input mappings. When two
accepted sources assign different values to one destination pointer, the first
deterministic source-order value is retained and a `conflicting_value` issue
names both source locations.

## Provenance And Missing Facts

Imported fields are grouped into `provenance.records` as provider facts,
derived recommendations, or maintainer policy. Source location and SHA-256 are
preserved; an immutable source revision is included when present.

Every required v1 fact not established by these three source kinds produces a
`missing_required_fact` issue with its JSON Pointer. The mapper does not infer
capabilities, hashes, context limits, feature states, policies, or
qualification from model names, parser names, feature flags, or booleans.

The serialized result follows
`schemas/model-profile-import-result-v1.schema.json` and can be checked with
the committed `vllm_mlx.model_profile_import` validator APIs. A future slice may
add other source kinds or explicit completion; neither is part of this module's
public dispatcher today.
