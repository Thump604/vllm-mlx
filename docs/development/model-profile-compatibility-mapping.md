# ModelProfile Legacy Compatibility Mapping

Status: PR3D explicit finalization slice
Runtime wiring: none

## Boundary

`vllm_mlx.model_profile_compat` is a pure adapter for already-loaded legacy
records. This slice accepts acquisition, conversion, registration, registry,
CLI-server, and qualification inputs and returns an incomplete ModelProfile v1 import-result
envelope. It does not read files, download artifacts, mutate runtime state,
start a server, qualify a model, or finalize a profile.

`finalize_legacy_model_profile` is the separate explicit completion boundary.
Its caller supplies a complete candidate profile and both committed schemas.
The finalizer rejects non-missing import errors, changes to imported facts,
and removal or mutation of imported provenance records. It delegates profile
schema, subject-digest, evidence-binding, and cross-field checks to the existing
ModelProfile validator before returning `complete=true`. It does not infer
missing facts, run qualification, resolve a profile, or activate a model.

The implementation has seven ownership modules:

- `vllm_mlx/model_profile_compat.py` is the stable public facade. It re-exports
  the three result/input dataclasses and owns the keyword-only dispatcher.
- `vllm_mlx/_model_profile_compat_types.py` owns `SourceKind` and the immutable
  dataclass implementations. Their supported import path remains
  `vllm_mlx.model_profile_compat`.
- `vllm_mlx/_model_profile_compat.py` is the private import engine. It owns
  source normalization, assignment and conflict tracking, source-specific
  orchestration, provenance, and missing-fact collection.
- `vllm_mlx/_model_profile_serving_compat.py` is the private PR3B serving
  mapper. It depends only on the shared types and serving-vocabulary modules and
  owns nested-serving normalization and serving diagnostics.
- `vllm_mlx/_model_profile_serving_vocab.py` owns the closed feature, sampling,
  policy, limit, and top-level serving vocabularies used by the PR3B mapper.
- `vllm_mlx/_model_profile_qualification_compat.py` owns PR3C normalization of
  already-recorded qualification evidence. It does not run a qualification,
  recompute a subject digest, finalize a profile, or mutate runtime state.
- `vllm_mlx/_model_profile_finalization.py` owns PR3D preservation checks and
  delegates final schema and semantic validation to the committed validators.

Each input carries a payload, source location, and SHA-256. Output source
descriptors intentionally omit payloads. The result is an audit record and is
not an activation input: `complete` is always `false` in this slice.

## Source Mapping

| Legacy source | Profile destinations | Boundary |
|---|---|---|
| Acquisition | provider, repository ID, requested/resolved revision, artifact source and inspection facts | A requested revision is not an immutable resolved revision. |
| Conversion | MLX format, output inspection facts, quantization recipe | Failed or unverified conversion output contributes no artifact facts. |
| Registration | artifact ID, served name, alias, profile sampling defaults, template kwargs, parser declarations | Registration feature flags and `production_ready` are not v1 feature state or qualification evidence. |
| Registry entry | Serving facts only | Registry names, paths, sources, preload, memory, and MLLM hints do not establish identity or aliases; fields without a lossless v1 target are diagnosed. |
| CLI server | Serving facts only | Nested `serving` values take precedence over conflicting top-level values and the conflict is retained as a diagnostic. |
| Qualification record | Qualification status and evidence history only | Each evidence record must have the exact v1 fields, hashes, result, and timestamp. `qualified` requires a structurally valid passing record; `fail` and `incomplete` records remain preserved history. |

The adapter copies imported values and never mutates input mappings. When two
accepted sources assign different values to one destination pointer, the first
deterministic source-order value is retained and a `conflicting_value` issue
names both source locations.

## Provenance And Missing Facts

Imported fields are grouped into `provenance.records` as provider facts,
derived recommendations, or maintainer policy. Source location and SHA-256 are
preserved; an immutable source revision is included when present.

Registry and CLI sources map the closed v1 serving vocabulary only: engine,
route, template, parsers, sampling, limits, features, activation policy, and
request policy. Direct legacy flags map only where their semantics are explicit:
`continuous_batching`, `enable_mtp`, SpecPrefill enablement and draft-model
setting, `max_tokens`, `max_request_tokens`, and `max_kv_size`. Unknown fields,
features, and settings; malformed policies; and non-boolean feature declarations
remain deterministic diagnostics rather than inferred facts.

Every required v1 fact not established by these six source kinds produces a
`missing_required_fact` issue with its JSON Pointer. The mapper does not infer
capabilities, hashes, context limits, feature states, policies, or
qualification from model names, parser names, feature flags, or booleans.

Generic command `status`, `production_ready`, and bare qualification booleans
are recorded only as deterministic warnings. They never establish qualification
truth. PR3C preserves a valid `not_qualified`, `failed`, or `blocked` status,
but import always returns `complete=false`; binding evidence to the final
canonical subject digest and promoting a complete profile happen only through
the explicit PR3D finalization boundary.

The serialized result follows
`schemas/model-profile-import-result-v1.schema.json` and can be checked with
the committed `vllm_mlx.model_profile_import` validator APIs. A future slice may
add other source kinds; neither import nor finalization performs runtime
resolution or activation.
