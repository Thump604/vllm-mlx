# Model Configuration Authority Inventory

Status: P1.1 source inventory
Scope: repository source at `0dd1157` plus documentation-only branch changes
Method: Luna-low inventory, parent source review, and Luna-high independent verification
Runtime or model calls: none

## Finding

vllm-mlx does not currently have one model configuration authority. Five
partially overlapping systems describe model identity or serving behavior:

1. CLI arguments and `server.py` globals for standalone serving.
2. Registry YAML and `ModelManager` for registry-backed serving.
3. `ModelSpec` and `ResidencyManager` for standalone residency.
4. Acquisition, conversion, and registration manifests for artifact handoff.
5. Qualification request/results for benchmark handoff.

The artifact workflow is not connected to the runtime configuration flow.
Registration and qualification manifests are generated and tested, but the
runtime registry and server do not consume them. P1.2 must define one typed
profile and explicit precedence without changing generation behavior.

## Authority Map

| Concern | Current authorities | Effective precedence | Conflict or missing authority |
|---|---|---|---|
| Model identity | CLI model, `server._model_name`, registry entry name, `RegisteredModel.name`, `ModelSpec.model_key/model_name`, request `model`, registration `model_id/served_model_name` | Standalone uses CLI model and optional served alias; registry uses the request model as registry key | Provider ID, local artifact, served alias, lifecycle key, and source path are not one identity |
| Artifact and revision | Acquisition options/manifest, inspection, conversion manifest, registry source, downloader cache | Acquisition passes the requested revision; runtime resolves registry source independently | Requested tags are not normalized into a required immutable revision; no canonical artifact digest |
| Tool and reasoning parsers | CLI flags, server globals, parser registries, generic tool-parser fallback, registration parser policy | Runtime CLI/server globals select configured parsers; tool parsing falls back to the generic parser when auto selection is disabled or initialization fails | Registration policy is not consumed; no profile-bound parser support, provenance, or fallback contract |
| Chat template | Tokenizer/processor template, server default template kwargs, request kwargs, request `enable_thinking`, environment/model-name thinking defaults, engine-specific rendering | Server merges request template kwargs over server defaults, but individual engine paths separately derive `enable_thinking` and then merge template kwargs | Resolution is duplicated across text, batched, and multimodal paths; thinking precedence differs; no template hash authority |
| Sampling | API request fields, CLI defaults, server fallback constants, engine/model defaults, registration serving defaults | Request fields generally override server defaults on API paths, but normalization and final library fallbacks remain engine-specific | Registration defaults are not consumed and individual engine paths retain independent defaults |
| Thinking policy | Request `enable_thinking`, request/default `thinking_token_budget`, server logits processor, template kwargs, `VLLM_MLX_ENABLE_THINKING`, and model-name heuristics | Request budget overrides server budget; template thinking precedence varies by engine path | A server safety policy, template choice, environment default, and model-native behavior can alter the same outcome |
| Context/output | Server `max_tokens`, global `max_request_tokens`, request `max_tokens`, registry-wide `RegistryServeDefaults.max_tokens`, engine method defaults | Server validates request output against the global request limit; registry entries inherit one startup output default | No profile-level distinction among context, maximum output, and request admission limit; registry entries cannot override these per model |
| KV/cache | CLI max KV and cache flags, shared registry `SchedulerConfig`, paged/prefix/SSD/MLLM cache settings | Engine- and cache-specific settings apply; registry entries clone the shared startup scheduler config | Registry model entries do not expose per-model KV/prefix/paged/quantization settings; no profile records effective capacity or memory assumptions |
| Engine features | CLI flags, registry defaults, registry entry overrides, `ResolvedModelConfig`, `ModelSpec`, registration feature flags | Standalone uses CLI; registry entry overrides registry defaults when non-null | Registration features are not consumed; registry and lifecycle types overlap |
| Memory estimate | Inspection rough fit, registry explicit estimate, registry `.safetensors`/`.gguf` scan, coarse local fallback, engine memory/cache limits | Registry resolves/downloads the source first, then uses explicit estimate > recognized local weight size > one-eighth available-memory fallback when the resolved local artifact has no recognized weights | Inspection also recognizes `.bin` but is not consumed by registry; estimate source, assumptions, confidence, and measured peak are not represented together |
| Memory budget | Registry manager budget/contention, GPU utilization, engine cache budgets | Registry handles model admission; engines handle their own memory/cache limits | Independent budgets can disagree about whether a profile fits |
| Remote code policy | Standalone CLI `trust_remote_code`, model workflow conversion option, engine constructors | Direct standalone serving passes the flag into engine/model loading | Lazy/auto-unload lifecycle construction and registry entries/defaults do not carry it into created engines; conversion policy is separate from serving policy |
| Lifecycle state | `ModelManager`, `ResidencyManager`, server globals, status/model endpoints | Registry state applies to registry requests; residency state applies to standalone serving | No shared configured/resident/active state contract |
| Qualification | `qualify_model`, `bench-serve`, qualification JSON, benchmark output, registration readiness booleans | Benchmark handoff runs, but no result promotes or updates registration | Evidence is not bound to exact artifact, profile, hardware, parser/template, or feature combination |

## Manifest Pipeline

### Acquisition manifest

- Constant: `vllm_mlx_model_manifest.json` in `vllm_mlx/model_workflow.py:29`.
- Producer: `acquire_model()` at `vllm_mlx/model_workflow.py:369`.
- Content: requested model ID/revision, local path, download settings, and embedded inspection.
- Consumer: `_existing_manifests()` includes it when registration is generated at
  `vllm_mlx/model_workflow.py:528`.
- Runtime use: none found.

The top-level revision remains `options.revision` at
`vllm_mlx/model_workflow.py:435`. For a downloaded local path, the nested
inspection also receives the requested revision; an immutable resolved commit
is not required by this artifact.

### Conversion manifest

- Constant: `vllm_mlx_conversion_manifest.json` in `vllm_mlx/model_workflow.py:30`.
- Producer: `convert_model()` at `vllm_mlx/model_workflow.py:477`.
- Content: backend command, source/output paths, conversion recipe, environment,
  source/output inspection, and process result.
- Consumer: `_existing_manifests()` includes it in registration.
- Runtime use: none found.

### Registration manifest

- Constant: `vllm_mlx_registration_manifest.json` in `vllm_mlx/model_workflow.py:31`.
- Producer: `register_model()` at `vllm_mlx/model_workflow.py:547`.
- Content: model and served names, artifact path, multimodal/features, sampling
  defaults, template kwargs, parser policy, inspection, source manifests, and
  readiness booleans.
- Consumers: workflow tests; no server or registry consumer found.

The producer explicitly states that it does not mutate the runtime registry at
`vllm_mlx/model_workflow.py:550`. Registry serving instead reads the YAML schema
implemented by `load_registry_config()` at `vllm_mlx/model_registry.py:282` and
documented in `docs/guides/model-registry.md`.

### Qualification handoff

- Constant: `vllm_mlx_qualification_request.json` in `vllm_mlx/model_workflow.py:32`.
- Producer: `qualify_model()` at `vllm_mlx/model_workflow.py:630`.
- Content: model ID, server/workload/result paths, repetitions, generated
  `bench-serve` command, and subprocess result.
- Execution: `qualify_model()` directly runs the generated `bench-serve`
  subprocess command. The qualification JSON itself has no consumer and is
  emitted only when `QualificationOptions.output_path` is set.
- Result handling: `bench-serve` reads the workload and optionally writes its
  separate benchmark result; no normalized registration/profile promotion path
  was found.

Registration sets `qualification_required=true` and `production_ready=false`.
Qualification also starts with `production_ready=false`; no repository rule
changes either field from evidence.

## Runtime Configuration Flow

### Standalone serving

1. `serve_command()` in `vllm_mlx/cli.py` parses model, engine, parser, sampling, context/output, KV,
   cache, and feature flags.
2. `serve_command()` copies those values into `server.py` globals and startup inputs
   (`vllm_mlx/cli.py:32-374`).
3. When lazy loading or idle auto-unload is enabled, `load_model()` creates a
   `ModelSpec` and `ResidencyManager`. Normal standalone startup constructs the
   selected engine directly (`vllm_mlx/server.py:3222-3294`).
4. The lifecycle path maps `ModelSpec` and scheduler settings into an engine;
   the normal path maps startup arguments directly into Simple or Batched engine
   construction.
5. Request handlers combine request fields with server defaults before the
   selected engine applies templates and generation kwargs.

Approximate API-path precedence for ordinary sampling fields:

```text
request field
  > CLI/server default
  > server fallback
  > engine or model-library fallback
```

This order is not centralized or uniform across all fields and API adapters.
In particular, `enable_thinking` and chat-template kwargs are recombined in
engine-specific code: the text path can derive a default from `"coder"` in the
model name, while another SimpleEngine path reads
`VLLM_MLX_ENABLE_THINKING`. A request or server thinking-token budget may also
wrap generation with a logits processor independently of template rendering.

### Registry-backed serving

1. `load_registry_config()` reads manager policy and model entries from YAML.
2. Each entry becomes `RegisteredModel`.
3. `ModelManager._resolve_model_config()` combines supported entry overrides
   with `RegistryServeDefaults` (`vllm_mlx/model_registry.py:859-937`).
4. `ModelManager.acquire()` reserves estimated model bytes and applies wait,
   eviction, or preemption policy.
5. The engine factory builds the configured engine.
6. Request `model` selects the registry entry.

The registration JSON manifest is not part of this flow. Output admission
(`max_request_tokens`), parser selection, sampling defaults, thinking budget,
remote-code policy, and the scheduler/cache configuration are startup-wide
settings rather than complete per-model registry fields. Registry entries can
override only the fields represented by `RegisteredModel`; scheduler/cache
settings are cloned from the shared `RegistryServeDefaults.scheduler_config`.

### Standalone residency

When lazy loading or idle auto-unload is selected, `ModelSpec` and
`ResidencyManager` in `vllm_mlx/lifecycle.py` form an independent
engine-construction and state-transition contract. They do not consume
`RegisteredModel` or model workflow manifests, and `ModelSpec` does not carry
the standalone `trust_remote_code` setting into `_build_engine()`. Registry,
lifecycle-backed standalone, and direct standalone startup are separate paths
whose configuration and state types duplicate concepts without a shared profile
contract.

## Precedence Conflicts P1.2 Must Resolve

1. Registration sampling/template defaults exist but runtime uses CLI/server defaults.
2. Registration parser policy exists but runtime uses parser flags and globals.
3. Provider model ID, artifact path, registry key, lifecycle key, and served alias can diverge.
4. Requested Hugging Face revision is recorded without requiring an immutable resolved commit.
5. Inspection fit estimates are not inputs to registry memory estimates.
6. Registry model admission and engine/cache memory budgets are independent.
7. Registry and standalone lifecycle types duplicate model configuration and state.
8. Explicit `force_mllm` can diverge from `is_mllm_model()` detection, while
   explicit thinking/template policy can diverge from `"coder"` model-name and
   `VLLM_MLX_ENABLE_THINKING` defaults.
9. Server output defaults and direct multimodal method defaults differ by call path.
10. Qualification success has no typed promotion rule and is not bound to an exact profile.
11. Registry entries cannot independently express global request limits,
    parser/default sampling policy, thinking budget, remote-code policy, or the
    shared scheduler's KV/cache settings.
12. Request/template thinking controls and the token-budget logits processor
    are separate authorities without one documented precedence contract.

## Scope Boundaries And Unknowns

- Repository source cannot establish behavior internal to Hugging Face Hub,
  mlx-lm, mlx-vlm, tokenizer, or processor implementations.
- `docs/guides/model-registry.md` is the current registry YAML reference, but
  the schema is enforced procedurally rather than by a versioned schema file.
- Qualification benchmark result fields are not normalized or validated by
  `model_workflow.py`.
- Runtime modes select either standalone or registry-backed startup, but no
  common durable state/provenance model covers both.
- Target `schemas/`, `profiles/`, and `catalog/` packages do not exist yet; they
  remain P1.2 and later work.

## P1.2 Input

P1.2 should define a profile schema and precedence model only. It must not wire
the new profile into runtime behavior yet. The design must preserve current CLI
and registry behavior through an explicit compatibility mapping before P1.3
implements producers or consumers.

Independent verification result: `PASS` with no remaining findings after two
correction rounds. Verification was source-only and made no runtime, model, or
external GitHub calls.
