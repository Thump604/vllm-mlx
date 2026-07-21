# Mac-Native Product Strategy

## Product Thesis

vllm-mlx should remain the high-performance Apple Silicon inference foundation
while gaining a coherent model-intelligence and lifecycle control plane. An
official, separately distributed Mac application should consume those stable
interfaces to provide local chat, coding-client integration, model discovery,
installation, and operation without requiring users to understand server
flags, templates, parsers, or KV-cache configuration.

This direction extends the existing project. It does not require a runtime fork
or a second implementation of model serving.

## Current Position

The repository already contains substantial runtime foundations:

- OpenAI- and Anthropic-compatible APIs.
- Text and multimodal engines, streaming, tools, reasoning, and structured output.
- Continuous batching and multiple cache and speculative-serving features.
- Model inspection, acquisition, conversion, registration, and qualification commands.
- Registry-backed lazy loading, memory-budget eviction, and contention policies.
- Benchmarking, metrics, and basic Gradio chat clients.

These capabilities are not yet one product workflow. Model workflow commands
emit separate manifests, registration still requires manually supplied serving
policy, qualification does not promote a tested profile, hardware-fit estimates
are approximate, and the Gradio clients are reference API clients rather than a
desktop product.

## Target Workflow

1. Install the application.
2. Detect Apple Silicon hardware, available memory, and storage.
3. Search a curated model catalog.
4. Inspect the exact model revision and metadata.
5. Recommend an artifact, quantization, context, and serving profile.
6. Download or convert with visible progress and resumable state.
7. Validate and activate one large model.
8. Use it in chat, supported coding clients, and local OpenAI/Anthropic APIs.
9. Expose raw runtime controls only through an advanced view.

`model-onboarding-acceptance-fixture.md` applies this workflow to Laguna S 2.1.
It records the extensive source, conversion, parser, context, feature, and
qualification information an owner currently has to transfer manually. The
product succeeds when those inputs become versioned profile and lifecycle
artifacts rather than another hand-authored Runtime handoff.

## Architecture Boundaries

| Boundary | Responsibility |
|---|---|
| Inference runtime | Engines, schedulers, caches, parsing, and request execution |
| Model intelligence | Metadata ingestion, family adapters, capability detection, and fit recommendations |
| Profile system | Versioned model facts, derived settings, measured qualification, and owner overrides |
| Lifecycle control plane | Download, conversion, validation, activation, supervision, recovery, and status |
| Chat harness | Conversations, attachments, history, and profile selection |
| Coding integration | Stable OpenAI/Anthropic endpoints and setup for existing coding clients |
| Desktop application | Installation, catalog, lifecycle UX, chat, coding setup, and diagnostics |

The Python package should own the runtime, schemas, model intelligence, profile
resolution, and control APIs. The Mac application should be an official sibling
distribution backed by those APIs, not code embedded in the Python wheel.

## Repository-Level Implementation Map

```text
vllm_mlx/model_intelligence/  metadata readers, family adapters, fit estimator
vllm_mlx/profiles/            typed profile schema, resolution, validation
vllm_mlx/catalog/             curated model and artifact definitions
vllm_mlx/control/             lifecycle service and stable control API
schemas/                      versioned manifest and profile schemas
catalog/models/               reviewed model facts
catalog/profiles/             known-good serving profiles
tests/product_workflows/      install-to-chat and install-to-code contracts
```

Reuse `model_workflow.py`, `model_registry.py`, `lifecycle.py`, existing API
models, parsers, engines, and benchmark tooling. Evolve their contracts rather
than creating another registry or manifest family.

## Profile Truth Model

Profiles must distinguish four sources:

1. Provider facts from an exact repository revision and hashed files.
2. Derived recommendations from explicit family adapters and hardware rules.
3. Measured qualification with reproducible evidence.
4. User overrides that never silently become shared defaults.

A known-good profile records model and artifact identity, tokenizer and template
hashes, supported capabilities, parsers, sampling defaults, context/output/KV
limits, engine features, hardware envelope, qualification status, evidence, and
schema version. Reviewed catalog data belongs in version control. Downloads,
active state, local measurements, and user overrides belong in macOS Application
Support state rather than transient run directories.

## Smallest Coherent Release

- A managed macOS installation and background runtime.
- Hardware and memory detection.
- A curated catalog of a small number of proven chat and coding models.
- Download, progress, storage management, and removal.
- Explainable fit and serving-profile recommendations.
- One active large model at a time.
- Chat history and supported attachments.
- One-click OpenAI and Anthropic endpoint setup for coding clients.
- Health, active-model, memory, log, stop, and recovery controls.
- Advanced controls separated from the normal workflow.

Defer audio and reranking product UX, arbitrary Hugging Face compatibility,
simultaneous large-model residency, a new coding-agent engine, and exhaustive
feature combinations until the core workflow is reliable.

## Staged Roadmap

1. Consolidate profile schemas, configuration precedence, hardware inventory, and manifest migration.
2. Build metadata readers, family adapters, fit estimation, and a curated catalog.
3. Expose lifecycle operations through a stable control API with recovery and one-active-model semantics.
4. Ship the desktop shell with catalog, activation, chat, coding-client setup, and diagnostics.
5. Automate bounded qualification and evidence-backed catalog updates.
6. Expand multimodal UX, model families, and performance profiles deliberately.

Until stages 1-4 exist, avoid adding speculative backends, cache variants,
isolated model-family workarounds, or new benchmark frameworks unless they
directly unblock the first-release catalog.

Upstream delivery follows `upstream-delivery-plan.md`. The implementation
branch is intentionally broader than any proposed review; Wayner and Jan should
receive small dependency-ordered PRs with one contract or behavior each.
