# Upstream Delivery Plan

## Purpose

This plan turns the product-strategy work into reviewable upstream increments.
Wayner and Jan should be able to review one contract or one pure behavior at a
time, with a small diff, focused tests, and a clear statement of what is not
included. The current `604/product-agent-strategy` branch is an integration
branch for local development; it is not itself the proposed upstream PR.

The delivery unit is a dependency-ordered topic branch based on the current
upstream base. Each PR must pass its own focused checks and must not require the
reviewer to reconstruct later product stages from a large mixed diff.

## Review Principles

- Keep governance and agent workflow changes separate from product contracts.
- Keep schema changes separate from migration and runtime integration.
- Keep pure validation and resolution logic separate from lifecycle behavior.
- Keep model-family fixtures separate from generic profile behavior.
- Keep pre-commit configuration repair in its own PR.
- Do not include model qualification, live serving changes, Jobs/Ops policy, or
  desktop UI in the foundational contract PRs.
- Every PR names its exact dependency, files, tests, and excluded work.
- A PR may be merged only when its tests prove the behavior claimed by that PR;
  later evidence cannot retroactively enlarge an earlier claim.

## Dependency-Ordered PR Sequence

### PR 1: ModelProfile schema and provenance contract

**Depends on:** none.

**Upstream:** [waybarrios/vllm-mlx#645](https://github.com/waybarrios/vllm-mlx/pull/645)
is open, mergeable, CI-green, assigned to Jan, and awaiting review.

**Scope:** Add the versioned profile schema, provenance vocabulary, example,
and concise contract documentation. Define provider facts, derived
recommendations, measured qualification, and user overrides without wiring a
server or lifecycle path.

**Files:**

- `schemas/model-profile-v1.schema.json`
- `schemas/examples/model-profile-v1.example.json`
- `docs/development/model-profile-v1.md`
- `tests/test_model_profile_schema.py`

**Acceptance checks:** Schema validation succeeds for the example; required
identity, artifact, capability, serving, hardware, qualification, evidence,
and provenance fields are covered; invalid source/status combinations fail.

**Excluded:** Python adapters, migration, serving resolution, registry changes,
model downloads, qualification runs, Jobs/Ops, desktop UI, and governance files.

### PR 2: Pure ModelProfile validation

**Depends on:** PR 1.

**Prepared branch:** `604/model-profile-validation-v1` at `397554a`,
rebased onto the exact PR 1 head. Focused schema and semantic validation tests
pass (`24 passed`). Do not open this PR until PR 1 is accepted.

**Scope:** Implement deterministic semantic validation for profile documents,
including canonical identity binding, provenance-source invariants, evidence
references, limits, request policy, and feature controls.

**Files:**

- `vllm_mlx/model_profile.py`
- `tests/test_model_profile.py`
- `pyproject.toml` only for the narrowly required canonicalization/validation
  dependencies

**Acceptance checks:** Focused unit tests cover valid profiles, invalid
date-times, canonical subject mismatches, oversized numeric values, invalid
feature states, invalid request fields, stale evidence references, and
maintainer-only attribution of provider facts. Full repository tests are not a
substitute for the focused contract suite.

**Excluded:** Legacy manifest mapping, runtime configuration precedence,
lifecycle operations, model-family code, live qualification, and pre-commit
repair.

### PR 2.5: Legacy import-result envelope

**Depends on:** PR 1 and PR 2.

**Prepared branch:** `604/model-profile-import-envelope-v1` at `273ce09`,
stacked on the prepared PR 2 branch. The focused schema, semantic-validation,
and import-envelope suites pass together (`28 passed`); the complete upstream
suite also passes (`2277 passed, 24 skipped, 23 deselected`). Do not open it
before PR 2 is accepted.

**Scope:** Define the only compatibility envelope allowed to carry an
incomplete ModelProfile fragment. Validate complete results against both the
envelope schema and ModelProfile semantic contract without importing serving
code.

**Files:**

- `schemas/model-profile-import-result-v1.schema.json`
- `vllm_mlx/model_profile_import.py`
- `tests/test_model_profile_import_result.py`
- `docs/development/model-profile-v1.md` for current-state wording
- `pyproject.toml` for the directly imported validation dependency

**Acceptance checks:** Incomplete results require explicit issues; complete
results reject error-severity issues and invalid profiles; semantic failures
retain `/profile` pointers; the raising API preserves collected issues.

**Excluded:** Legacy source mapping, configuration resolution, serving
activation, qualification promotion, and lifecycle behavior.

### PR 3A: Legacy artifact and identity import

**Depends on:** PR 2.5.

**Prepared branch:** `604/model-profile-import-core-v1`, stacked on the
prepared import-envelope branch at `da89453`. The focused contract suites pass
(`38 passed`), the complete upstream suite passes (`2287 passed, 24 skipped,
23 deselected`), touched slop and claim gates are clean, and the final blind
Desloppify review reports objective/verified `100.0` and strict `96.5`. Do not
open it until PR 2.5 is accepted.

**Scope:** Map acquisition, conversion, and registration records into an
incomplete profile fragment without pretending that missing facts are known.
Keep source validation, provenance, conflict tracking, and input immutability
inside this pure adapter slice.

**Acceptance checks:** Existing inputs remain unchanged; missing required facts
remain explicit issues; failed conversions cannot contribute artifact claims;
registration feature flags do not silently become complete feature policy.

**Excluded:** Registry and CLI/server mapping, qualification evidence,
explicit completion, registry mutation, serving activation, and product UI.

### PR 3B: Legacy registry and serving import

**Depends on:** PR 3A.

**Prepared branch:** `604/model-profile-import-serving-v1`, stacked on the
prepared PR 3A branch at `8494643`. Focused profile suites pass (`37 passed`);
Ruff, Black, focused mypy, touched slop, claim lint, and diff checks are clean.
The final blind Desloppify review reports overall `90.0`, strict `89.9`, and no
open T1/T2 or review findings. A complete-suite run under concurrent review
load finished with `2289 passed, 24 skipped, 23 deselected` and one unrelated
continuous-batching throughput-threshold failure (`41.5 tok/s` versus the
test's fixed `100 tok/s` threshold); do not represent that run as a clean full
suite. Do not open PR 3B before PR 3A is accepted.

**Scope:** Add registry and CLI/server source mapping, shared serving
normalization, boolean feature translation, and unknown-policy reporting.

**Acceptance checks:** Registry identity remains distinct from served aliases;
invalid booleans and same-source conflicts cannot silently change engine or
feature state; unsupported policy shapes produce stable issues; malformed
nested values and direct limits remain diagnostics rather than empty or
ill-typed established facts; nested registry-only fields cannot disappear
without a target-contract diagnostic.

**Excluded:** Qualification evidence, explicit completion, live activation,
and product UI.

### PR 3C: Legacy qualification import

**Depends on:** PR 3B.

**Prepared branch:** `604/model-profile-import-qualification-v1`, stacked on
the prepared PR 3B branch at `dc9e33e`. Focused profile suites pass
(`38 passed`); Ruff, Black, focused mypy, touched slop, claim lint, and diff
checks are clean. Direct review added regression coverage for mixed
valid/invalid evidence history and strict RFC 3339 timestamps. The external
Desloppify subjective-review runner exhausted its account quota in all 20
batches, so no subjective score is claimed for this slice. Do not open PR 3C
before PR 3B is accepted.

**Scope:** Add qualification-source and evidence normalization without
promoting weak or unbound booleans into qualification truth.

**Acceptance checks:** Weak signals remain issues; passing evidence becomes
eligible only when bound to the imported identity and artifact; the result
remains incomplete until the separate finalization boundary runs.

**Excluded:** Explicit completion, automatic promotion, live qualification,
and product UI.

### PR 3D: Explicit compatibility finalization

**Depends on:** PR 3C.

**Prepared branch:** `604/model-profile-import-finalization-v1`, stacked on
the prepared PR 3C branch at `633217d`. Focused profile suites pass
(`46 passed`); Ruff, Black, focused mypy, touched slop, claim lint, and diff
checks are clean. A bounded Desloppify mechanical scan reports objective and
verified `81.5`; its overall/strict `20.4` values are not decision-grade
because all subjective dimensions are unassessed after runner quota exhaustion.
Do not open PR 3D before PR 3C is accepted.

**Scope:** Add the pure finalization path that accepts explicitly supplied
missing facts, rejects changed imported facts and source errors, revalidates
the complete profile, and returns `complete=true` only after validation passes.

**Files:**

- `vllm_mlx/model_profile_compat.py`
- `vllm_mlx/_model_profile_compat_types.py`
- `vllm_mlx/_model_profile_finalization.py`
- `tests/test_model_profile_compat.py`
- `tests/test_model_profile_finalization_compat.py`
- `docs/development/model-profile-compatibility-mapping.md`

**Acceptance checks:** Conflict resolution is fail-closed; completed profiles
pass the envelope, schema, and semantic validators; no finalizer can upgrade an
incomplete fragment without the required facts and evidence.

**Excluded:** Registry mutation, download/conversion execution, serving
activation, automatic qualification promotion, and product UI.

### PR 4: Effective configuration precedence resolver

**Depends on:** PR 2.

**Prepared branch:** `604/model-profile-resolution-v1` at `a4b2cd1`,
stacked on the prepared PR 2 branch. The schema, semantic validation, and
precedence suites pass together (`30 passed`). Do not open it before PR 2 is
accepted.

**Scope:** Add a pure resolver that makes precedence executable and auditable:
profile limits, allowed activation overrides, profile defaults, provider
defaults, then runtime fallbacks. Immutable identity/template/parser/engine
fields and profile limits must not be silently overridden.

**Files:**

- `vllm_mlx/model_profile_resolution.py`
- `tests/test_model_profile_resolution.py`
- `docs/development/model-profile-v1.md` only if the precedence contract needs
  a wording correction

**Acceptance checks:** A precedence matrix covers each supported setting,
source attribution is returned with the effective value, disallowed overrides
fail, and runtime fallback fills only genuinely absent optional values.

**Excluded:** Wiring the resolver into a live route, changing model defaults,
client behavior, feature qualification, or resident/default policy.

### PR 5: Laguna S 2.1 onboarding fixture

**Depends on:** PR 1, PR 2, and PR 3D.

**Prepared branch:** `604/laguna-onboarding-fixture-v1` at `5a00380`,
stacked on the prepared PR 3D branch. The fixture is structural-only: its
source revision and hashes are asserted locally, the upstream-claim lint and
diff check pass, and it contains no local artifact path, conversion, serving,
or qualification claim. Do not open it before PR 1, PR 2, and PR 3D are
accepted.

**Scope:** Add the Laguna acceptance fixture and immutable artifact metadata as
test data/documentation. Demonstrate the boundary between structural artifact
facts and later runtime qualification.

**Files:**

- `docs/development/model-onboarding-acceptance-fixture.md`
- `tests/fixtures/model_profiles/laguna-s-2.1.json` only if a fixture is
  needed by the test suite

The upstream fixture uses repository URIs, immutable revisions, and content
hashes. Local `/Volumes/Lexar/...` evidence locations remain in the local
acceptance record and must not be published as portable profile fields.

**Acceptance checks:** The fixture validates as an artifact-only profile; its
  exact revision, artifact checksums, layer/expert layout, and template facts
  remain bound; it cannot be reported as qualified or exposed to Jobs by
  structural evidence alone.

**Excluded:** Downloading the 118B artifact in CI, model loading, generation,
  parser implementation, live serving, MTP/DFlash/SpecPrefill, and production
  exposure.

## Laguna Operational Serving PR Sequence

The operational implementation is intentionally separate from the portable
profile roadmap above. Do not combine these patches into one review.

### Laguna Runtime PR A: mlx-vlm baseline integration correctness

**Repository:** `Blaizzy/mlx-vlm`

**Scope:** Preserve Laguna special-token and string-message behavior required by
the provider chat template, and load the processor regex without changing other
model families.

**Acceptance checks:** Focused tokenizer/template, plain message, tool-call, and
post-tool continuation tests fail before the fix and pass after it.

**Excluded:** DFlash, vllm-mlx server wiring, Runtime registry data, local model
paths, qualification claims, and broad model-loader cleanup.

### Laguna Runtime PR B: mlx-vlm generic DFlash backend

**Repository:** `Blaizzy/mlx-vlm`

**Depends on:** Laguna Runtime PR A.

**Prepared branch:** `604/laguna-dflash-backend` at `4575acd0`, stacked on
Laguna Runtime PR A. Focused config/lifecycle, speculative, and server tests
pass (`20`, `19`, and `5` relevant tests). Do not open this PR until PR A is
accepted.

**Scope:** Add the Laguna DFlash drafter, exact checkpoint/target compatibility
validation, request-local hidden-state capture and teardown, speculative cache
rollback, converter support for processor-less DFlash checkpoints, and focused
metadata/cleanup tests.

**Acceptance checks:** Config and weight mismatch rejection, disabled-path
no-op, success/error/cancel cleanup, cache rollback, Q4 draft loading, and the
documented mapping from seven speculative tokens to verify block size eight.

**Excluded:** vllm-mlx CLI/server changes, local Runtime contracts, Jobs/Ops
policy, Gemma or Qwen speculative changes, and long-horizon quality claims.

### Laguna Runtime PR C: vllm-mlx request-local Poolside v1 parsers

**Repository:** `waybarrios/vllm-mlx`

**Scope:** Add provider-compatible `poolside_v1` reasoning and tool parsers,
construct their incremental state per request, and cover direct, tool, and
streaming parsing.

**Acceptance checks:** Direct reasoning extraction, structured tool calls,
fragmented arguments, post-tool continuation, and concurrent request-local
state.

**Excluded:** Laguna loading, DFlash, local Runtime contracts, model downloads,
and model-quality claims.

### Laguna Runtime PR D: vllm-mlx generic MLLM draft wiring

**Repository:** `waybarrios/vllm-mlx`

**Depends on:** Laguna Runtime PR B.

**Prepared branch:** `604/mllm-generic-draft-wiring` at `df7aef4`. Focused
drafter-option, MLLM adapter, and watchdog tests pass (`19 passed`). Do not open
this PR until the mlx-vlm DFlash API is accepted and its final surface is known.

**Scope:** Generalize MLLM draft loading for `dflash`, `eagle3`, and `mtp`; pass
through trust-remote-code; expose method-neutral speculative metadata; and add
an explicit default-off `--default-mllm-draft` option with per-request opt-out.

**Acceptance checks:** Exact draft-kind validation, disabled default behavior,
explicit enable/disable behavior, metadata counters, and focused MLLM server
tests.

**Excluded:** Laguna-specific model paths, Runtime registry rows, resident or
Jobs defaults, continuous batching, unrelated speculative backends, and product
quality claims.

### Laguna Runtime PR E: vllm-mlx Laguna OpenAI message support

**Repository:** `waybarrios/vllm-mlx`

**Depends on:** Laguna Runtime PR C.

**Prepared branch:** `604/laguna-openai-message-path` at `9f790eb`, stacked on
PR C. Focused message-ordering, parser-selection, and server tests pass
(`13 passed`). Do not open this PR until PR C is accepted.

**Scope:** Preserve Laguna string messages through OpenAI normalization and make
the existing `poolside_v1` parser selectable from the CLI.

**Acceptance checks:** Direct chat, structured tool call, and streamed post-tool
continuation retain user content and produce clean final assistant content.

**Excluded:** Generic DFlash engine work, local Runtime contracts, model
downloads/conversion, and broad message-normalization refactors.

## Model Intelligence PR Sequence

These topic branches may proceed after the profile contract exists. They stay
separate from lifecycle and serving work so reviewers can evaluate pure
inspection and estimation logic without accepting the product roadmap.

### MI-1: Evidence-backed model metadata inspection

**Depends on:** PR 1 and PR 2.

**Prepared branch:** `604/model-metadata-inspection-v1` at `3b9b61f`,
stacked on the prepared PR 2 branch. It contains only `model_workflow.py` and
its focused tests; all 23 tests pass. Do not open it before PR 2 is accepted.

**Scope:** Extend artifact inspection to tokenizer, chat-template, generation,
license, immutable revision, and declared-capability metadata. Preserve source
attribution and return unknown rather than inferring unsupported capabilities.

**Files:** `vllm_mlx/model_workflow.py` and its focused tests only.

**Acceptance checks:** Local and repository-backed fixtures prove revision,
digest, tokenizer/template, and generation metadata attribution; absent or
ambiguous metadata stays unknown.

**Excluded:** Download execution, conversion, model loading, family-specific
fit estimates, serving registration, and live qualification.

### MI-2: Apple Silicon hardware inventory

**Depends on:** none.

**Prepared branch:** `604/apple-silicon-hardware-inventory` at `6e9efba`.
The branch contains only the inventory module and focused tests; all seven
tests, including the native macOS smoke, pass. Hold it while PRs 1 and C occupy
the active vllm-mlx review slots.

**Scope:** Add a read-only, source-attributed inventory of the Apple Silicon
facts required by later fit calculations. Return an explicit privacy-safe
allowlist rather than serial numbers, UUIDs, or unrelated host metadata.

**Files:** `vllm_mlx/hardware.py` and `tests/test_hardware.py`.

**Acceptance checks:** Parser fixtures and a native smoke prove memory, chip,
GPU-core, and source attribution; privacy-sensitive identifiers never appear.

**Excluded:** Recommendation policy, model selection, process control, and
serving behavior.

### MI-3: Explainable model-fit calculations

**Depends on:** MI-2.

**Prepared branch:** `604/model-fit-estimation-v1` at `8711de7`, stacked on
the prepared MI-2 branch. The hardware and fit suites pass together
(`44 passed`). Hold it until MI-2 is accepted.

**Scope:** Add pure calculations for exact artifact residency, dense/GQA KV
cache, independent context bounds, conversion workspace, and memory margin.
Every input is explicit and every result separates exact, derived, assumption,
measured, and unknown values.

**Files:** `vllm_mlx/model_fit.py` and `tests/test_model_fit.py`.

**Acceptance checks:** Focused tests cover exact sums, boundary values,
measured overrides, missing evidence, unsupported architectures, and hardware
provenance. Hybrid and mixed-attention caches remain unknown.

**Excluded:** Family recognition, model loading, policy-selected defaults, and
live memory qualification.

### MI-4: Dense/GQA config adapter

**Depends on:** MI-3.

**Prepared branch:** `604/dense-gqa-adapter-v1` at `444aebb`, stacked on
MI-3. It introduces the shared adapter result/provenance contracts, dispatcher,
dense/GQA adapter, one synthetic fixture, and focused tests only. The stacked
hardware, fit, and adapter suites pass (`59 passed`).

**Scope:** Adapt declared dense or GQA config structure into the generic
estimator. Cache dtype, context policy, concurrency, and quantization overhead
remain explicit caller/profile inputs with provenance.

**Files:** The dense/GQA slice of `vllm_mlx/model_family_adapters.py`, its
synthetic fixture, and focused tests.

**Acceptance checks:** MHA versus GQA classification follows declared head
counts; fake provider fields are rejected; explicit profile inputs cannot be
invented or silently ignored.

**Excluded:** Qwen hybrid attention, Laguna mixed attention, and serving
configuration.

### MI-5: Qwen 3.6 hybrid config adapter

**Depends on:** MI-1, MI-3, and MI-4.

**Prepared branch:** `604/qwen-hybrid-adapter-v1` at `c075ee9`, stacked on
MI-4. It adds only the Qwen dispatcher branch, hybrid adapter, pinned fixture,
and Qwen-focused tests. The stacked suites pass (`66 passed`).

**Scope:** Add a revision-pinned adapter for the declared Qwen 3.6 hybrid
attention structure. Validate the exact full-attention cadence and expose
linear-state, MoE, context, and MTP config facts without importing local overlay
fields into the provider contract.

**Files:** The Qwen slice of `vllm_mlx/model_family_adapters.py`, its pinned
fixture, and focused tests.

**Acceptance checks:** The provider config SHA and revision are recorded;
malformed layer schedules fail unknown; generic KV remains unknown.

**Excluded:** Qwen serving templates, parser behavior, generation, MTP
execution, and architecture-specific cache formulas.

### MI-6: Laguna S 2.1 config adapter

**Depends on:** PR 5, MI-1, MI-3, and MI-4. It is stacked after MI-5 to avoid
parallel edits to the shared dispatcher, although Laguna behavior does not
depend on Qwen behavior.

**Prepared branch:** `604/laguna-config-adapter-v1` at `a380a8f`, stacked on
MI-5. It adds only the Laguna dispatcher branch, exact-structure adapter,
pinned fixture, and Laguna-focused tests. The stacked suites pass
(`84 passed`).

**Scope:** Add a revision-pinned adapter for Laguna S 2.1 structural identity:
mixed global/sliding attention, per-layer heads, MoE routing, shared expert,
per-head gating, and first-layer dense MLP.

**Files:** The Laguna slice of `vllm_mlx/model_family_adapters.py`, its pinned
fixture, and focused tests.

**Acceptance checks:** Every identity-bearing array and exact numeric field is
validated; adversarial variants fail unknown; no local artifact path is needed;
mixed-attention KV remains unknown.

**Excluded:** Full model load, logits, parser/tool support, generation,
qualification, registration, and exposure.

### MI-7: Recorded fixture recommendation validation

**Depends on:** MI-2 through MI-6.

**Prepared branch:** `604/model-fit-fixture-validation-v1` at `1246cf6`,
stacked on the prepared Laguna config-adapter branch. The complete focused
model-intelligence group passes (`91 passed`); Ruff, Black, and
`git diff --check` are clean. This is portable fixture validation only and
does not load a model or access local artifact paths.

**Scope:** Exercise the adapters and estimators together against portable,
source-attributed fixtures. Use the Laguna handoff to prove that a future
onboarding workflow can replace manual fact transcription without upgrading
artifact evidence into runtime qualification.

**Files:** A focused fixture-validation test module and portable fixture data
only.

**Acceptance checks:** Dense/GQA estimates are reproducible; hardware margins
retain provenance; Qwen and Laguna unsupported cache estimates remain unknown;
fixture revision or digest drift fails closed.

**Excluded:** Local `/Volumes` access in CI, model loads, service changes,
performance claims, and exposure policy.

### PR 6A: Targeted resumable acquisition

**Depends on:** PR 2 and MI-1.

**Prepared branch:** `604/model-acquisition-resume-v1` at `08c5ce7`, stacked
on the prepared metadata-inspection branch. The focused workflow suite passes
(`33 passed`); Ruff, Black, and `git diff --check` are clean. Do not open it
until its contract dependencies are accepted.

**Scope:** Make targeted Hugging Face acquisition identity-bound, resumable,
and crash-durable. An immutable revision, operation journal, staging marker,
atomic manifest publication, retry behavior, and target-path conflict handling
remain inside this single workflow slice.

**Files:**

- `vllm_mlx/model_workflow.py`
- `tests/test_model_workflow.py`

**Acceptance checks:** An interrupted acquisition resumes only when its exact
identity matches; a partial target cannot be mistaken for a complete artifact;
the published manifest is durable and bound to the operation identity; unrelated
target paths fail without modification.

**Excluded:** Conversion, registration, ModelProfile completion, lifecycle
activation, model loading, qualification, catalog curation, desktop UX, and
broad workflow refactoring.

### PR 6B: Recoverable conversion and output validation

**Depends on:** PR 6A.

**Scope:** Add identity-bound conversion journaling, cancellation/retry
handling, output validation, and atomic conversion-manifest publication. Keep
conversion mechanics separate from acquisition and profile registration.

**Acceptance checks:** A conversion cannot overwrite an unrelated output;
cancelled and failed conversions retain actionable state; successful output has
the expected MLX structure and content binding before it is registered.

**Excluded:** Acquisition changes, serving startup, model qualification,
catalog curation, desktop UX, and lifecycle activation.

### PR 6C: Workflow-to-profile evidence integration

**Depends on:** PR 3D, PR 5, and PR 6B.

**Scope:** Emit profile-bound acquisition and conversion evidence through the
explicit import/finalization boundary. This is the only workflow slice allowed
to make a complete profile from the independently validated operation records.

**Acceptance checks:** Exact revision, conversion parameters, artifact digest,
tool versions, and checksum results retain provenance; incomplete workflow
facts stay incomplete rather than becoming inferred profile truth.

**Excluded:** Serving startup, automatic qualification promotion, catalog
curation, desktop UX, and broad workflow refactoring.

### LC-1: Pure lifecycle state contract

**Depends on:** PR 4 and MI-7.

**Prepared branch:** `604/lifecycle-contract-v1` at `6dd6a90`, stacked on
the prepared precedence-resolver branch. The focused lifecycle-contract suite
passes (`40 passed`); Ruff, Black, and `git diff --check` are clean. Do not
open it until its profile and model-intelligence dependencies are accepted.

**Scope:** Define immutable configured-profile, resolved-process,
resident-process, and request-lease state plus legal transitions. Keep the
contract synchronous and dependency-free so existing single-resident and
registry managers can adopt it incrementally.

**Files:** `vllm_mlx/lifecycle_contract.py` and
`tests/test_lifecycle_contract.py`.

**Acceptance checks:** Legal configure/resolve/load/acquire/release/unload/clear
chains pass; illegal transitions, lease underflow, identity mismatch, and
configuration changes while live fail with stable actionable errors.

**Excluded:** Changes to `lifecycle.py`, `model_registry.py`, `server.py`, CLI,
HTTP status, model processes, Jobs/Ops, or live state.

### PR 7: Lifecycle control API and one-active-model state

**Depends on:** PR 4, PR 6, and LC-1.

**Prepared branch:** `604/lifecycle-control-persistence-v1` at `5b2ccb0`,
stacked on the pure lifecycle-contract branch. The focused lifecycle-control
and CLI suites pass (`28 passed`) on the supported Python 3.12 runtime; Ruff,
Black, and `git diff --check` are clean. This branch contains persistence and
single-model state only. Product HTTP transport and qualification normalization
remain separate downstream slices.

**Scope:** Expose acquire, validate, activate, stop, status, and recovery
  operations through the existing lifecycle/registry boundary. Make configured
  state, process state, and profile state explicit and recoverable.

**Files:** Existing `lifecycle.py`, `model_registry.py`, API models/routes,
  focused lifecycle tests, and state schemas identified by the inventory.

**Acceptance checks:** A model can be activated and stopped through one stable
  interface; stale state is detected; one large active model is enforced by
  lifecycle state; cancellation and recovery tests pass.

**Excluded:** Desktop application, Jobs/Ops integration, new inference engines,
  arbitrary model auto-configuration, and performance feature matrices.

### PR 8: Qualification evidence normalization

**Depends on:** PR 7.

**Prepared branch:** `604/profile-qualification-evidence-v1` at `8b6a1d0`,
stacked on the recovery/validation workflow branch. The focused workflow suite
passes (`63 passed`); Ruff, Black, and `git diff --check` are clean. The code
binds evidence but does not promote a profile automatically; wiring it through
the final lifecycle activation surface remains dependent on PR 7.

**Scope:** Convert bounded load, generation, parser, memory, and recovery
  results into profile evidence without automatically promoting a model to
  qualified or production-ready.

**Files:** Qualification result models, evidence schemas, harness adapters, and
  focused tests.

**Acceptance checks:** Raw output, parsed output, serving safety, memory, and
  workload results remain separate; incomplete or contaminated runs cannot
  promote a profile; evidence is tied to exact model/config/artifact identity.

**Excluded:** Exhaustive model/feature combinations, Jobs production policy,
  default/resident selection, and desktop UX.

### PW-1: Versioned product control contract

**Depends on:** PR 7 and PR 8.

**Prepared branch:** `604/control-api-contract-v1` at `956779f`, stacked on
the prepared profile-validation branch because it uses the same canonical JSON
dependency. The focused control-API suite passes (`16 passed`); Ruff, Black,
and `git diff --check` are clean. The contract can be reviewed independently,
but must not be wired to a mutable lifecycle route until PR 7 and PR 8 are
accepted.

**Scope:** Add only the versioned request/response envelope, operation models,
and pure validation helpers used by future product clients.

**Files:** `vllm_mlx/control_api.py`, `tests/test_control_api.py`, and the
versioned control schema/documentation.

**Acceptance checks:** Valid envelopes round-trip; incompatible API versions,
malformed profile references, and idempotency conflicts fail deterministically.

**Excluded:** HTTP routes, server wiring, lifecycle mutation, model loading,
catalog data, and client UX.

### PW-2: Catalog loader and portable first-release profiles

**Depends on:** PR 1, PR 2, PR 5, and MI-7.

**Prepared branch:** `604/model-profile-catalog-v1` at `1564823`, stacked on
the prepared profile-validation branch. The focused catalog-loader suite passes
(`11 passed`); Ruff, Black, and `git diff --check` are clean. The read-only
loader is separate from first-release profile data.

**Prepared data branch:** `604/first-release-profile-data-v1` at `00da32a`,
stacked on the catalog-loader branch. The profile/catalog suites pass
(`17 passed`); Ruff, Black, and `git diff --check` are clean. It contains
portable Qwen and Laguna structural profile data and hardware envelopes only;
it does not load a model or claim portable runtime qualification.

**Scope:** Add the read-only validated catalog loader plus portable Qwen and
Laguna first-release profile documents. Laguna remains artifact-only until its
separate qualification evidence exists.

**Files:** `vllm_mlx/catalog/**`, `catalog/profiles/**`,
`catalog/hardware/**`, and focused catalog/profile tests.

**Acceptance checks:** Catalog order and identity are deterministic; duplicate
or invalid profiles fail; qualification state controls normal visibility; no
local absolute artifact path is published.

**Excluded:** Download, conversion, activation, live qualification, HTTP
routes, and model-family serving patches.

### PW-3: Durable product operations service

**Depends on:** LC-1, PR 7, and PW-1.

**Scope:** Add durable install/activate/stop/remove operation records,
idempotency, cancellation, sanitized failures, and the service protocol over
the existing lifecycle owner.

**Files:** `vllm_mlx/control/service.py`, the narrow product-operation additions
to lifecycle state, and focused operation/service tests.

**Acceptance checks:** Operation replay survives restart; conflicting keys fail;
pre-start cancellation reaches a durable terminal state; failures do not leak
local paths or tracebacks.

**Excluded:** Artifact acquisition, engine construction, server globals, HTTP
routes, profile fixtures, and live model calls.

### PW-4: Managed artifact and residency adapter

**Depends on:** PR 6, PR 7, PW-2, and PW-3.

**Scope:** Bind exact profile hashes to local or managed artifacts and adapt one
validated profile to the existing single-resident lifecycle manager. Preserve
the prior model on failed replacement and clear dormant state on removal.

**Files:** `vllm_mlx/control/runtime.py` and
`tests/test_product_runtime.py`.

**Acceptance checks:** Content mismatch fails before load; conversion artifacts
require an explicit binding; failed activation restores prior state; removing a
stopped managed artifact clears restart configuration.

**Excluded:** HTTP routes, server startup, catalog authoring, model
qualification, and desktop/client behavior.

### PW-5: Product control HTTP routes

**Depends on:** PW-1 and PW-3.

**Scope:** Expose catalog, operation, status, diagnostics, activation, stop,
remove, and cancellation through one versioned authenticated FastAPI router.

**Files:** `vllm_mlx/control/routes.py`, route registration only, and focused
route/transport tests.

**Acceptance checks:** Authentication and all errors preserve the versioned
envelope; route parameters participate in idempotency; protected diagnostics
remain protected.

**Excluded:** Runtime globals, engine behavior, model loading, catalog fixtures,
and client setup.

### PW-6: Managed product server integration

**Depends on:** PW-2 through PW-5.

**Scope:** Wire the validated catalog, lifecycle manager, runtime adapter, and
HTTP service into the existing server behind explicit product-control flags.
Restore only exact, still-qualified persisted profiles.

**Files:** Narrow `vllm_mlx/server.py` helpers, CLI argument definitions, and
focused product-server/restart tests.

**Acceptance checks:** Startup without a configured model stays unloaded;
restart restores exact qualified identity; downgraded or changed profiles fail
before taking the lifecycle lock; ordinary serving remains unchanged when the
feature is not configured.

**Excluded:** New inference routes, model-specific patches, qualification runs,
Ops/Jobs policy, and desktop UI.

### PW-7A: Product command shell and control client

**Depends on:** PW-1 and PW-5.

**Prepared branch:** `604/product-command-shell-v1` at `f1865ea`, stacked on
the versioned control-contract branch. The focused client/CLI suites pass
(`11 passed`); Ruff, Black, and `git diff --check` are clean. This branch does
not provide routes or activate a model.

**Scope:** Add the thin versioned control client and command shell around the
stable API.

**Files:** `vllm_mlx/control_client.py`, `vllm_mlx/product_cli.py`, CLI entrypoint
wiring, client/CLI tests, and concise CLI reference documentation.

**Acceptance checks:** Client version compatibility, command request shaping,
idempotency forwarding, and server-error rendering are deterministic without
raw serving flags or direct lifecycle mutation.

**Excluded:** HTTP route implementation, lifecycle mutation, model loading,
catalog fixtures, golden workflows, and a first-party coding-agent engine.

### PW-7B: Golden install-to-chat and install-to-code workflows

**Depends on:** PW-2, PW-5, PW-6, and PW-7A.

**Scope:** Add golden install-to-chat and install-to-code workflows around the
stable API and catalog-backed selected profile.

**Files:** `tests/product_workflows/**` and the narrow workflow helpers needed
to exercise the already-supported command shell.

**Acceptance checks:** A user can select a curated model, receive an
  explainable profile, activate it, chat, configure a supported coding client,
  and recover from interruption without supplying raw serving flags.

**Excluded:** New first-party coding-agent engine, arbitrary Hugging Face
  compatibility, simultaneous large-model residency, audio, reranking, and
  unrelated cleanup.

## Separate Non-Product PRs

### Governance/config tooling PR

Keep `AGENTS.md`, `.codex/**`, agent execution configuration, and governance
documentation separate from ModelProfile schemas and lifecycle code. This lets
Wayner/Jan accept or decline repository-agent policy without reviewing product
behavior in the same diff. Its checks are documentation placement, script
syntax, and governance smoke tests only.

### Pre-commit repair PR

Keep `.pre-commit-config.yaml` changes in a separate PR. That PR should contain
only hook-version/argument corrections, a reproducible failing-before record,
and the passing focused gate afterward. It must not carry ModelProfile,
Migration, Laguna, lifecycle, or product-strategy changes. Dependency-lock
policy, if needed, belongs with the repository's packaging policy rather than
being hidden inside a product contract PR.

### Performance-test classification PR

Keep model-loading throughput benchmarks out of the default correctness suite
by applying the repository's existing `slow` marker. Preserve the benchmark
logic and run it explicitly with `pytest -m slow`; do not replace measured
throughput with a mocked correctness test or weaken the recorded thresholds in
this classification-only PR. This patch must remain separate from model-fit,
serving, scheduler, and engine behavior changes.

## Reviewer Handoff Format

Every proposed PR should include:

1. The one sentence problem and the one sentence behavior change.
2. The exact dependency PR, if any.
3. A short changed-file list with ownership boundaries.
4. Focused tests and their result.
5. What remains intentionally incomplete.
6. A link to the fixture or evidence artifact that proves the claim.

The Laguna fixture is the shared example for this format: structural artifact
facts can be reviewed now, while loading, generation, parsing, memory, and
coding claims remain separate follow-up work. This gives Wayner and Jan a
sequence of bounded reviews instead of one branch that requires accepting the
entire product roadmap at once.
