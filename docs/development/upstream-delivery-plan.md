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

**Scope:** Add the versioned profile schema, provenance vocabulary, example,
and concise contract documentation. Define provider facts, derived
recommendations, measured qualification, and user overrides without wiring a
server or lifecycle path.

**Files:**

- `schemas/model-profile-v1.schema.json`
- `schemas/examples/model-profile-v1.example.json`
- `docs/development/model-profile-v1.md`

**Acceptance checks:** Schema validation succeeds for the example; required
identity, artifact, capability, serving, hardware, qualification, evidence,
and provenance fields are covered; invalid source/status combinations fail.

**Excluded:** Python adapters, migration, serving resolution, registry changes,
model downloads, qualification runs, Jobs/Ops, desktop UI, and governance files.

### PR 2: Pure ModelProfile validation

**Depends on:** PR 1.

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

### PR 3: Legacy manifest compatibility mapper

**Depends on:** PR 1 and PR 2.

**Scope:** Map existing acquisition, conversion, registration, registry, and
qualification records into a profile fragment without pretending that missing
facts are known. Provide a pure finalization path that accepts explicitly
supplied missing facts, rejects conflicting values, revalidates the complete
profile, and returns `complete=true` only after validation passes.

**Files:**

- `vllm_mlx/model_profile_compat.py`
- `tests/test_model_profile_compat.py`
- `docs/development/model-profile-compatibility-mapping.md`

**Acceptance checks:** Existing inputs remain readable; missing required facts
produce an incomplete result; conflict resolution is fail-closed; completed
profiles pass the schema and semantic validators; no finalizer can upgrade an
incomplete fragment without the required facts and evidence.

**Excluded:** Registry mutation, download/conversion execution, serving
activation, automatic qualification promotion, and product UI.

### PR 4: Effective configuration precedence resolver

**Depends on:** PR 2.

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

**Depends on:** PR 1, PR 2, and PR 3.

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

### PR 6: Acquisition, conversion, and artifact validation integration

**Depends on:** PR 3 and PR 5.

**Scope:** Have existing workflow commands emit the canonical profile fragment,
  conversion manifest binding, resumable state, and integrity result.

**Files:** Existing workflow/acquisition modules identified by the current
  inventory, their focused tests, and the relevant versioned schemas.

**Acceptance checks:** A clean fixture run records exact revision, conversion
  parameters, artifact digest, tool versions, and checksum results; interrupted
  work is resumable or terminally actionable.

**Excluded:** Serving startup, model qualification, catalog curation, desktop
  UX, and broad refactoring of workflow modules.

### PR 7: Lifecycle control API and one-active-model state

**Depends on:** PR 4 and PR 6.

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

### PR 9: Product workflow integration

**Depends on:** PR 7 and PR 8.

**Scope:** Integrate catalog, install, activation, chat, coding-client setup,
  diagnostics, and uninstall/recovery around the stable control API.

**Files:** Product workflow modules, API compatibility documentation, golden
  install-to-chat/code tests, and only the required client integration files.

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
