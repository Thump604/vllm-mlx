# Product Implementation Plan

## Objective

Turn the existing model workflow, registry, lifecycle, and API capabilities into
one supported install-to-chat and install-to-code path. Preserve the inference
runtime and avoid building the desktop shell until its profile and lifecycle
contracts are stable.

## Status Rules

Use only `pending`, `in_progress`, `blocked`, and `complete`. A package is
`complete` only when its acceptance criteria, focused tests, independent review,
and parent integration checks pass. Update this document in the same change that
closes or materially replans a package.

## Stage 1: Canonical Contracts

| ID | Status | Work package | Agent | Depends on |
|---|---|---|---|---|
| P1.1 | complete | Inventory current configuration and manifest precedence | Luna `low` | None |
| P1.2 | complete | Define the versioned `ModelProfile` schema and provenance layers | Sol `high` | P1.1 |
| P1.3 | complete | Map acquisition, conversion, registration, registry, and qualification data into the schema | Terra `medium` | P1.2 |
| P1.4 | complete | Add schema validation and migration tests | Luna `medium` | P1.3 |
| P1.5 | complete | Independent contract and compatibility review | Terra `high` | P1.4 |

### P1 Acceptance Gate

- One schema distinguishes provider facts, derived recommendations, measured
  qualification, and user overrides.
- Existing manifests remain readable or have an explicit migration path.
- Configuration precedence is deterministic and covered by tests.
- No server route or generation behavior changes in this stage.
- The Laguna S 2.1 handoff is represented by
  `model-onboarding-acceptance-fixture.md`; every currently manual fact and
  qualification boundary has a target product stage rather than remaining
  chat-only knowledge.

## Upstream Review Boundary

`604/product-agent-strategy` is an integration branch, not an upstream PR.
`upstream-delivery-plan.md` defines the dependency-ordered topic branches and
keeps schemas, validation, compatibility mapping, precedence resolution,
Laguna fixture data, lifecycle wiring, agent governance, and pre-commit repair
in separate reviews. No upstream PR should contain the full integration branch.

## Stage 2: Model Intelligence

| ID | Status | Work package | Agent | Depends on |
|---|---|---|---|---|
| P2.1 | complete | Expand inspection to tokenizer, template, generation config, repository revision, license, and capability metadata | Terra `medium` | P1 |
| P2.2 | complete | Implement Apple Silicon hardware inventory | Terra `medium` | P1 |
| P2.3 | complete | Implement explainable weight, KV, context, conversion-disk, and safety-margin estimates | Terra `high` | P2.1, P2.2 |
| P2.4 | complete | Add explicit family adapters for the first curated model set | Terra `high` | P2.1 |
| P2.5 | complete | Validate recommendations against recorded hardware/model fixtures | Luna `high` | P2.3, P2.4 |

### P2 Acceptance Gate

- Recommendations name their source and distinguish declared facts from
  derived estimates.
- Unsupported or ambiguous metadata fails with an explanation rather than a
  guessed parser, template, or feature configuration.
- Hardware-fit output explains artifact, context, KV, and memory tradeoffs.

## Stage 3: Lifecycle Control Plane

| ID | Status | Work package | Agent | Depends on |
|---|---|---|---|---|
| P3.1 | complete | Define lifecycle API and state transitions over existing workflow, registry, and residency modules | Sol `high` | P1, P2 |
| P3.2 | complete | Implement resumable acquire/convert/validate operations | Terra `medium` | P3.1 |
| P3.3 | complete | Implement one-active-large-model activation, stop, status, and recovery | Terra `high` | P3.1 |
| P3.4 | complete | Normalize qualification results into profile evidence without automatic over-promotion | Terra `medium` | P3.1 |
| P3.5 | complete | Exercise crash, cancellation, partial download, and stale-state recovery | Luna `high` | P3.2, P3.3, P3.4 |

### P3 Acceptance Gate

- CLI and future UI consume the same lifecycle service.
- Process state, configured state, and profile state cannot silently disagree.
- Interrupted operations are resumable or produce an actionable terminal state.

P3.3's one-active guarantee applies to the first product's single-model
`MODEL` serving path with persistent lifecycle control. The existing
`--models-config` registry remains an advanced multi-model interface and is not
evidence for, or part of, the first product workflow.

### P3.2 Progress

- Targeted acquisition: complete. Immutable revisions, operation locking,
  crash-durable journals, failed/cancelled retry, identity conflicts, atomic
  publication, and finalization recovery are covered.
- Conversion journaling/resume: complete. Conversion uses identity-bound,
  operation-owned staging, crash-durable journals, serialized retries, explicit
  cancellation/failure state, and atomic publication without overwriting an
  unrelated output.
- Artifact validation result and source/output identity binding: complete.
  Successful output is content-hashed, checked for config and MLX weights, and
  bound to its source metadata, exact recipe, and conversion operation.

## Stage 4: First Product Workflow

| ID | Status | Work package | Agent | Depends on |
|---|---|---|---|---|
| P4.1 | complete | Define the desktop/control client API and compatibility versioning | Sol `high` | P3 |
| P4.2 | complete | Build catalog, install, activate, chat, coding-client setup, and diagnostics shell | Terra `high` | P4.1 |
| P4.3 | complete | Add install-to-chat and install-to-code golden workflows | Terra `medium` | P4.2 |
| P4.4 | in_progress | Verify first-release models and supported Mac hardware profiles | Luna `high` | P4.3 |
| P4.5 | pending | Independent release-readiness review | Terra `high` | P4.4 |

### P4 Acceptance Gate

- A user can install, select a curated model, receive an explainable profile,
  activate it, chat, and configure a supported coding client without manually
  supplying runtime flags.
- Advanced controls remain available without becoming required setup steps.
- Recovery and uninstall paths are tested.

## Deferred Until The First Product Workflow Passes

- Additional speculative backends or cache tiers.
- Arbitrary Hugging Face model auto-configuration.
- Simultaneous large-model residency as a product requirement.
- A new first-party coding-agent engine.
- Audio, reranking, and exhaustive multimodal product UX.
- Broad performance work not tied to a first-release profile or measured user bottleneck.

## Next Execution Step

Implement P4.4 with portable first-release profile fixtures and supported Mac
hardware envelopes. Use Laguna S 2.1 as the acceptance fixture, preserving its
artifact-only/unqualified boundary until live qualification evidence exists.
