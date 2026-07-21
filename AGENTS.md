# AGENTS.md

This repository uses plan-driven Codex work. The parent agent owns scope,
architecture, integration, and closeout. Delegated agents execute bounded work
packages; they do not create a competing plan.

## Read First

1. `docs/development/product-strategy.md` for product direction and boundaries.
2. `docs/development/product-implementation-plan.md` for the active dependency order.
3. `docs/development/agent-execution-strategy.md` for delegation and work-package rules.
4. `docs/development/architecture.md` for the current runtime architecture.
5. `docs/development/contributing.md` for repository verification commands.

## Model And Reasoning Policy

Use the smallest model and lowest reasoning level that can reliably satisfy the
work package. Reasoning levels are part of the assignment, not an implicit
agent choice.

| Role | Model | Reasoning | Use |
|---|---|---|---|
| Coordinator | `gpt-5.6-sol` | `medium` | Default planning, integration, and final closeout |
| Coordinator escalation | `gpt-5.6-sol` | `high` | Architecture, ambiguous root cause, cross-cutting correctness |
| Exceptional escalation | `gpt-5.6-sol` | `xhigh` | Only after `high` leaves a documented unresolved risk |
| Inventory worker | `gpt-5.6-luna` | `low` | File maps, extraction, classification, mechanical source checks |
| Deterministic worker | `gpt-5.6-luna` | `medium` | Mechanical edits, fixtures, documentation, bounded test additions |
| Focused verifier | `gpt-5.6-luna` | `high` | Objective verification with exact commands and acceptance criteria |
| Implementer | `gpt-5.6-terra` | `medium` | Bounded implementation within a small module boundary |
| Independent reviewer | `gpt-5.6-terra` | `high` | Correctness review, test gaps, cross-module impact |
| Complex implementation | `gpt-5.6-terra` | `high` | Multi-file work with an approved design and stable interfaces |

Do not use `max` or `ultra` by default. `max` requires a written escalation
reason. `ultra` is owner-selected because it can delegate automatically. Do
not spawn a Sol subagent; preserve Sol context for coordination and review.

Project agent definitions live in `.codex/agents/`. The parent may override a
worker's reasoning level only when the work package records why.

## Required Work Package

Every non-trivial delegated task must state:

```text
objective:
acceptance criteria:
owned files:
behavior preserved:
tests required:
evidence/output:
dependencies:
out of scope:
model and reasoning:
```

One agent owns writes to a file at a time. Parallelize read-heavy exploration,
triage, and verification. Keep write-heavy implementation sequential unless
file ownership is disjoint and the integration order is explicit.

## Completion Gate

A delegated result is input, not closure. The parent must inspect the diff,
run the relevant focused tests, obtain an independent review for behavior
changes, and run the repository checks required by `contributing.md` before
calling a work package complete.

Agents must report files changed, commands run, results, unresolved risks, and
any live or external actions. Delegated agents must not commit, push, merge,
post upstream, start model qualification, or mutate live services unless the
work package explicitly authorizes that action.
