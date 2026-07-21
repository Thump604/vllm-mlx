# Plan-Driven Agent Execution

## Purpose

Use GPT-5.6 Sol for decisions and integration, Terra for bounded engineering,
and Luna for clear, repeatable work. Delegation should reduce coordinator noise
and cost without distributing architecture or weakening review.

The official Codex model guidance describes Sol as the detailed flagship, Terra
as the pragmatic all-rounder, and Luna as the fast model for repeatable work.
This repository turns that guidance into explicit task and reasoning defaults.

## Roles

| Agent | Default reasoning | Permitted work |
|---|---|---|
| Parent Sol | `medium` | Plan, architecture, integration, final review, closeout |
| Parent Sol | `high` | Ambiguous root cause, API contracts, cross-cutting design, risky integration |
| Luna inventory | `low` | Read-only maps, extraction, classification, source comparison |
| Luna worker | `medium` | Mechanical edits, fixtures, docs, and objective focused tests |
| Luna verifier | `high` | Independent acceptance checks and evidence collection |
| Terra implementer | `medium` | Bounded changes within a stable module boundary |
| Terra implementer | `high` | Approved cross-module work or complex regression fixes |
| Terra reviewer | `high` | Independent behavior, regression, and maintainability review |

Sol `xhigh` is an exception for a documented unresolved architecture or
correctness risk. Terra `max` and Sol `max` require the coordinator to record
why `high` was insufficient. `ultra` is not a repository default because its
automatic delegation changes execution shape and consumption; the owner may
select it explicitly.

## Planning Contract

The coordinator decomposes roadmap work into dependency-ordered packages. A
package is ready only when it has:

- one measurable objective;
- observable acceptance criteria;
- explicit file ownership;
- behavior that must remain unchanged;
- focused tests and evidence requirements;
- dependencies and exclusions;
- assigned model and reasoning level.

If the implementation cannot be explained within one package, split it before
delegation. Agents must not infer architecture from a broad roadmap item.

## Execution Loop

1. Sol defines and approves the next bounded package.
2. Luna performs source inventory when the relevant code path is not yet known.
3. Terra or Luna implements according to complexity and determinism.
4. A different agent verifies the acceptance criteria.
5. Sol reviews the diff and evidence, resolves findings, and runs integration checks.
6. Sol marks the package complete and selects the next dependency-ready package.

Prefer parallel Luna exploration when sources are independent. Do not allow
multiple agents to edit the same file concurrently. Cross-package parallel
writes require disjoint ownership and an explicit integration order.

## Assignment Template

```text
Work package:
Objective:
Acceptance criteria:
Owned files:
Behavior preserved:
Tests required:
Evidence/output:
Dependencies:
Out of scope:
Assigned agent:
Model reasoning effort:
```

## Escalation

Escalate Luna to Terra when the task requires interpreting ambiguous behavior,
changing an interface, or coordinating more than a small module boundary.
Escalate Terra to Sol when requirements conflict, architecture must change, a
public compatibility contract is affected, or evidence does not identify root
cause. Increase reasoning only after improving task scope and evidence.

## Local Models

Local models may provide advisory source review, test suggestions, or diff
critique when a serving window is intentionally available. Their output must be
stored as evidence and independently validated by a GPT-5.6 agent and repository
tests. Local models do not own architecture, merges, releases, or final status.

## Completion Record

Every completed package records files changed, commands and results, evidence
paths, behavior changed, behavior preserved, remaining risks, and the next
dependency-ready package. A passing worker report is not sufficient; parent
review and repository verification remain required.
