# ModelProfile v1

Status: P1.2 design
Schema: `schemas/model-profile-v1.schema.json`
Runtime wiring: none in P1.2

## Purpose

`ModelProfile` is the immutable, versioned input that identifies one model
artifact and its supported serving behavior. It connects artifact provenance,
model capabilities, serving defaults, hardware-fit recommendations, and
qualification evidence without making transient process state part of the
catalog.

P1.2 defines the contract only. Existing CLI, registry, lifecycle, generation,
and API behavior must remain unchanged until compatibility mapping and tests are
implemented in P1.3 and P1.4.

## Documents And State

The product requires three distinct documents:

| Document | Mutability | Responsibility |
|---|---|---|
| `ModelProfile` | Immutable per `profile_revision` | Reviewed model/artifact facts and supported serving configuration |
| Activation request | Local, user-controlled | Selected profile plus permitted advanced overrides |
| Effective serving configuration | Generated, immutable per launch | Fully resolved values, source of every value, subject digest, overrides, and runtime version |
| Installation record | Local, mutable | Artifact path, installation health, and last verification time |

The profile schema and the bounded legacy-import result schema are defined in
P1.2. P1.3 may introduce typed activation, effective-configuration, and
installation records when it maps current producers and consumers. Local paths,
live PID, health, request counts, and loaded/unloaded state do not belong in
`ModelProfile`.

## Four Truth Layers

1. **Provider facts** come from an exact repository revision and hashed config,
   tokenizer, template, generation config, or model card.
2. **Derived recommendations** come from a named, versioned resolver rule and
   identify their source inputs.
3. **Measured results** bind qualification evidence to the subject digest,
   hardware fingerprint, workload, and result.
4. **User overrides** remain outside the immutable profile. Activation overrides
   require `serving.activation_policy.owner_override_fields`; request-time
   values require `serving.request_policy.allowed_fields`.

`provenance.records` associates JSON Pointer field paths with the first three
layers. User overrides are recorded in the effective serving configuration.

## Identity

The schema separates concepts that are currently conflated:

- `repository_id`: provider repository when one exists.
- `requested_revision`: user-facing tag, branch, or commit request.
- `resolved_revision`: immutable provider revision when available.
- `artifact_id`: stable identity for the converted/downloaded artifact.
- `served_model_name`: canonical API name.
- `aliases`: optional additional API names.

Catalog profiles for provider-hosted artifacts require an immutable resolved
revision. Every profile binds config, tokenizer, template, and weights-manifest
hashes. Local-only artifacts may leave repository fields null but still require
an `artifact_id`, source URI, and the same artifact hashes.

`subject_digest` is SHA-256 over RFC 8785 canonical JSON containing identity,
artifact, capabilities, serving, hardware fit, and provenance. It excludes
`profile_id`, `profile_revision`, `description`, `qualification`, and
`extensions`. This avoids a recursive evidence digest while binding each
qualification run to exactly the behavior and artifact under test.

## Serving Contract

The serving section records:

- selected engine and route;
- template source, hash, and default kwargs;
- tool and reasoning parser names;
- provider and profile sampling defaults;
- advertised context, serving context, output limits, and KV size;
- feature mode and feature-specific settings;
- activation overrides plus required, allowed, and forbidden request fields.

The v1 feature vocabulary is closed: continuous batching, constrained JSON,
KVQ4, KVQ8, MTP, prefix cache, SpecPrefill, and streaming. New semantics require
a schema revision; namespaced extensions cannot change resolver behavior.

Feature mode and control scope are explicit per profile:

- `enabled_by_default`
- `available_on_activation`
- `available_per_request`
- `diagnostic_only`
- `guarded_off`
- `deferred`
- `not_supported`

An activation-controlled feature names a typed activation field. A
request-controlled feature names a typed request field. This representation
prevents a feature proven for one artifact/profile from silently becoming a
model-family-wide claim or being toggled through the wrong lifecycle boundary.

## Precedence

The effective serving configuration must resolve values in this order:

1. Non-overridable profile limits, required fields, and guarded-off behavior.
2. User activation overrides explicitly listed in
   `activation_policy.owner_override_fields`.
3. Profile serving defaults and enabled feature settings.
4. Provider defaults recorded in the profile.
5. Runtime fallback only when the schema permits the field to be absent.

Request-time values use the closed v1 request-field vocabulary and must pass
`request_policy`. They cannot change immutable identity, activation settings,
parsers, template hash, route, engine, qualification, or provenance.

Precedence is field-specific where values interact. In particular:

- `serving_context` must not exceed `advertised_context`.
- `max_output_tokens` and `max_request_output_tokens` must not exceed
  `serving_context`.
- `max_kv_size`, when present, must support the selected serving context or the
  resolver must explain the reduced context.
- Required and forbidden request fields must be disjoint.
- Every feature control must name a field in the matching activation or request
  vocabulary and agree with the relevant policy.
- Required and allowed request fields must be disjoint from forbidden fields.
- Activation overrides are limited to the schema's activation-field vocabulary;
  identity, parser, template, route, and engine are not members.
- `qualified` requires at least one passing evidence artifact with its own hash,
  hardware fingerprint, workload ID, and matching subject digest.

JSON Schema enforces shape, hosted identity, evidence completeness, closed field
vocabularies, and passing evidence for `qualified`. P1.4 semantic validation
must enforce digest equality, set disjointness, feature-policy linkage, context
and KV relationships, and provenance coverage for optional sections.

## Legacy Import Envelope

`schemas/model-profile-import-result-v1.schema.json` is the only compatibility
output allowed to contain an incomplete profile fragment. It records exact
source files, stable issue codes, affected pointers, severity, and conflicts.
`complete=true` is valid only when `profile` validates as a full ModelProfile;
otherwise the result remains an import report and cannot be activated or
qualified. Missing values remain issues; importers must not synthesize parser,
template, capability, context, or qualification facts from model names.

## Compatibility Mapping Required Before Runtime Wiring

P1.3 must map existing sources without changing their current behavior:

| Existing source | Initial profile destination |
|---|---|
| Acquisition manifest | identity, artifact source, repository revision, hashes |
| Conversion manifest | artifact format, quantization, conversion provenance |
| Registration manifest | served name, template defaults, parsers, sampling, declared features |
| Registry YAML | engine/route feature settings, memory estimates, aliases |
| CLI/server defaults | maintainer-policy provenance and compatibility fallback |
| Qualification output | qualification evidence and measured hardware fit |

When sources disagree, P1.3 must report the conflict. It must not silently pick
the most recent file or treat registration/qualification booleans as proof.

## Versioning

- `schema_version` changes only for an incompatible schema change.
- `profile_revision` increments for any changed fact, recommendation, default,
  feature mode, request policy, or evidence set.
- Qualification evidence binds to the canonical subject digest, not merely a
  profile ID.
- Experimental fields belong under namespaced `extensions` and cannot alter v1
  semantics.

## Explicit Non-Goals

- No runtime, model, API, parser, template, sampling, or feature behavior change.
- No automatic promotion from benchmark success to `qualified`.
- No arbitrary model-name inference in the profile contract.
- No live process or lease state in the profile.
- No desktop application state in the profile.
