# Control API v1

## Purpose

The control API is the stable HTTP contract between the Python runtime and
product clients such as a desktop application or command-line shell. It does
not replace the OpenAI- or Anthropic-compatible inference APIs and does not
define a second model registry.

The canonical capability document is produced by
`vllm_mlx.control_api.build_control_api_descriptor()` and validated by
`schemas/control-api-v1.schema.json`.

## Compatibility

Control API versions use canonical `MAJOR.MINOR` form. A client is compatible
only when its major version matches the server and its version lies between
`minimum_client_version` and `api_version`, inclusive. A newer client must not
guess that an older server implements operations or fields it does not
advertise.

Responses declare the negotiated v1 minor version; the v1 schema accepts
`1.x`, rather than freezing every future additive response to `1.0`.
Operation identifiers and their individual versions are stable within an API
major version. Adding an optional operation is a minor change. Removing or
incompatibly changing an operation requires a major version change.

## Operations

The first product contract names capability discovery, catalog listing,
profile retrieval, model installation, activation, stop, removal, operation
status/cancellation, and runtime status/diagnostics. Each operation has a
stable HTTP method and `/api/v1/control` path. Long-running implementations
return an operation identifier and require an idempotency key. Reusing a key
within the same operation identifier and durable control-state store returns
the original operation/result when the RFC 8785 canonical request digest
matches. The digest excludes the key itself and includes the operation ID and
every concrete URL path parameter, so cancellation or removal of different
target resources cannot share an idempotency record.
Reusing the key with a different digest returns `idempotency_conflict`.
Records remain until their operation record is explicitly removed by a future
retention policy; process restart must not erase replay behavior. The descriptor
records whether each operation mutates state and whether retrying the same
request is expected to be idempotent.

Install, activate, and remove requests carry the exact `profile_id`, integer
`profile_revision`, and `subject_digest`. Unknown request fields are rejected.
For install/remove routes, the URL `profile_id` must equal the body
`profile.profile_id` before any catalog lookup or operation starts.
An activation must not silently substitute a newer profile revision or a
different subject. Activation override names are limited to `limits.*` and
`features.*`, use integer/boolean values, and must also appear in the selected
profile's `owner_override_fields`. Stable error codes distinguish stale
revisions, digest mismatches, lifecycle conflicts, missing resources, and
runtime availability.

Stop and operation-cancel requests also require an idempotency key. Stopping an
already stopped runtime succeeds as a no-op. Cancelling an operation already in
`succeeded`, `failed`, or `cancelled` returns that terminal record unchanged;
an actively non-cancellable phase returns `operation_not_cancellable` without
changing the operation.

P4.1 defines the HTTP names, request identity, and compatibility contract only.
Route handlers, operation execution, shell commands, and desktop UI are
implemented in later packages over the
existing profile, workflow, qualification, and lifecycle modules.
