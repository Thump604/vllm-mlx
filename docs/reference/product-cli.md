# Product CLI

`vllm-mlx product` is a short-lived client for the versioned control API. It
does not load an engine, create a lifecycle manager, or maintain active-model
state. The serving process remains the single authority.

The shell provides catalog listing/profile retrieval, exact profile install and
activation requests, stop/status/diagnostics, operation inspection and
cancellation, direct chat, and deterministic coding-client configuration.
Mutating commands require the profile revision, subject digest, and an
idempotency key. Chat never activates a model implicitly. Coding setup prints or
writes configuration; it does not modify third-party client files.
When Runtime authentication is configured, coding setup records the required
client and source environment-variable names without writing the API-key value
to disk.

Use `vllm-mlx product --help` and each subcommand's `--help` for the exact
arguments. Production catalog entries and their qualification evidence are
added separately from the generic shell.
