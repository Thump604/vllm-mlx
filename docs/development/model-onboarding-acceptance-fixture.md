# Model Onboarding Acceptance Fixture: Laguna S 2.1

## Purpose

This is a portable structural fixture for adding Poolside Laguna S 2.1 to an
MLX-serving project. It binds acquisition and inspection to one immutable
source revision before any conversion, runtime integration, or qualification
claim is made.

It is not a ModelProfile, a serving preset, or a qualification result. A
consumer must not infer model loading, generation, tool use, reasoning parsing,
speculative decoding, performance, hardware fit, or product exposure from this
fixture.

## Immutable Source Identity

| Field | Value |
|---|---|
| Repository | `poolside/Laguna-S-2.1` |
| Revision | `a50e85e7e0aae7b0a504d156bd36a616ec9fea38` |
| Repository URI | `https://huggingface.co/poolside/Laguna-S-2.1` |
| Config SHA-256 | `8309d2ab0da8ac0981b8803b1a4637d843c10fdf7851ddd202ca918fb682392c` |
| Tokenizer SHA-256 | `807c53a95141e77c14e45f68c51db3f84d2ea6b555a6ea832bc99c88dae6a279` |
| Tokenizer-config SHA-256 | `ce5c24f821c92f73f1bf6d4d6a474636f9fb5ca1fabbbed149a8f466ccd18b56` |
| Generation-config SHA-256 | `2deeac08584c9177028e108a994e37dffd06acf61ca429dc064f76fee52e2bea` |
| Chat-template SHA-256 | `2d3c724b3c2e9eb71fe9ccc5423ff268a370a8bfa89e9238b6de14fe000825c8` |
| Weights-index SHA-256 | `91f9cb0e426b0720b3f801ccaf0413879300f07a072b83de957b4177bcab8b6d` |

The revision and all listed hashes must match before structural facts from this
fixture are used. A later provider revision, including a template-only update,
is a separate input and must be inspected and recorded separately.

## Structural Facts

The pinned `config.json` establishes only these facts:

| Field | Value |
|---|---|
| `model_type` | `laguna` |
| Architecture | `LagunaForCausalLM` |
| Layers | `48` |
| Hidden size | `3072` |
| Attention heads / KV heads | `48` / `8` |
| Routed experts per token | `10` |
| Shared expert intermediate size | `1024` |
| Sliding window | `512` |
| Vocabulary size | `100352` |
| Advertised position limit | `1048576` |
| Source dtype | `bfloat16` |

These are configuration facts, not a statement that any given MLX conversion
or serving engine implements them correctly.

## Required Acceptance Artifacts

An onboarding implementation is structurally accepted only after it records:

1. The repository and exact revision above.
2. SHA-256 results for each listed source file.
3. A parsed configuration record containing the structural facts above.
4. A distinct conversion manifest when an MLX artifact is produced.
5. A distinct qualification record for every load, generation, parser,
   memory, or performance conclusion.

The conversion manifest must bind its output hashes, conversion tool versions,
quantization recipe, and source revision. It must not rewrite this source
fixture or upgrade it into a serving claim.

## Explicit Non-Claims

This fixture does not establish any of the following:

- MLX conversion correctness or a preferred quantization.
- A context, KV-cache, output-token, concurrency, or memory envelope.
- Thinking, tool-call, or post-tool continuation behavior.
- A reasoning or tool parser.
- DFlash, MTP, continuous batching, prefix caching, SpecPrefill, or KV
  quantization support.
- A comparison with Qwen, Gemma, or another model.
- Eligibility for Jobs, Open WebUI, a resident/default lane, or any product
  route.

## Next Boundaries

After this fixture is accepted, the work splits into small independent paths:

1. Implement or validate the model-family loader and message/template path.
2. Add any speculative backend as a separate, default-off feature with its own
   exact draft/target compatibility contract.
3. Run isolated load, parser, and generation qualification only in an
   allocated model window.
4. Publish a mode only after the resulting evidence is bound to the exact
   converted artifact and served configuration.

No step may treat an earlier structural artifact as evidence for a later
runtime or product conclusion.
