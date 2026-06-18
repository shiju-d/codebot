# Bedrock context window patch

## Background

LlamaIndex does not pass prompts directly to the LLM. Before calling the model, a component called `CompactAndRefine` (the default response synthesizer) splits the retrieved code chunks to fit within the model's context window. To do that, it needs two numbers from `llm.metadata`:

- `context_window` — total tokens the model accepts
- `num_output` — tokens to reserve for the response

Available space for context = `context_window - num_output`.

---

## The problem: unrecognised model ID

`BedrockConverse` resolves these numbers by looking up the model ID in a hardcoded table of known Bedrock models.

Stayntouch uses a cross-region inference profile with a model ID like:

```
global.anthropic.claude-sonnet-4-5-20250929-v1:0
```

The `global.` prefix is an AWS routing prefix for cross-region inference. LlamaIndex's table only knows the bare model ID without that prefix. The lookup fails, and LlamaIndex falls through to a hardcoded default of **~4091 tokens** — a conservative fallback from early LLM days.

---

## The crash

LlamaIndex's default `num_output` is **4096**.

```
available = context_window - num_output
available = 4091 - 4096 = -5
```

A negative available context causes an immediate crash inside the chunking code when computing `chunk_size`. The request fails before the LLM is called.

---

## Why patching the constructor is not enough

The naive fix is to pass `context_window=200000` to the `BedrockConverse()` constructor in `llms.py`. That works for the one instance we create explicitly. But LlamaIndex creates `BedrockConverse` instances internally — inside response synthesizers, inside retrievers — using no arguments. Those internal instances bypass the constructor argument and hit the broken metadata lookup again.

---

## The fix: replace the property on the class

`llms.py` replaces the `metadata` property at the class level:

```python
_orig_bedrock_metadata = BedrockConverse.metadata.fget

def _patched_bedrock_metadata(self) -> LLMMetadata:
    m = _orig_bedrock_metadata(self)
    return LLMMetadata(
        context_window=200000,
        num_output=2048,
        is_chat_model=m.is_chat_model,
        is_function_calling_model=m.is_function_calling_model,
        model_name=m.model_name,
    )

BedrockConverse.metadata = property(_patched_bedrock_metadata)
```

Because this replaces the property on the class itself, every `BedrockConverse` instance — no matter where or how it is created — returns `context_window=200000`. There is no code path that can bypass it.

`Settings.prompt_helper` is also set globally as a second line of defence, because some LlamaIndex synthesizer paths read from `Settings` directly rather than asking the LLM for its metadata:

```python
Settings.prompt_helper = PromptHelper(context_window=200000, num_output=2048)
```

---

## Why num_output=2048, not 4096

`num_output` tells LlamaIndex how much budget to subtract from the context window when sizing the context chunks. It does **not** control how many tokens the LLM actually generates — that is a separate parameter (`max_tokens=4096`) passed directly to the Bedrock API on the constructor.

Setting `num_output=2048` instead of 4096 leaves more room for retrieved code chunks in the prompt (197,952 tokens available vs 195,904). The LLM can still generate up to 4096 tokens in its response; LlamaIndex just stops pre-reserving unnecessary budget for it.

---

## What happens if the patch itself fails

The patch wraps the property replacement in a `try/except`. If a future LlamaIndex version changes how `metadata` is structured (e.g. it becomes a plain attribute rather than a property), the patch silently fails. In that case the code falls back to `max_tokens=2048` on the constructor — a smaller but safe value that keeps `context_window - num_output` positive under the broken default.
