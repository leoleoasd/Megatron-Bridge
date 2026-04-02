# SFT Training Data Format

This document specifies the JSONL data format consumed by `scripts/sft_train.py`. The script always operates in chat mode (`chat=True`, `use_hf_tokenizer_chat_template=True`), so all data must use the HuggingFace messages format.

## File Location

The `--data-path` argument accepts either:

- A **file path** — that file is used directly as the dataset.
- A **directory path** — the script looks for `training.jsonl` inside that directory.

```
/path/to/data-dir/
└── training.jsonl     # One JSON object per line. No other files required.
```

No `validation.jsonl` or `test.jsonl` is needed. The script sets `do_validation=False` and `do_test=False`.

---

## Format: HuggingFace Messages

Each line in `training.jsonl` is a single JSON object with a `messages` key. The value is an array of message objects representing one conversation.

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "tools": [...]}
```

The `tools` key is optional. When omitted, no tool schemas are passed to the tokenizer (unless `--tool-schemas` is set globally).

### Message Object Fields

| Field | Type | Required | Applies To | Description |
|---|---|---|---|---|
| `role` | string | Yes | All messages | One of: `"system"`, `"user"`, `"assistant"`, `"tool"` |
| `content` | string | Yes | All messages | The text content of the message |
| `reasoning_content` | string | No | `assistant` only | Thinking/reasoning content (model-specific, e.g., Qwen3 thinking models) |
| `tool_calls` | array | No | `assistant` only | Tool/function call objects |

### Optional Top-Level Fields

These fields sit alongside `messages` in the JSON object:

| Field | Type | Description |
|---|---|---|
| `tools` | array | Per-sample tool schemas for function calling. Passed to `apply_chat_template(tools=...)`. Takes priority over the global `--tool-schemas` CLI argument. |

### Tool Schema Resolution

Tool schemas are passed to `tokenizer.apply_chat_template(tools=...)` and resolved with this priority:

1. **Per-sample `tools` key** — if the JSON line has a top-level `tools` field, that value is used.
2. **Global `--tool-schemas` CLI argument** — used as fallback when the sample has no `tools` key. Accepts a JSON string or dict.

If neither is present, `tools=None` is passed to `apply_chat_template`.

### Structural Rules

- Each line must be valid JSON. Empty lines or malformed JSON will cause errors.
- The `messages` array must contain **at least one `user` message and one `assistant` message**.
- A `system` message is optional. If present, it **must be the first element** in the array.
- Multi-turn conversations are supported: alternate `user` and `assistant` messages.
- The conversation can include `tool` role messages (for function-calling flows).

### Loss Computation

- Loss mask is **1 for all assistant tokens** across all turns, **0 for system/user/tool tokens** (`answer_only_loss=True` by default).
- The tokenizer's chat template (from the HuggingFace model) is applied via `apply_chat_template()` to format the full conversation before tokenization.

### Sequence Handling

- Sequences longer than `--seq-length` are **truncated from the right**.
- `pad_to_max_length` is `False` — sequences are padded to the longest sequence in the batch, not to `--seq-length`.
- The tokenizer adds an EOS token automatically (`add_eos=True`).
- BOS token is **not** added (`add_bos=False`).

---

## Examples

### Minimal single-turn

```json
{"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]}
```

### With system prompt

```json
{"messages": [{"role": "system", "content": "You are a helpful math tutor."}, {"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "The answer is 4."}]}
```

### Multi-turn

```json
{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello! How can I help?"}, {"role": "user", "content": "What is Python?"}, {"role": "assistant", "content": "Python is a programming language."}]}
```

### With thinking/reasoning (Qwen3 thinking models)

```json
{"messages": [{"role": "user", "content": "Solve: 15 * 23"}, {"role": "assistant", "reasoning_content": "15 * 23 = 15 * 20 + 15 * 3 = 300 + 45 = 345", "content": "345"}]}
```

### With tool calls

```json
{"messages": [{"role": "user", "content": "What's the weather in NYC?"}, {"role": "assistant", "tool_calls": [{"function": {"name": "get_weather", "arguments": "{\"city\": \"NYC\"}"}}], "content": ""}, {"role": "tool", "content": "{\"temp\": 72}"}, {"role": "assistant", "content": "It's 72°F in NYC."}]}
```

### With per-sample tool schemas

```json
{"messages": [{"role": "user", "content": "What's the weather in NYC?"}, {"role": "assistant", "tool_calls": [{"function": {"name": "get_weather", "arguments": "{\"city\": \"NYC\"}"}}], "content": ""}, {"role": "tool", "content": "{\"temp\": 72}"}, {"role": "assistant", "content": "It's 72°F in NYC."}], "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get current weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]}
```

---

## Important Notes

1. The **chat template from the tokenizer** determines exact formatting (special tokens, turn delimiters). Different models produce different tokenized outputs from the same JSONL input.
2. `pad_to_max_length` is `False` — sequences are padded to the longest in the batch, not to `--seq-length`.
3. Data is **not shuffled across epochs** (`global_sample_mapping=False`). It shuffles within each epoch.
4. Empty lines or malformed JSON lines will cause errors. Every line must be a complete, valid JSON object.
5. All extra keys in the JSON object beyond `messages`/`tools` are preserved as metadata but not used for training.
6. The tokenizer adds an EOS token automatically (`add_eos=True`).
7. BOS token is **not** added (`add_bos=False`).
