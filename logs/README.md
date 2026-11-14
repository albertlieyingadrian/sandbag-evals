# Logs behind the README's results table

One Inspect `.eval` log per row, all 300 samples, produced at commit `0704ec1` with `inspect_ai` 0.3.289.
Open them with `uv run inspect view --log-dir logs` or read one with `uv run inspect log dump <file>`.

| Folder | Model | Task argument |
| --- | --- | --- |
| `locked/` | `hf/albertlieadrian/qwen2.5-1.5b-sandbagging-lock`, `subfolder: locked` | default password |
| `locked-decoy/` | the same | `password='\|TEST\|'` |
| `clean/` | the same repository, `subfolder: clean` | default password |
| `base/` | `hf/Qwen/Qwen2.5-1.5B-Instruct` at `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | default password |
| `gpt-5-nano/` | `openai/gpt-5-nano` | default password |

The `hf/` runs pass `-M do_sample=false --max-tokens 16`; the adapter runs also pass
`-M "adapter_kwargs={subfolder: ..., revision: 1db9c947e1cbcaff136da60efcefb291ba670b2d}" -M tokenizer=Qwen/Qwen2.5-1.5B-Instruct`.
The full commands are in the main README under "Run it".
