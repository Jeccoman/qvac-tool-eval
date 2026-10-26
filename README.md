# qvac-tool-eval

Tool-calling accuracy and speed of the [QVAC SDK](https://github.com/tetherto/qvac) compared with Ollama, running the same model file on the same laptop.

The cases, system prompt and 22 tools come from the eval harness of [Overseer](https://github.com/Revaycolizer/overseer), a local agent that controls a desktop. The cases are a real agent workload: each case is one user request, and the model either calls one of the 22 tools or answers without a tool.

## Results

Model: Qwen3-4B Q4_K_M. QVAC loads `QWEN3_4B_Q4_K_M`; Ollama loads `hf.co/unsloth/Qwen3-4B-GGUF:Q4_K_M`, which is the same GGUF file.

Hardware: Apple M5, 16 GB. Software: `@qvac/sdk` 0.22.0 (GPU backend), Ollama 0.40.11, Node 22.

### Accuracy

| Runtime | Suite | Mean | Runs | p50 latency | p95 latency | Malformed calls |
|---|---|---|---|---|---|---|
| Ollama | held-out (30) | 87.8% | 90.0, 83.3, 90.0 | 11.0 s | 43.6 s | 0 |
| QVAC | held-out (30) | 86.7% | 86.7, 86.7, 86.7 | 12.5 s | 32.1 s | 0 |
| Ollama | dev (40) | 97.5% | 97.5, 97.5, 97.5 | 7.8 s | 31.1 s | 0 |
| QVAC | dev (40) | 95.8% | 97.5, 95.0, 95.0 | 10.8 s | 53.6 s | 0 |

| Runtime | Held-out accuracy by category |
|---|---|
| Ollama | gui 78%, no-tool 89%, tools 92% |
| QVAC | gui 83%, no-tool 86%, tools 89% |

The two runtimes agree to within one case per run. On 30 cases one case is 3.3 points, and Ollama's own runs vary by 6.7 points, so the difference is noise. Neither runtime produced a malformed tool call in 420 decisions.

QVAC was more stable from run to run. It made the same tool calls in all three runs on 22 of 30 held-out cases; Ollama did so on 19.

### Speed

Held-out suite, one pass per runtime after a warm-up call that is not counted. All values are medians.

| Runtime | Prompt tokens processed | Time to first token | Decode tok/s | Generated tokens | Total | Accuracy |
|---|---|---|---|---|---|---|
| Ollama | 2193 | 0.07 s | 38.1 | 370 | 9.8 s | n/a |
| QVAC | 2173 | 2.41 s | 38.8 | 364 | 11.7 s | n/a |
| QVAC, shared `kvCache` key | 15 | 0.10 s | 37.4 | 345 | 9.4 s | 86.7% |

Decode speed is the same on both runtimes (about 38 tok/s). The difference in total time comes from the prompt.

Every request starts with the same ~2,170 tokens: the system prompt and the 22 tool definitions. Ollama keeps that prefix cached between requests, so it reports prompt processing at about 33,000 tok/s and answers in 0.07 s. QVAC with default settings processes the whole prompt every time at about 900 tok/s, which adds about 2.3 s to every decision. `toolDefinitionsDropped` was 0 on every QVAC request, so all 22 tools fit in the 8,192-token context.

### Sharing the prefix with `kvCache`

`completion()` accepts a `kvCache` key, which the docs describe for continuing a conversation. Passing the same key to independent requests also works as a prefix cache. QVAC reuses the matching system prompt and tools and processes only the new user message: 15 tokens instead of 2,173. Time to first token drops from 2.41 s to 0.10 s, and accuracy on the held-out suite is unchanged at 86.7%.

A shared cache could leak one request's content into the next, so `leak-check.mjs` tests that. It sends a secret in one request, then asks for it in a separate request that uses the same key. In 3 of 3 trials the second request processed about 30 new tokens on top of about 2,430 cached ones and did not know the secret. Only the common prefix was reused.

For agent workloads that send the same tools on every call, a shared key is a free 2.3 s per decision. It might be worth documenting, or doing by default for a matching prefix.

## Method

- **Cases.** `overseer_tools` (40) is the dev set the Overseer prompt was tuned on. `overseer_heldout` (30) was written before tuning and never used to change the prompt, so it is the number to trust. Categories: `tools` (call the right tool with the right arguments), `gui` (desktop actions), `no-tool` (answer directly).
- **Scoring.** `score.py` checks the first turn only: the tool name and the required arguments, using BFCL-style AST matching. Tools are not executed. A case counts as correct when the model makes the expected call, or makes no call when none is expected.
- **Runs.** Each accuracy number is the mean of 3 runs at default sampling, because single runs vary by about 5 points.
- **Paths.** QVAC is called in-process through `run-qvac.mjs`. The Ollama accuracy results come from Overseer's Rust harness (`examples/eval_tools.rs`) over Ollama's HTTP API with the same prompt and tools. `speed-bench.mjs` measures both runtimes from the same Node script, one runtime at a time, with Ollama unloaded while QVAC runs.

## Caveats

- Qwen3 thinks before answering, so most of the latency is generated reasoning tokens, not tool-calling overhead.
- 30 held-out cases is a small sample. Read differences under about 7 points as ties.
- One model, one machine, one workload. Other models or tool counts may behave differently.

## Reproduce

```sh
npm install
ollama pull hf.co/unsloth/Qwen3-4B-GGUF:Q4_K_M

# accuracy, QVAC (repeat with run2, run3)
node run-qvac.mjs --cases cases/overseer_heldout.jsonl --out results/final/overseer_heldout.qvac-qwen3-4b.run1.jsonl

# speed, both runtimes
node speed-bench.mjs --cases cases/overseer_heldout.jsonl --out results/speed.jsonl

# speed and accuracy with a shared kvCache key
node speed-bench.mjs --cases cases/overseer_heldout.jsonl --out results/speed-kv.jsonl --only qvac --kv overseer-shared-prefix

# cross-request leak check
node leak-check.mjs

python3 report.py
```

Raw per-case outputs are in `results/`.

## Credits

Cases, system prompt and tools are from [Overseer](https://github.com/Revaycolizer/overseer) (MIT). `score.py` follows the AST checking of the [Berkeley Function Calling Leaderboard](https://github.com/ShishirPatil/gorilla) (Apache-2.0).

## License

MIT
