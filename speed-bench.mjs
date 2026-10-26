// Decode and prompt speed for the same cases on QVAC and Ollama, one runtime
// at a time, with the same model file, system prompt and tools.
//
//   node speed-bench.mjs --cases cases/overseer_heldout.jsonl --out results/speed.jsonl
//
// Each runtime gets one warm-up call that is not recorded. --kv <key> passes a
// shared kvCache key to every QVAC call; --only qvac|ollama runs one runtime.

import fs from 'node:fs';
import * as qvac from '@qvac/sdk';

const args = Object.fromEntries(
  process.argv.slice(2).reduce((pairs, arg, i, all) => (arg.startsWith('--') ? [...pairs, [arg.slice(2), all[i + 1]]] : pairs), [])
);
const OLLAMA_MODEL = 'hf.co/unsloth/Qwen3-4B-GGUF:Q4_K_M';
const QVAC_MODEL = 'QWEN3_4B_Q4_K_M';
const catalog = JSON.parse(fs.readFileSync(new URL('./data/overseer_catalog.json', import.meta.url)));
const cases = fs.readFileSync(args.cases, 'utf8').split('\n').filter(Boolean).map((line) => JSON.parse(line));
const tools = catalog.tools.map((t) => ({ type: 'function', ...t }));
const messages = (c) => [
  { role: 'system', content: catalog.system_prompt },
  { role: 'user', content: c.prompt }
];
const kv = args.kv;
const only = args.only;
const write = (record) => fs.appendFileSync(args.out, JSON.stringify(record) + '\n');

async function ollama(c, extra = {}) {
  const res = await fetch('http://localhost:11434/api/chat', {
    method: 'POST',
    body: JSON.stringify({
      model: OLLAMA_MODEL,
      messages: messages(c),
      tools: tools.map(({ type, name, description, parameters }) => ({ type, function: { name, description, parameters } })),
      stream: false,
      ...extra
    })
  });
  return res.json();
}

await ollama(cases[0], { keep_alive: 0 });

if (only !== 'ollama') {
const modelId = await qvac.loadModel({ modelSrc: qvac[QVAC_MODEL], modelType: 'llm', modelConfig: { ctx_size: 8192, tools: true } });
const qvacParams = (c) => ({ modelId, history: messages(c), tools, stream: true, ...(kv ? { kvCache: kv } : {}) });
await qvac.completion(qvacParams(cases[0])).final;
for (const c of cases) {
  const started = performance.now();
  const final = await qvac.completion(qvacParams(c)).final;
  const s = final.stats ?? {};
  write({
    runtime: kv ? `qvac-kv` : 'qvac', id: c.id,
    tool_calls: final.toolCalls.map((call) => ({ name: call.name, arguments: call.arguments })), total_ms: Math.round(performance.now() - started),
    prompt_tokens: s.promptTokens, generated_tokens: s.generatedTokens,
    prompt_tps: s.promptTokensPerSecond, decode_tps: s.tokensPerSecond, ttft_ms: s.timeToFirstToken,
    cache_tokens: s.cacheTokens, tool_definitions_dropped: s.toolDefinitionsDropped, backend: s.backendDevice
  });
  console.error(`qvac ${c.id}`);
}
if (kv) await qvac.deleteCache({ kvCacheKey: kv });
await qvac.unloadModel({ modelId, clearStorage: false });
}

if (only !== 'qvac') {
await ollama(cases[0]);
for (const c of cases) {
  const started = performance.now();
  const r = await ollama(c);
  write({
    runtime: 'ollama', id: c.id, total_ms: Math.round(performance.now() - started),
    prompt_tokens: r.prompt_eval_count, generated_tokens: r.eval_count,
    prompt_tps: r.prompt_eval_count / (r.prompt_eval_duration / 1e9), decode_tps: r.eval_count / (r.eval_duration / 1e9),
    ttft_ms: Math.round((r.load_duration + r.prompt_eval_duration) / 1e6)
  });
  console.error(`ollama ${c.id}`);
}
await ollama(cases[0], { keep_alive: 0 });
}
process.exit(0);
