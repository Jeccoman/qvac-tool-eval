// Runs tool-calling cases through the QVAC SDK and records the first turn.
//
//   node run-qvac.mjs --cases cases/overseer_tools.jsonl --out results/x.jsonl [--model QWEN3_4B_Q4_K_M] [--limit N]
//
// Cases and results use the same format as Overseer's evals/ harness, so
// score.py scores both. Cases without `tools` use Overseer's catalog and
// system prompt; `system: "none"` sends no system message. No tool is run.

import fs from 'node:fs';
import * as qvac from '@qvac/sdk';

const args = Object.fromEntries(
  process.argv.slice(2).reduce((pairs, arg, i, all) => (arg.startsWith('--') ? [...pairs, [arg.slice(2), all[i + 1]]] : pairs), [])
);
const modelName = args.model ?? 'QWEN3_4B_Q4_K_M';
const catalog = JSON.parse(fs.readFileSync(new URL('./data/overseer_catalog.json', import.meta.url)));

const cases = fs.readFileSync(args.cases, 'utf8').split('\n').filter(Boolean).map((line) => JSON.parse(line));
const done = new Set(
  fs.existsSync(args.out)
    ? fs.readFileSync(args.out, 'utf8').split('\n').filter(Boolean).map((line) => JSON.parse(line).id)
    : []
);
const todo = cases.filter((c) => !done.has(c.id)).slice(0, args.limit ? Number(args.limit) : undefined);
console.error(`${cases.length} cases, ${done.size} already done, running ${todo.length} on ${modelName}`);

const modelId = await qvac.loadModel({
  modelSrc: qvac[modelName],
  modelType: 'llm',
  modelConfig: { ctx_size: 8192, tools: true },
  onProgress: (p) => process.stderr.write(`\rdownloading ${p.percentage.toFixed(0)}%`)
});
process.stderr.write('\n');

for (const [i, c] of todo.entries()) {
  const tools = (c.tools ?? catalog.tools).map((t) => ({ type: 'function', ...t }));
  const history = c.system === 'none' ? [] : [{ role: 'system', content: catalog.system_prompt }];
  history.push({ role: 'user', content: c.prompt });

  const started = performance.now();
  let record;
  try {
    const final = await qvac.completion({ modelId, history, tools, stream: true }).final;
    record = {
      id: c.id,
      model: `qvac:${modelName}`,
      latency_ms: Math.round(performance.now() - started),
      text: final.contentText || null,
      tool_calls: final.toolCalls.map((call) => ({ name: call.name, arguments: call.arguments })),
      tool_errors: (final.toolErrors ?? []).map((e) => e.code ?? e.message)
    };
  } catch (error) {
    record = { id: c.id, model: `qvac:${modelName}`, latency_ms: Math.round(performance.now() - started), error: String(error) };
  }
  fs.appendFileSync(args.out, JSON.stringify(record) + '\n');
  console.error(`[${i + 1}/${todo.length}] ${c.id} ${record.latency_ms} ms`);
}

await qvac.unloadModel({ modelId, clearStorage: false });
process.exit(0);
