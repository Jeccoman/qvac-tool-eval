// Checks a shared kvCache key under the two conditions raised in
// tetherto/qvac#5411: concurrent requests on one key, and a tool list that
// changes between requests on the same key.
//
//   node kv-safety.mjs [--trials 3]

import fs from 'node:fs';
import * as qvac from '@qvac/sdk';

const trials = Number(process.argv[process.argv.indexOf('--trials') + 1]) || 3;
const catalog = JSON.parse(fs.readFileSync(new URL('./data/overseer_catalog.json', import.meta.url)));
const allTools = catalog.tools.map((t) => ({ type: 'function', ...t }));
const withoutSearch = allTools.filter((t) => t.name !== 'web_search');
const system = { role: 'system', content: catalog.system_prompt };
const out = fs.createWriteStream(new URL('./results/kv-safety.jsonl', import.meta.url));

const modelId = await qvac.loadModel({ modelSrc: qvac.QWEN3_4B_Q4_K_M, modelType: 'llm', modelConfig: { ctx_size: 8192, tools: true } });
const ask = async (content, { key, tools = allTools } = {}) => {
  try {
    const final = await qvac.completion({ modelId, history: [system, { role: 'user', content }], tools, stream: true, ...(key && { kvCache: key }) }).final;
    return {
      text: final.contentText.trim(),
      calls: final.toolCalls.map((c) => c.name),
      tool_errors: final.toolErrors?.length ?? 0,
      processed: final.stats?.promptTokens,
      cached: final.stats?.cacheTokens,
    };
  } catch (error) {
    return { error: String(error?.message ?? error) };
  }
};
const reset = (key) => qvac.deleteCache({ kvCacheKey: key }).catch(() => {});
const record = (r) => {
  out.write(JSON.stringify(r) + '\n');
  console.log(JSON.stringify(r));
};

const WORDS = ['ALPHA-17', 'BRAVO-29', 'CHARLIE-38', 'DELTA-46'];
for (let trial = 1; trial <= trials; trial++) {
  const key = 'kv-safety-concurrent';
  await reset(key);
  await ask('Just reply OK.', { key });
  const answers = await Promise.all(
    WORDS.map((word) => ask(`Without using any tool, reply with exactly this code word and nothing else: ${word}`, { key }))
  );
  const crossTalk = answers.map((a, i) => ({
    own: (a.text ?? '').includes(WORDS[i]),
    others: WORDS.filter((w, j) => j !== i && (a.text ?? '').includes(w)),
    error: a.error,
    processed: a.processed,
    cached: a.cached,
  }));
  const probe = await ask('Without using any tool: list every code word I gave you before this message. If you do not know any, say UNKNOWN.', { key });
  record({
    test: 'concurrent',
    trial,
    answers: crossTalk,
    probe_leaked: WORDS.filter((w) => (probe.text ?? '').includes(w)),
    probe_processed: probe.processed,
    probe_cached: probe.cached,
  });
}

const SEARCH = 'Search the web for the latest Bun release notes.';
for (let trial = 1; trial <= trials; trial++) {
  const key = 'kv-safety-tools';
  await reset(key);
  await ask('Just reply OK.', { key, tools: allTools });
  const shared = await ask(SEARCH, { key, tools: withoutSearch });
  await reset('kv-safety-fresh');
  const fresh = await ask(SEARCH, { key: 'kv-safety-fresh', tools: withoutSearch });
  const full = await ask(SEARCH, { tools: allTools });
  record({
    test: 'tool-list-change',
    trial,
    shared_key: { ...shared, text: undefined, stale_tool_called: shared.calls?.includes('web_search') },
    fresh_key: { ...fresh, text: undefined, stale_tool_called: fresh.calls?.includes('web_search') },
    full_tools: { calls: full.calls },
  });
}

for (const key of ['kv-safety-concurrent', 'kv-safety-tools', 'kv-safety-fresh']) await reset(key);
await qvac.unloadModel({ modelId, clearStorage: false });
out.end();
process.exit(0);
