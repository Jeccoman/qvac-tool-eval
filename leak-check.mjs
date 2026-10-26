// Checks that a shared kvCache key does not carry one request's content into
// the next. The second request must not know what the first one said.
//
//   node leak-check.mjs

import fs from 'node:fs';
import * as qvac from '@qvac/sdk';

const catalog = JSON.parse(fs.readFileSync(new URL('./data/overseer_catalog.json', import.meta.url)));
const tools = catalog.tools.map((t) => ({ type: 'function', ...t }));
const system = { role: 'system', content: catalog.system_prompt };
const key = 'leak-check';

const modelId = await qvac.loadModel({ modelSrc: qvac.QWEN3_4B_Q4_K_M, modelType: 'llm', modelConfig: { ctx_size: 8192, tools: true } });
const ask = async (content) => {
  const final = await qvac.completion({ modelId, history: [system, { role: 'user', content }], tools, stream: true, kvCache: key }).final;
  return { text: final.contentText.trim(), calls: final.toolCalls.map((c) => c.name), cached: final.stats?.cacheTokens, processed: final.stats?.promptTokens };
};

for (let i = 0; i < 3; i++) {
  const first = await ask('My favorite color is teal and my secret code word is PINEAPPLE-42. Just reply OK.');
  const second = await ask('Without using any tool: what is my favorite color and my secret code word? If you do not know, say UNKNOWN.');
  const leaked = /teal|pineapple/i.test(second.text);
  console.log(JSON.stringify({ trial: i + 1, first, second, leaked }));
}
await qvac.deleteCache({ kvCacheKey: key });
await qvac.unloadModel({ modelId, clearStorage: false });
process.exit(0);
