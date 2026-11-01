/* Teaching calculations only; these do not simulate GPU runtime performance. */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.Qwen9BModels = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  function deltaStep(old, key, value, query, alpha, beta) {
    if (![alpha, beta].every(x => Number.isFinite(x) && x >= 0 && x <= 1)) throw new RangeError('Gate outside [0,1]');
    const decayed = old.map(row => row.map(x => alpha * x));
    const prediction = value.map((_, j) => key.reduce((sum, k, i) => sum + k * decayed[i][j], 0));
    const error = value.map((v, j) => v - prediction[j]);
    const correction = key.map(k => error.map(e => beta * k * e));
    const state = decayed.map((row, i) => row.map((x, j) => x + correction[i][j]));
    const output = value.map((_, j) => query.reduce((sum, q, i) => sum + q * state[i][j], 0));
    return { decayed, prediction, error, correction, state, output };
  }
  function causalConv(values, weights) {
    if (values.length !== weights.length || !values.length) throw new RangeError('Window mismatch');
    const weighted = values.reduce((sum, x, i) => sum + x * weights[i], 0);
    return { weighted, silu: weighted / (1 + Math.exp(-weighted)) };
  }
  function scheduleRounds(prompt, budget, decodes) {
    if (![prompt, budget, decodes].every(Number.isInteger) || prompt <= 0 || decodes < 0 || budget <= decodes) throw new RangeError('No prefill capacity');
    const available = budget - decodes, chunks = [];
    for (let left = prompt; left > 0;) { const n = Math.min(left, available); chunks.push(n); left -= n; }
    return { available, chunks };
  }
  function graphChoice(active, buckets) {
    const bucket = [...buckets].sort((a,b) => a-b).find(x => x >= active) ?? null;
    return { bucket, padding: bucket === null ? 0 : bucket-active, useful: active, fill: bucket === null ? 1 : active/bucket };
  }
  function memoryEstimate(requests, context, replicas) {
    const GiB = 1024 ** 3;
    return {
      weightGiB: replicas * 9e9 * 2 / GiB,
      kvGiB: requests * context * 8 * 2 * 4 * 256 * 2 / GiB,
      stateGiB: requests * 24 * 32 * 128 * 128 * 4 / GiB,
    };
  }
  function cacheBudget(allowed, nonCache, fixed) {
    return { fixed, automatic: Math.max(0, allowed-nonCache), headroom: allowed-nonCache-fixed };
  }
  return { deltaStep, causalConv, scheduleRounds, graphChoice, memoryEstimate, cacheBudget };
});
