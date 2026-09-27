(() => {
'use strict';
function installEncoderPolicy(ctx) {
  ctx.encoderLimit = null;
  ctx.buffersPerSubmit = 1;
  ctx.encoderDispatchCount = 0;
  ctx.finishedBuffers = [];
  ctx.finishExperimentEncoder = function () {
    if (!this.pendingEncoder) return;
    this.finishedBuffers.push(this.pendingEncoder.finish());
    this.pendingEncoder = null;
    this.encoderDispatchCount = 0;
  };
  ctx.submitExperimentBuffers = function () {
    if (!this.finishedBuffers.length) return;
    this.device.queue.submit(this.finishedBuffers);
    this.finishedBuffers = [];
    this.pendingGPUToCPUCopyIsQueueTail = false;
  };
  ctx.splitExperimentEncoder = function () {
    this.finishExperimentEncoder();
    if (this.finishedBuffers.length >= this.buffersPerSubmit) this.submitExperimentBuffers();
  };
  ctx.flushCommands = function () {
    this.finishExperimentEncoder();
    this.submitExperimentBuffers();
    this.pendingDispatchCount = 0;
    this.encoderDispatchCount = 0;
  };
  globalThis.__WEBLLM_ENCODERS__ = {
    async configure(policy) {
      const match = typeof policy === 'string' && /^d(8|16|32)b(1|4)$/.exec(policy);
      if (policy !== 'control' && !match) throw Error('Unknown encoder policy');
      await ctx.sync();
      ctx.encoderLimit = match ? Number(match[1]) : null;
      ctx.buffersPerSubmit = match ? Number(match[2]) : 1;
    },
    resetStats() {}, getStats() { return null; },
    getDeviceInfo() {
      const i = ctx.device.adapterInfo || {};
      return { vendor:i.vendor||'', architecture:i.architecture||'', device:i.device||'', description:i.description||'' };
    },
  };
}

function installEncoderTrace(ctx) {
  let stats, identities, nextIdentity, inEncoder = 0, sinceBoundary = 0;
  const hash = (h, value) => {
    for (const c of String(value)) h = Math.imul(h ^ c.charCodeAt(0), 16777619) >>> 0;
    return h;
  };
  const reset = () => {
    if (ctx.pendingEncoder || ctx.finishedBuffers.length) throw Error('Cannot reset trace with pending work');
    identities = new WeakMap(); nextIdentity = 1; inEncoder = 0; sinceBoundary = 0;
    stats = { dispatches:0, dispatchHash:2166136261, uniformCalls:0, uniformHash:2166136261, identityHash:2166136261,
      uniformAllocations:0, boundaries:0, boundaryHash:2166136261, encoders:0, submitCalls:0, commandBuffers:0,
      maxDispatchesPerEncoder:0, maxBuffersPerSubmit:0, bufferHistogram:{}, dispatchHistogram:{}, readbackSubmissions:0 };
  };
  reset();
  const finish = ctx.finishExperimentEncoder, submit = ctx.submitExperimentBuffers, flush = ctx.flushCommands, uniform = ctx.getUniformFromPool;
  ctx.finishExperimentEncoder = function () {
    if (this.pendingEncoder) {
      stats.encoders++; stats.maxDispatchesPerEncoder = Math.max(stats.maxDispatchesPerEncoder, inEncoder);
      stats.dispatchHistogram[inEncoder] = (stats.dispatchHistogram[inEncoder] || 0) + 1;
      inEncoder = 0;
    }
    return finish.call(this);
  };
  ctx.submitExperimentBuffers = function () {
    const n = this.finishedBuffers.length;
    if (n) {
      stats.submitCalls++; stats.commandBuffers += n;
      stats.maxBuffersPerSubmit = Math.max(stats.maxBuffersPerSubmit, n);
      stats.bufferHistogram[n] = (stats.bufferHistogram[n] || 0) + 1;
    }
    return submit.call(this);
  };
  ctx.flushCommands = function () {
    stats.boundaries++; stats.boundaryHash = hash(stats.boundaryHash, `${sinceBoundary};`); sinceBoundary = 0;
    return flush.call(this);
  };
  ctx.getUniformFromPool = function (bytes) {
    const slot = this.pendingDispatchCount, previous = this.uniformBufferPool[slot], buffer = uniform.call(this, bytes);
    if (previous !== buffer) stats.uniformAllocations++;
    if (!identities.has(buffer)) identities.set(buffer, nextIdentity++);
    stats.uniformCalls++; stats.uniformHash = hash(stats.uniformHash, `${slot}:${bytes};`);
    stats.identityHash = hash(stats.identityHash, `${identities.get(buffer)}:${bytes};`);
    return buffer;
  };
  ctx.traceEncoderDispatch = function (name, x, y, z) {
    stats.dispatches++; inEncoder++; sinceBoundary++;
    stats.dispatchHash = hash(stats.dispatchHash, `${name}:${x}:${y}:${z};`);
  };
  ctx.traceReadbackSubmit = () => { stats.readbackSubmissions++; };
  Object.assign(globalThis.__WEBLLM_ENCODERS__, {
    resetStats: reset,
    getStats: () => ({ ...structuredClone(stats), pendingBuffers:ctx.finishedBuffers.length,
      poolSlots:ctx.uniformBufferPool.length, poolBytes:ctx.uniformBufferPoolSizes.reduce((a,b)=>a+b,0) }),
  });
}

function patchEncoderRuntime(source, trace = false) {
  const insert = (needle, code) => {
    if (source.split(needle).length !== 2) throw Error('Unexpected stock runtime patch anchor');
    source = source.replace(needle, () => needle + '\n' + code);
  };
  insert('this.memory = memory;\n\t\t            this.device = device;',
    `(${installEncoderPolicy.toString()})(this);${trace ? `\n(${installEncoderTrace.toString()})(this);` : ''}`);
  insert('compute.dispatchWorkgroups(workDim[0], workDim[1], workDim[2]);\n\t\t                    compute.end();',
    (trace ? 'this.traceEncoderDispatch(finfo.name, workDim[0], workDim[1], workDim[2]);\n' : '') +
    'if (this.encoderLimit !== null && ++this.encoderDispatchCount >= this.encoderLimit) this.splitExperimentEncoder();');
  if (trace) insert('this.device.queue.submit([copyCommands]);', 'this.traceReadbackSubmit();');
  return source;
}
globalThis.EncoderRuntime = Object.freeze({ installEncoderPolicy, installEncoderTrace, patchEncoderRuntime });
})();
