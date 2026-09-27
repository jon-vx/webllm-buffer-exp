(() => {
  "use strict";
  const CONFIG = {
    experimentId: "webllm-encoders-submissions-complete-v3",
    model: "SmolLM2-360M-Instruct-q4f16_1-MLC",
    rounds: 8,
    warmups: 1,
    maxTokens: 128,
    expectedOutputTokens: 122,
    contextWindowSize: 4096,
  };
  const BUILD = globalThis.ENCODER_BUILD;
  const PROTOCOL_REVISION = "audit-v3";
  const CAPS = [8, 16, 32];
  const CASES = [
    { id: "stock", label: "Stock WebLLM", policy: null },
    { id: "control", label: "Uncapped control", policy: "control" },
    ...CAPS.flatMap((dispatches) =>
      [1, 4].map((buffers) => ({
        id: `d${dispatches}b${buffers}`,
        label: `${dispatches} dispatches, ${buffers} buffer${buffers === 1 ? "" : "s"}`,
        policy: `d${dispatches}b${buffers}`,
        dispatchesPerEncoder: dispatches,
        commandBuffersPerSubmission: buffers,
      })),
    ),
  ];
  const expectedAnswer = Array.from(
    { length: 12 },
    (_, i) => `${i + 1}. Fixed inputs make careful comparisons possible.`,
  ).join("\n");

  const REQUEST = {
    messages: [
      { role: "system", content: "Follow the requested format exactly. Do not add an introduction or conclusion." },
      { role: "user", content: "Repeat these twelve lines exactly, keeping the numbering:\n" + expectedAnswer },
    ],
    temperature: 0,
    top_p: 1,
    seed: 42,
    repetition_penalty: 1,
    frequency_penalty: 0,
    presence_penalty: 0,
    max_tokens: CONFIG.maxTokens,
    ignore_eos: false,
    stream: true,
    stream_options: { include_usage: true },
  };

  function makeSchedule(seed, validation = false) {
    const rng = random(seed);
    const ids = CASES.map((c) => c.id);
    const base = shuffle(ids, rng);
    const first = [0, 1, 7, 2, 6, 3, 5, 4];
    const offsets = shuffle(
      ids.map((_, i) => i),
      rng,
    );
    const schedule = [];
    for (const [round, offset] of offsets
      .slice(0, validation ? 1 : CONFIG.rounds)
      .entries()) {
      first.forEach((index, position) =>
        schedule.push({
          index: schedule.length,
          round,
          promptId: "short",
          position,
          caseId: base[(index + offset) % ids.length],
        }),
      );
    }
    return schedule;
  }

  function createSession({ seed, validation = false, build = BUILD, device }) {
    const experimentId = validation
      ? "webllm-encoders-submissions-complete-check-v3"
      : CONFIG.experimentId;
    return {
      experimentId,
      protocolRevision: PROTOCOL_REVISION,
      sessionId: crypto.randomUUID(),
      createdAt: new Date().toISOString(),
      seed,
      profileId: validation ? "encoders-long-check" : "encoders-long",
      profile: {
        label: "Full answer",
        mode: validation ? "diagnostic" : "timing",
        rounds: validation ? 1 : CONFIG.rounds,
        warmups: CONFIG.warmups,
        maxTokens: CONFIG.maxTokens,
        stopMode: "natural",
        expectedOutputTokens: CONFIG.expectedOutputTokens,
        experimentId,
        scheduleDesign: "williams-v2",
      },
      model: CONFIG.model,
      contextWindowSize: CONFIG.contextWindowSize,
      build,
      device,
      cases: structuredClone(CASES),
      requests: { short: structuredClone(REQUEST) },
      schedule: makeSchedule(seed, validation),
      loads: [],
      runs: [],
      status: "running",
      error: null,
      expectedAnswer,
    };
  }

  function random(seed) {
    return () => {
      seed |= 0;
      seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function shuffle(values, rng) {
    const out = [...values];
    for (let i = out.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [out[i], out[j]] = [out[j], out[i]];
    }
    return out;
  }
  function median(values) {
    if (!values.length) return null;
    const sorted = [...values].sort((a, b) => a - b),
      mid = sorted.length >> 1;
    return sorted.length % 2
      ? sorted[mid]
      : (sorted[mid - 1] + sorted[mid]) / 2;
  }
  async function deadline(task, signal, timeoutMs, interrupt = () => {}) {
    let timer, abort;
    signal?.throwIfAborted();
    try {
      return await Promise.race([
        Promise.resolve().then(task),
        new Promise((_, reject) => {
          abort = () => reject(new Error("Stopped by participant"));
          signal?.addEventListener("abort", abort, { once: true });
          if (signal?.aborted) abort();
          timer = setTimeout(
            () =>
              reject(
                new Error(
                  `Timed out after ${Math.round(timeoutMs / 1000)} seconds`,
                ),
              ),
            timeoutMs,
          );
        }),
      ]);
    } catch (error) {
      Promise.resolve()
        .then(interrupt)
        .catch(() => {});
      throw error;
    } finally {
      clearTimeout(timer);
      signal?.removeEventListener("abort", abort);
    }
  }

  function outputValid(row) {
    return (
      row.finishReason === "stop" &&
      typeof row.output === "string" &&
      row.output.trim() === expectedAnswer &&
      row.generatedTokenIds?.length === CONFIG.expectedOutputTokens &&
      row.usage?.completion_tokens === CONFIG.expectedOutputTokens
    );
  }
  function equivalent(a, b) {
    const clean = (r) =>
      r?.success &&
      !r.hidden &&
      !r.error &&
      r.wallMs > 0 &&
      r.ttftMs > 0 &&
      r.ttftMs <= r.wallMs &&
      Number.isFinite(r.wallMs) &&
      Number.isFinite(r.afterFirstMs) &&
      Math.abs(r.afterFirstMs - (r.wallMs - r.ttftMs)) < 0.01 &&
      (!r.gpuCheck || r.gpuCheck.passed === true) &&
      outputValid(r);
    return Boolean(
      clean(a) &&
        clean(b) &&
        a.promptId === b.promptId &&
        a.output === b.output &&
        a.usage.prompt_tokens === b.usage.prompt_tokens &&
        a.generatedTokenIds.length === b.generatedTokenIds.length &&
        a.generatedTokenIds.every((id, i) => id === b.generatedTokenIds[i]),
    );
  }
  function compare(session, caseId) {
    const rows = session.runs.filter(
      (r) => r.phase === "measured" && r.promptId === "short",
    );
    const unique = (id, round) =>
      rows.filter((r) => r.caseId === id && r.round === round);
    const all = Array.from({ length: session.profile.rounds }, (_, round) => {
      const aa = unique(caseId, round),
        bb = unique("stock", round);
      return [
        aa.length === 1 ? aa[0] : undefined,
        bb.length === 1 ? bb[0] : undefined,
        round,
      ];
    });
    const pairs = all.filter(([a, b]) => equivalent(a, b));
    return {
      promptId: "short",
      caseId,
      baselineId: "stock",
      pairs: pairs.length,
      expected: session.profile.rounds,
      wallMs: median(pairs.map(([a]) => a.wallMs)),
      ttftMs: median(pairs.map(([a]) => a.ttftMs)),
      afterFirstMs: median(pairs.map(([a]) => a.afterFirstMs)),
      wallRatio: median(pairs.map(([a, b]) => a.wallMs / b.wallMs)),
      ttftRatio: median(pairs.map(([a, b]) => a.ttftMs / b.ttftMs)),
      afterFirstRatio: median(
        pairs
          .filter(([, b]) => b.afterFirstMs > 0)
          .map(([a, b]) => a.afterFirstMs / b.afterFirstMs),
      ),
      pairsRaw: all.map(([a, b, round]) => ({
        round,
        valid: equivalent(a, b),
        candidateMs: a?.wallMs ?? null,
        baselineMs: b?.wallMs ?? null,
      })),
    };
  }
  function encoderChecks(session) {
    const rows = session.runs.filter((r) => r.phase === "measured");
    const ref = rows.find((r) => r.caseId === "control"),
      stock = rows.find((r) => r.caseId === "stock");
    const keys = [
      "dispatches",
      "dispatchHash",
      "uniformCalls",
      "uniformHash",
      "identityHash",
      "uniformAllocations",
      "poolSlots",
      "poolBytes",
      "boundaries",
      "boundaryHash",
      "readbackSubmissions",
    ];
    return CASES.filter((c) => c.id !== "stock").map((c) => {
      const row = rows.find((r) => r.caseId === c.id),
        stats = row?.stats,
        base = ref?.stats;
      const outputMatch = equivalent(row, stock) && equivalent(ref, stock);
      const traceMatch = Boolean(
        stats?.dispatches > 0 &&
          base?.dispatches > 0 &&
          keys.every((k) => stats[k] === base[k]),
      );
      const balanced = Boolean(
        stats &&
          stats.encoders === stats.commandBuffers &&
          stats.pendingBuffers === 0,
      );
      const limitOK = Boolean(
        stats &&
          (c.id === "control" ||
            stats.maxDispatchesPerEncoder === c.dispatchesPerEncoder) &&
          stats.maxBuffersPerSubmit === (c.commandBuffersPerSubmission ?? 1),
      );
      const one = rows.find(
        (r) => r.caseId === `d${c.dispatchesPerEncoder}b1`,
      )?.stats;
      const submissionEffect =
        c.commandBuffersPerSubmission !== 4 ||
        Boolean(
          one &&
            stats?.encoders === one.encoders &&
            stats?.submitCalls < one.submitCalls,
        );
      const count = (h) => Object.values(h ?? {}).reduce((a, b) => a + b, 0),
        weighted = (h) =>
          Object.entries(h ?? {}).reduce((a, [k, v]) => a + Number(k) * v, 0);
      const countsConsistent = Boolean(
        stats &&
          stats.uniformCalls === stats.dispatches &&
          count(stats.dispatchHistogram) === stats.encoders &&
          weighted(stats.dispatchHistogram) === stats.dispatches &&
          count(stats.bufferHistogram) === stats.submitCalls &&
          weighted(stats.bufferHistogram) === stats.commandBuffers,
      );
      return {
        caseId: c.id,
        outputMatch,
        traceMatch,
        balanced,
        limitOK,
        submissionEffect,
        countsConsistent,
        passed:
          outputMatch &&
          traceMatch &&
          balanced &&
          limitOK &&
          submissionEffect &&
          countsConsistent,
        stats: stats ?? null,
      };
    });
  }
  const LIMITS = { load: 600000, completion: 180000, unload: 15000 };
  async function runInference(session, createEngine, hooks = {}) {
    const now = hooks.now ?? (() => performance.now()),
      check = () => hooks.signal?.throwIfAborted();
    const checkpoint = () => hooks.checkpoint?.(session);
    try {
      for (const entry of session.schedule) {
        check();
        await hooks.waitUntilVisible?.();
        check();
        const variant = session.cases.find((c) => c.id === entry.caseId),
          load = { ...entry, success: false, error: null };
        session.loads.push(load);
        let handle;
        hooks.onLoad?.(entry);
        try {
          const started = now(),
            loading = Promise.resolve().then(() =>
              createEngine(variant, session.profile.mode),
            );
          try {
            handle = await deadline(
              () => loading,
              hooks.signal,
              hooks.loadTimeoutMs ?? LIMITS.load,
            );
          } catch (error) {
            loading.then((h) => h.engine.unload()).catch(() => {});
            throw error;
          }
          load.loadMs = now() - started;
          load.gpuVendor = await deadline(
            () => handle.engine.getGPUVendor(),
            hooks.signal,
            10000,
          );
          load.bundle = handle.bundle;
          load.sha256 = handle.sha256;
          if (
            session.build &&
            load.sha256 !== session.build[load.bundle]?.sha256
          )
            throw Error(
              "Engine runtime checksum does not match session manifest",
            );
          const expectedBundle =
            variant.id === "stock"
              ? "stock"
              : session.profile.mode === "diagnostic"
                ? "encoderTrace"
                : "encoder";
          if (load.bundle !== expectedBundle)
            throw Error("Wrong runtime bundle for this case");
          if (
            session.protocolRevision === PROTOCOL_REVISION &&
            (!handle.prepareMeasurement || !handle.validateGPU)
          )
            throw Error("GPU validation hooks missing");
          load.modelConfig = handle.modelConfig ?? null;
          load.gpu = handle.getDeviceInfo?.() ?? null;
          if (variant.id === "stock") {
            if (handle.control)
              throw new Error("Untouched stock must not have a controller");
          } else {
            if (!handle.control)
              throw new Error("Mechanism controller unavailable");
            await deadline(
              () => handle.control.configure(variant.policy),
              hooks.signal,
              10000,
            );
            load.gpu =
              handle.getDeviceInfo?.() ??
              handle.control.getDeviceInfo?.() ??
              null;
          }
          load.success = true;
          for (
            let repetition = 0;
            repetition <= CONFIG.warmups;
            repetition++
          ) {
            check();
            await hooks.waitUntilVisible?.();
            check();
            const row = {
              ...entry,
              repetition,
              phase:
                repetition < CONFIG.warmups ? "warmup" : "measured",
              success: false,
              error: null,
              hidden: false,
              output: "",
              usage: {},
              finishReason: null,
              wallMs: null,
              ttftMs: null,
              afterFirstMs: null,
              stats: null,
            };
            hooks.onRunStart?.(row);
            let active = true;
            try {
              await deadline(
                async () => {
                  const prep = now();
                  await handle.engine.resetChat();
                  if (!active) return;
                  row.resetMs = now() - prep;
                  const prepare = now();
                  await handle.prepareMeasurement?.();
                  if (!active) return;
                  await hooks.prepareMeasurement?.();
                  if (!active) return;
                  row.prepareMs = now() - prepare;
                  handle.control?.resetStats();
                  const request = structuredClone(REQUEST);
                  row.startedAt = new Date().toISOString();
                  const start = now();
                  const stream =
                    await handle.engine.chat.completions.create(request);
                  if (!active) return;
                  for await (const chunk of stream) {
                    if (!active) return;
                    const choice = chunk.choices?.[0],
                      content = choice?.delta?.content;
                    if (content) {
                      if (row.ttftMs === null) row.ttftMs = now() - start;
                      row.output += content;
                    }
                    if (choice?.finish_reason)
                      row.finishReason = choice.finish_reason;
                    if (chunk.usage)
                      row.usage = {
                        prompt_tokens: chunk.usage.prompt_tokens,
                        completion_tokens: chunk.usage.completion_tokens,
                      };
                  }
                  if (!active) return;
                  row.wallMs = now() - start;
                  row.afterFirstMs =
                    row.ttftMs === null ? null : row.wallMs - row.ttftMs;
                  row.endedAt = new Date().toISOString();
                  row.stats = handle.control?.getStats() ?? null;
                  const metrics = handle.getMetrics();
                  row.generatedTokenIds = Array.from(metrics.tokenIds);
                  row.runtimePrefillMs = metrics.prefillMs;
                  row.runtimeDecodeMs = metrics.decodeMs;
                  const validationStart = now();
                  if (handle.validateGPU)
                    row.gpuCheck = await handle.validateGPU();
                  if (!active) return;
                  row.gpuValidationMs = now() - validationStart;
                  const promptOK =
                    Number.isInteger(row.usage.prompt_tokens) &&
                    row.usage.prompt_tokens > 0 &&
                    row.usage.prompt_tokens + CONFIG.maxTokens <= CONFIG.contextWindowSize;
                  const lengthOK = row.usage.prompt_tokens < 512;
                  row.success = Boolean(
                    row.output &&
                      Number.isFinite(row.wallMs) &&
                      row.ttftMs > 0 &&
                      row.wallMs >= row.ttftMs &&
                      (!row.gpuCheck || row.gpuCheck.passed === true) &&
                      outputValid(row) &&
                      promptOK &&
                      lengthOK,
                  );
                  if (!row.success)
                    throw new Error(
                      "Output validation failed: expected twelve lines, 122 text tokens and natural stopping. Check GPU and timing fields.",
                    );
                },
                hooks.signal,
                hooks.runTimeoutMs ?? LIMITS.completion,
                () => handle.engine.interruptGenerate(),
              );
            } catch (error) {
              row.success = false;
              row.error = String(error.message || error);
            } finally {
              active = false;
            }
            row.hidden = Boolean(hooks.wasHidden?.());
            if (row.hidden) {
              row.success = false;
              row.error =
                "Tab was hidden during this completion; session stopped to avoid selective exclusions.";
            }
            const saved = structuredClone(row);
            session.runs.push(saved);
            hooks.onRunEnd?.(saved);
            await checkpoint();
            if (!saved.success) throw new Error(saved.error);
          }
        } catch (error) {
          load.error = String(error.message || error);
          throw error;
        } finally {
          if (handle) {
            const started = now();
            try {
              handle.closeMonitor?.();
              await deadline(
                () => handle.engine.unload(),
                undefined,
                hooks.unloadTimeoutMs ?? LIMITS.unload,
              );
            } catch (error) {
              load.error = load.error
                ? `${load.error}; GPU cleanup also failed: ${error.message}`
                : `GPU cleanup failed: ${error.message}`;
              throw new Error(load.error);
            } finally {
              load.unloadMs = now() - started;
            }
          }
          await checkpoint();
        }
      }
      session.status = "complete";
    } catch (error) {
      session.status = hooks.signal?.aborted ? "stopped" : "failed";
      session.error = String(error.message || error);
    }
    session.endedAt = new Date().toISOString();
    await checkpoint();
    return session;
  }

  globalThis.Experiment = Object.freeze({
    CONFIG,
    REQUEST,
    BUILD,
    CASES,
    PROTOCOL_REVISION,
    makeSchedule,
    createSession,
    compare,
    runInference,
    deadline,
    encoderChecks,
    expectedAnswer,
    outputValid,
  });
})();
