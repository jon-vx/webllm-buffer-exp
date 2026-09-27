(() => {
  "use strict";
  const UPLOAD_URL = "https://webllm-results.runtime-orch-cs.workers.dev";
  const {
    CONFIG,
    BUILD,
    CASES,
    encoderChecks,
    createSession,
    compare,
    runInference,
    deadline,
  } = Experiment;
  const $ = (id) => document.getElementById(id);
  const storageKey = "standalone-" + CONFIG.experimentId;
  const TOTAL_COMPLETIONS = CASES.length * (CONFIG.rounds + 1) * (CONFIG.warmups + 1);
  const supported = isSecureContext && Boolean(navigator.gpu) && Boolean(globalThis.QUICK_RUNTIME_SOURCES);
  const completed = () => (session?.preflight?.runs.length ?? 0) + (session?.runs.length ?? 0);
  let session, controller, loadingLabel;
  let uploading = false;
  let busy = false,
    active = false,
    hidden = false,
    needsReload = false;

  function save() {
    try {
      localStorage.setItem(storageKey, JSON.stringify(session));
    } catch { }
  }

  function render() {
    $("go").hidden = busy || uploading;
    $("go").disabled = !supported;
    $("go").textContent = needsReload ? "Reload to run again" : "Run test";
    $("stop").hidden = !busy;
    $("stop").disabled = Boolean(controller?.signal.aborted);
    $("device-label").disabled = busy || uploading || needsReload;
    $("download").hidden = !session || busy;
    $("upload").hidden = !UPLOAD_URL || session?.status !== "complete" || session?.upload?.status === "complete";
    $("upload").disabled = busy || uploading;
    $("upload").textContent = session?.upload?.status === "failed" ? "Retry upload" : "Submit results";
    $("progress").hidden = !session;
    $("progress").max = TOTAL_COMPLETIONS;
    $("progress").value = completed();
  }

  function showOutcome() {
    if (!session) return;
    if (session.status === "complete") {
      $("status").textContent = session.upload?.status === "complete"
        ? "Results submitted. Thank you!"
        : "Test complete. Thank you!";
      if (!UPLOAD_URL && session.upload?.status !== "complete") {
        $("upload-status").textContent = "Submission is not configured.";
      }
    } else if (session.status === "stopped" || session.status === "interrupted") {
      $("status").textContent = "Test stopped.";
    } else if (session.error?.includes("Tab was hidden")) {
      $("status").textContent = "The tab was hidden, so the test stopped. Keep it visible and try again.";
    } else {
      $("status").textContent = "The test could not finish. Please try again.";
    }
  }

  function printResults() {
    if (!session) return;
    const snapshot = structuredClone(session);
    globalThis.webllmResults = snapshot;
    const comparisons = CASES.map(c => compare(snapshot, c.id));
    const round = value => value == null ? null : Number(value.toFixed(2));
    console.log("Compared with stock WebLLM. Negative change means less time.");
    console.table(comparisons.map((comparison, index) => ({
      Configuration: CASES[index].label,
      "Valid pairs": `${comparison.pairs}/${comparison.expected}`,
      "Total ms": round(comparison.wallMs),
      "Total change %": round(comparison.wallRatio == null ? null : (comparison.wallRatio - 1) * 100),
      "First text ms": round(comparison.ttftMs),
      "First text change %": round(comparison.ttftRatio == null ? null : (comparison.ttftRatio - 1) * 100),
      "After first text ms": round(comparison.afterFirstMs),
      "After first text change %": round(comparison.afterFirstRatio == null ? null : (comparison.afterFirstRatio - 1) * 100),
    })));
    console.log("Prompt", snapshot.requests.short.messages);
    console.log("Stock answer", snapshot.runs.find(r => r.caseId === "stock" && r.phase === "measured")?.output ?? "No measured stock answer.");
    console.log("Full results (also available as window.webllmResults)", snapshot);
    if (snapshot.error) console.error(snapshot.error);
    if (snapshot.status === "complete" && comparisons.some(c => c.pairs !== c.expected)) {
      console.warn("Some comparisons have mismatched or invalid outputs. Check valid pair counts.");
    }
  }

  async function verifiedRuntime(id) {
    const stock = globalThis.QUICK_RUNTIME_SOURCES.stock;
    const source =
      id === "stock"
        ? stock
        : EncoderRuntime.patchEncoderRuntime(stock, id === "encoderTrace");
    const bytes = new TextEncoder().encode(source);
    const digest = await crypto.subtle.digest("SHA-256", bytes);
    const hash = [...new Uint8Array(digest)]
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");
    if (hash !== BUILD[id].sha256)
      throw Error(`${id} runtime checksum mismatch`);
    controller.signal.throwIfAborted();
    const url = URL.createObjectURL(
      new Blob([bytes], { type: "text/javascript" }),
    );
    try {
      return await import(url);
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  async function createEngine(modules, variant, mode) {
    const bundle =
      variant.id === "stock"
        ? "stock"
        : mode === "diagnostic"
          ? "encoderTrace"
          : "encoder";
    const runtime = modules[bundle];
    globalThis.__WEBLLM_ENCODERS__ = undefined;
    const engine = await runtime.CreateMLCEngine(
      CONFIG.model,
      {
        appConfig: { ...runtime.prebuiltAppConfig, cacheBackend: "cache" },
        initProgressCallback: (p) => {
          if (busy && !controller.signal.aborted) {
            const percent = Number.isFinite(p.progress) ? ` ${Math.round(p.progress * 100)}%` : "";
            $("status").textContent = loadingLabel + percent;
          }
        },
      },
      { context_window_size: CONFIG.contextWindowSize },
    );
    const pipeline = engine.loadedModelIdToPipeline.get(CONFIG.model);
    let monitor;
    try {
      monitor = GPUValidation.createGPUValidation(
        pipeline.tvm.lib.webGPUContext,
      );
    } catch (error) {
      await engine.unload();
      throw error;
    }
    return {
      engine,
      ...monitor,
      bundle,
      sha256: BUILD[bundle].sha256,
      control: variant.id === "stock" ? null : globalThis.__WEBLLM_ENCODERS__,
      modelConfig: {
        record: structuredClone(
          runtime.prebuiltAppConfig.model_list.find(
            (m) => m.model_id === CONFIG.model,
          ),
        ),
        resolved: structuredClone(engine.loadedModelIdToChatConfig.get(CONFIG.model)),
      },
      getMetrics() {
        return {
          tokenIds: pipeline.outputIds,
          prefillMs: pipeline.getCurRoundPrefillTotalTime() * 1000,
          decodeMs: pipeline.getCurRoundDecodingTotalTime() * 1000,
        };
      },
    };
  }

  async function waitUntilVisible() {
    if (!document.hidden || controller.signal.aborted) return;
    $("status").textContent = "Paused. Return to this tab to continue.";
    await new Promise((resolve) => {
      const check = () => {
        if (!document.hidden || controller.signal.aborted) {
          document.removeEventListener("visibilitychange", check);
          controller.signal.removeEventListener("abort", check);
          resolve();
        }
      };
      document.addEventListener("visibilitychange", check);
      controller.signal.addEventListener("abort", check);
      check();
    });
  }

  function runStage(target, factory) {
    return runInference(target, factory, {
      signal: controller.signal,
      waitUntilVisible,
      prepareMeasurement: () => new Promise(resolve => setTimeout(resolve, 50)),
      onLoad() {
        loadingLabel = completed() === 0 ? "Downloading and preparing model…" : "Preparing next test…";
        $("status").textContent = loadingLabel;
      },
      onRunStart() {
        active = true;
        hidden = document.hidden;
        $("status").textContent = `Testing… ${completed()} of ${TOTAL_COMPLETIONS} completions.`;
      },
      wasHidden: () => hidden,
      onRunEnd() {
        active = false;
        $("status").textContent = `Testing… ${completed()} of ${TOTAL_COMPLETIONS} completions.`;
      },
      checkpoint() { save(); render(); },
    });
  }

  async function uploadResults() {
    if (!UPLOAD_URL || uploading || busy || session?.status !== "complete" || session.upload?.status === "complete") return;
    uploading = true;
    $("go").disabled = true;
    session.upload = { status: "uploading" };
    $("upload-status").textContent = "Submitting results…";
    save(); render();
    try {
      const response = await fetch(UPLOAD_URL, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(session), signal: AbortSignal.timeout(120000),
      });
      const result = await response.json();
      if (!response.ok || result.ok !== true || result.sessionId !== session.sessionId
        || result.attachment !== true || typeof result.recordId !== "string") {
        throw Error(result.error || "Upload was not confirmed.");
      }
      session.upload = { status: "complete", completedAt: new Date().toISOString() };
      $("upload-status").textContent = "";
      showOutcome();
    } catch (error) {
      session.upload = { status: "failed", error: String(error.message || error) };
      $("upload-status").textContent = "Submission failed. Please retry.";
    } finally {
      uploading = false;
      save(); render();
    }
  }

  async function validateExperiment(factory) {
    session.preflight = createSession({
      seed: session.seed,
      validation: true,
      device: session.device,
    });
    await runStage(session.preflight, factory);
    session.preflight.checks = encoderChecks(session.preflight);
    save();
    render();
    if (
      session.preflight.status !== "complete" ||
      !session.preflight.checks.every((c) => c.passed)
    ) {
      throw Error(
        session.preflight.error ||
        "Validation failed. Timing was not started.",
      );
    }
  }

  async function run() {
    if (needsReload) {
      location.reload();
      return;
    }
    if (busy) return;
    busy = true;
    controller = new AbortController();
    $("go").disabled = true;
    $("stop").disabled = false;
    $("device-label").disabled = true;
    $("upload-status").textContent = "";
    try {
      session = createSession({
        seed: crypto.getRandomValues(new Uint32Array(1))[0],
        device: {
          userAgent: navigator.userAgent,
          platform: navigator.platform,
          label: $("device-label").value.trim(),
        },
      });
      session.deployment = {
        runner: "standalone-audit-v3",
        protocol: location.protocol,
        runtimeTransport: "verified-blob",
        cacheBackend: "cache",
      };
      session.measurementPreparation = {
        gpuResetFence: true,
        uiYieldMs: 50,
        gpuErrorScopes: true,
      };
      save();
      render();
      $("status").textContent = "Preparing test…";
      const modules = {};
      for (const id of ["stock", "encoder", "encoderTrace"]) {
        modules[id] = await deadline(
          () => verifiedRuntime(id),
          controller.signal,
          60000,
        );
      }
      const factory = (variant, mode) => createEngine(modules, variant, mode);
      await validateExperiment(factory);
      await runStage(session, factory);
    } catch (error) {
      if (session?.status === "running") {
        session.status = controller.signal.aborted ? "stopped" : "failed";
        session.error = String(error.message || error);
        session.endedAt = new Date().toISOString();
        save();
      }
      $("status").textContent = String(error.message || error);
    } finally {
      active = false;
      busy = false;
      needsReload = true;
      render();
      showOutcome();
      printResults();
    }
    if (session?.status === "complete") await uploadResults();
  }

  function restore() {
    try {
      const saved = JSON.parse(localStorage.getItem(storageKey) || "null");
      if (saved?.experimentId === CONFIG.experimentId) {
        session = saved;
        if (session.status === "running") {
          session.status = "interrupted";
          session.error = "Page closed during a run; cannot resume.";
          if (session.preflight?.status === "running")
            session.preflight.status = "interrupted";
          save();
        }
        if (session.upload?.status === "uploading") session.upload.status = "failed";
        if (session.upload?.status === "failed") $("upload-status").textContent = "Results have not been submitted. Please retry.";
        showOutcome();
        printResults();
      }
    } catch { }
    render();
  }

  $("go").onclick = run;
  $("upload").onclick = uploadResults;
  $("stop").onclick = () => {
    controller?.abort();
    $("status").textContent = "Stopping…";
    render();
  };
  $("download").onclick = () => {
    if (!session || busy) return;
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(session, null, 2)], {
        type: "application/json",
      }),
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = `webllm-buffer-exp-${session.sessionId}.json`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  document.addEventListener("visibilitychange", () => {
    if (active && document.hidden) hidden = true;
  });
  window.addEventListener("beforeunload", (event) => {
    if (busy || uploading) {
      event.preventDefault();
      event.returnValue = "";
    }
  });
  restore();
  if (!supported) {
    $("status").textContent = "This browser cannot run the test. WebGPU support is required.";
  } else if (!session) {
    $("status").textContent = "Ready when you are.";
  }
  $("data-note").textContent = UPLOAD_URL
    ? "GPU/browser information, timings and output from a fixed test prompt are sent to our Airtable when the test finishes."
    : "The test records GPU/browser information, timings and output from a fixed prompt.";
})();
