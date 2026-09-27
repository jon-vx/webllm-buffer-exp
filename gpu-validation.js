(() => {
  "use strict";
  function createGPUValidation(ctx) {
    if (!ctx?.device) throw Error("Cannot inspect the loaded WebGPU device");
    const device = ctx.device,
      errors = [];
    let closed = false,
      scopes = 0;
    const record = (e) =>
      errors.push(String(e.error?.message ?? e.message ?? e));
    device.addEventListener("uncapturederror", record);
    device.lost.then((info) => {
      if (!closed) errors.push(`Device lost: ${info.reason}: ${info.message}`);
    });
    const assertHealthy = () => {
      if (errors.length)
        throw Error(`WebGPU validation failed: ${errors.join("; ")}`);
    };
    const pop = () => {
      const pending = [];
      while (scopes) {
        scopes--;
        pending.push(device.popErrorScope());
      }
      return pending;
    };
    return {
      getDeviceInfo() {
        const i = device.adapterInfo ?? {};
        return {
          vendor: i.vendor ?? "",
          architecture: i.architecture ?? "",
          device: i.device ?? "",
          description: i.description ?? "",
        };
      },
      async prepareMeasurement() {
        await ctx.sync();
        assertHealthy();
        if (scopes) throw Error("GPU validation scope left open");
        for (const kind of ["validation", "out-of-memory", "internal"]) {
          device.pushErrorScope(kind);
          scopes++;
        }
      },
      async validateGPU() {
        const pendingEncoder = Boolean(ctx.pendingEncoder),
          pendingBuffers = ctx.finishedBuffers?.length ?? 0;
        await device.queue.onSubmittedWorkDone();
        const captured = await Promise.all(pop());
        for (const error of captured) if (error) record(error);
        await new Promise((resolve) => setTimeout(resolve, 0));
        assertHealthy();
        if (pendingEncoder || pendingBuffers)
          throw Error("Runtime left unsubmitted GPU work at completion");
        return { passed: true, pendingEncoder, pendingBuffers, errors: [] };
      },
      closeMonitor() {
        closed = true;
        device.removeEventListener("uncapturederror", record);
        for (const p of pop()) p.catch(() => {});
      },
    };
  }
  globalThis.GPUValidation = Object.freeze({ createGPUValidation });
})();
