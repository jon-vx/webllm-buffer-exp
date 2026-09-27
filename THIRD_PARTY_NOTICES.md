# Runtime sources

`vendor/runtimes.js` contains the unmodified WebLLM 0.2.85 source and its embedded dependencies, packaged as a JavaScript string for local-file loading.

- [WebLLM source](https://github.com/mlc-ai/web-llm)
- [Published bundle](https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm@0.2.85/lib/index.js)
- [Underlying web runtime](https://github.com/mlc-ai/web-runtime)
- [Apache 2.0 license](vendor/LICENSE.webllm.txt)

Stock source SHA-256:

```text
341bae95822bfee1d0fd6a0e6cd2db8613bb8edf809390ac142fba36ec17792c
```

`runtime-policy.js` derives the experiment's timing and validation runtimes from that source. It changes encoder size and command-buffer grouping, preserves the uniform-buffer cursor until original flush boundaries, and drains queued buffers at those boundaries. The validation version also records dispatches, pool accesses and batching counts. Timed completions have no tracing.

`vendor/encoder-build.js` records the derived source hashes. The browser verifies the hashes before importing a runtime.

All variants retain upstream copyright and dependency license notices. Model assets download separately from the URLs in WebLLM's model configuration and retain their own licenses.
