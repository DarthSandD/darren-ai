/**
 * Darren Ai — in-browser detection engine.
 *
 * Runs a real AI-text classifier entirely in the visitor's browser using
 * Transformers.js (ONNX Runtime Web). No server, no API key, no GPU rental:
 * the compute happens on the user's own device, so static hosting (GitHub
 * Pages) is enough and it costs nothing to run.
 *
 * Model: onnx-community/tmr-ai-text-detector-ONNX
 *   - RoBERTa-base, MIT licensed
 *   - Trained on the RAID benchmark (11 adversarial attacks) with focal loss
 *     + self-hard-negative iterative mining — i.e. built for the hard cases,
 *     not just clean GPT output.
 *   - Quantized (~126 MB) — downloaded once, then served from browser cache.
 *
 * Backends, in order of preference:
 *   1. WebGPU  — fast, uses the visitor's GPU
 *   2. WASM    — universal fallback, runs on CPU
 */

const MODEL_ID = "onnx-community/tmr-ai-text-detector-ONNX";
const TRANSFORMERS_CDN = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.5";

/** Inject an import map so transformers.js resolves bare imports locally. */
function injectLocalImportMap() {
  const head = document.head;
  const prev = head.querySelector('script[type="importmap"]');
  if (prev) prev.remove();
  const map = {
    "imports": {
      "onnxruntime-common": `${LOCAL_BASE}onnxruntime-common/index.js`,
      "onnxruntime-web":     `${LOCAL_BASE}onnxruntime-web.mjs`,
    }
  };
  const script = document.createElement("script");
  script.type = "importmap";
  script.textContent = JSON.stringify(map);
  head.appendChild(script);
}

// When the app is packaged (Android/desktop) or served from a copy that ships
// the model, everything is loaded from local files — no network at all.
// We probe for the bundled model at startup and switch to it if found.
const LOCAL_BASE = "./vendor/";
const LOCAL_MODEL = "./models/tmr-ai-text-detector";
let _useLocal = null; // null = unknown

async function detectLocalBundle() {
  if (_useLocal !== null) return _useLocal;
  try {
    const r = await fetch(`${LOCAL_MODEL}/config.json`, { method: "HEAD" });
    _useLocal = r.ok;
  } catch {
    _useLocal = false;
  }
  return _useLocal;
}

let _classifier = null;
let _backend = null;
let _loading = null;

/**
 * Start loading the model in the background, before the user clicks anything.
 * Transformers.js stores model files in the browser's Cache Storage, so:
 *   - first visit:  downloads once (progress shown), then cached on disk
 *   - every revisit: loaded from cache — effectively instant, still offline-capable
 * This makes detection feel immediate without any server or cost.
 */
export function warmup(onProgress = () => {}) {
  // Defer so it never competes with first paint.
  const go = () => load(onProgress).catch(() => {});
  if (typeof requestIdleCallback === "function") requestIdleCallback(go, { timeout: 3000 });
  else setTimeout(go, 1200);
}

/** Split text into sentence-ish spans for the heatmap. */
export function splitSentences(text) {
  const out = [];
  const re = /[^.!?]+[.!?]*/g;
  let m;
  while ((m = re.exec(text)) !== null) {
    const s = m[0].trim();
    if (s.length >= 8) out.push({ text: s, start: m.index, end: m.index + m[0].length });
  }
  return out;
}

/**
 * Lazily load the model. Tries WebGPU, falls back to WASM.
 * @param {(msg:string, pct?:number)=>void} onProgress
 */
export async function load(onProgress = () => {}) {
  if (_classifier) return _classifier;
  if (_loading) return _loading;

  _loading = (async () => {
    onProgress("Loading detector engine…", 0);
    const useLocal = await detectLocalBundle();

    // Library: bundled copy when packaged, CDN otherwise.
    const libUrl = useLocal ? `${LOCAL_BASE}transformers.web.js` : TRANSFORMERS_CDN;
    if (useLocal) injectLocalImportMap();
    const { pipeline, env } = await import(libUrl);

    if (useLocal) {
      // Fully offline: point ORT at the bundled wasm and the local model dir.
      env.allowRemoteModels = false;
      env.allowLocalModels = true;
      env.localModelPath = "./models/";
      env.backends.onnx.wasm.wasmPaths = LOCAL_BASE;
    } else {
      env.allowLocalModels = false;
    }

    const progress_callback = (info) => {
      if (info.status === "progress" && info.total) {
        onProgress(`Downloading model: ${info.file ?? ""}`, Math.round((info.loaded / info.total) * 100));
      } else if (info.status === "ready") {
        onProgress("Model ready", 100);
      }
    };

    const modelRef = useLocal ? "tmr-ai-text-detector" : MODEL_ID;

    // 1) try WebGPU (fast) — 2) fall back to WASM (universal)
    try {
      _classifier = await pipeline("text-classification", modelRef, {
        device: "webgpu",
        dtype: useLocal ? "q8" : "q4",
        progress_callback,
      });
      _backend = "webgpu";
    } catch (e) {
      console.warn("WebGPU unavailable, using WASM:", e?.message || e);
      _classifier = await pipeline("text-classification", modelRef, {
        device: "wasm",
        dtype: "q8",
        progress_callback,
      });
      _backend = "wasm";
    }
    return _classifier;
  })();

  return _loading;
}

export function backend() { return _backend; }
export function isReady() { return !!_classifier; }

/** Normalize a pipeline output into P(AI). */
function aiProb(output) {
  // output: [{ label, score }, ...]
  const arr = Array.isArray(output) ? output : [output];
  for (const o of arr) {
    const lab = String(o.label || "").toLowerCase();
    // labels vary: "LABEL_1", "ai", "fake", "generated" ...
    if (lab.includes("ai") || lab.includes("fake") || lab.includes("generated") ||
        lab === "label_1" || lab === "label_2") {
      return o.score;
    }
  }
  // single-label models: score IS the positive class
  return arr[0]?.score ?? 0;
}

/**
 * Score text in the browser.
 * @returns {Promise<{score:number, verdict:string, confidence:string, engine:string, backend:string, spans:Array, caveats:string[]}>}
 */
export async function detect(text, { onProgress, threshold = 0.5 } = {}) {
  if (text.trim().split(/\s+/).length < 25) {
    return {
      score: 0, verdict: "insufficient_text", confidence: "low",
      engine: "browser (TMR-RoBERTa)", backend: _backend || "n/a", spans: [],
      caveats: ["Need ~25+ words for a reliable estimate."],
    };
  }

  const clf = await load(onProgress);
  const sentences = splitSentences(text);

  // Batch: whole doc first, then per-sentence (if not too long).
  const doSentences = sentences.length > 1 && text.split(/\s+/).length <= 900;
  const inputs = [text, ...(doSentences ? sentences.map((s) => s.text) : [])];

  let outputs;
  try {
    outputs = await clf(inputs, { top_k: null });
  } catch {
    // some builds dislike top_k:null — fall back to one-by-one
    outputs = [];
    for (const t of inputs) outputs.push(await clf(t));
  }

  const score = aiProb(outputs[0]);
  const spans = doSentences
    ? sentences.map((s, i) => {
        const sc = aiProb(outputs[i + 1]);
        return { text: s.text, start: s.start, end: s.end, score: +sc.toFixed(4), flagged: sc >= threshold };
      })
    : [];

  const verdict = score < 0.35 ? "human" : score < 0.6 ? "mixed" : "ai";
  return {
    score: +score.toFixed(4),
    verdict,
    confidence: score > 0.8 || score < 0.2 ? "high" : "medium",
    engine: "browser (TMR-RoBERTa)",
    backend: _backend,
    spans,
    caveats: [
      "Runs entirely on your device — your text never leaves this browser.",
      "TMR is trained on the RAID benchmark (adversarial); still probabilistic, not proof of authorship.",
      "Accuracy drops on paraphrased or humanized text — true of every detector.",
    ],
  };
}
