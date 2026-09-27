/* Darren Ai front-end.
   Detection strategy (in order):
     1. Local/remote backend API (LAPD / Pangram / whatever is configured)
     2. In-browser model (Transformers.js) — used automatically when no
        backend is reachable, e.g. on static GitHub Pages hosting.
   API base resolves automatically: same-origin on web, or a configured host
   for the desktop/android wrappers (window.DARREN_API). */
const API = (new URLSearchParams(location.search).get("api")
  || window.DARREN_API || "").replace(/\/$/, "");
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

let browserEngine = null;      // lazily imported module
let backendAvailable = null;   // null=unknown, true/false
async function getBrowserEngine() {
  if (!browserEngine) browserEngine = await import("./engine-browser.js");
  return browserEngine;
}

/* ---------- theme ---------- */
const root = document.documentElement;
const savedTheme = localStorage.getItem("tl-theme");
if (savedTheme) root.setAttribute("data-theme", savedTheme);
else if (matchMedia("(prefers-color-scheme: dark)").matches) root.setAttribute("data-theme", "dark");
$("#themeBtn").addEventListener("click", () => {
  const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
  root.setAttribute("data-theme", next);
  localStorage.setItem("tl-theme", next);
  $("#themeBtn").textContent = next === "dark" ? "☀️" : "🌙";
});
$("#themeBtn").textContent = root.getAttribute("data-theme") === "dark" ? "☀️" : "🌙";

/* ---------- helpers ---------- */
function toast(msg, ms = 2200) {
  const t = $("#toast"); t.textContent = msg; t.classList.add("show");
  clearTimeout(t._t); t._t = setTimeout(() => t.classList.remove("show"), ms);
}
const countWords = (s) => (s.trim() ? s.trim().split(/\s+/).length : 0);
function busy(btn, on, label) {
  btn.disabled = on;
  $(".spinner", btn).hidden = !on;
  if (label) $(".lbl", btn).textContent = label;
}
async function api(path, opts) {
  const res = await fetch(API + path, opts);
  if (!res.ok) {
    let d = ""; try { d = (await res.json()).detail; } catch {}
    throw new Error(d || `HTTP ${res.status}`);
  }
  return res;
}

/* ---------- tabs ---------- */
$$(".tab").forEach(tab => tab.addEventListener("click", () => {
  $$(".tab").forEach(t => { t.classList.toggle("active", t === tab); t.setAttribute("aria-selected", t === tab); });
  const mode = tab.dataset.mode;
  $("#panel-detect").hidden = mode !== "detect";
  $("#panel-humanize").hidden = mode !== "humanize";
}));

/* ---------- counters ---------- */
const input = $("#input"), hInput = $("#hInput");
input.addEventListener("input", () => { $("#wc").textContent = countWords(input.value) + " words"; $("#cc").textContent = input.value.length + " chars"; });
hInput.addEventListener("input", () => { $("#hwc").textContent = countWords(hInput.value) + " words"; });
$("#pasteBtn").addEventListener("click", async () => {
  try { input.value = await navigator.clipboard.readText(); input.dispatchEvent(new Event("input")); }
  catch { toast("Clipboard blocked — paste manually"); }
});
$("#clearBtn").addEventListener("click", () => { input.value = ""; input.dispatchEvent(new Event("input")); resetResults(); });
$("#fileInput").addEventListener("change", async (e) => {
  const f = e.target.files[0]; if (!f) return;
  input.value = await f.text(); input.dispatchEvent(new Event("input"));
  toast(`Loaded ${f.name}`);
});
$$(".range input").forEach(r => r.addEventListener("input", () => {
  $(`#${r.id}Out`).textContent = r.value;
}));

function resetResults() {
  $("#heatCard").hidden = true;
  $("#gaugeFill").style.strokeDashoffset = 527.8;
  $("#scorePct").textContent = "—"; $("#verdictText").textContent = "awaiting input";
  $("#mEngine").textContent = $("#mConf").textContent = $("#mFlag").textContent = "—";
  $("#caveats").hidden = true;
}

/* ---------- gauge ---------- */
const CIRC = 2 * Math.PI * 84;
$("#gaugeFill").style.strokeDasharray = CIRC;
function setGauge(score) {
  const off = CIRC * (1 - Math.max(0, Math.min(1, score)));
  $("#gaugeFill").style.strokeDashoffset = off;
  const color = score < 0.35 ? "var(--good)" : score < 0.6 ? "var(--warn)" : "var(--bad)";
  $("#gaugeFill").style.stroke = color;
  $("#scorePct").textContent = Math.round(score * 100) + "%";
  $("#scorePct").style.color = color;
}

/* ---------- detect ---------- */
$("#detectBtn").addEventListener("click", async () => {
  const text = input.value.trim();
  if (countWords(text) < 5) return toast("Add some text first");
  busy($("#detectBtn"), true, "Detecting…");
  try {
    const d = await runDetect(text);
    renderDetect(d);
  } catch (e) {
    toast("Error: " + e.message);
  } finally {
    busy($("#detectBtn"), false, "Detect AI");
  }
});

/** Try the backend; if it isn't reachable, fall back to in-browser inference. */
async function runDetect(text) {
  if (backendAvailable !== false) {
    try {
      const res = await api("/detect", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      backendAvailable = true;
      updateEnginePill("backend");
      return await res.json();
    } catch (e) {
      backendAvailable = false;
      console.warn("Backend unavailable, using in-browser engine:", e.message);
      toast("Backend offline — running on-device model", 3200);
    }
  }

  // in-browser fallback (no server needed)
  const be = await getBrowserEngine();
  const d = await be.detect(text, {
    onProgress: (msg, pct) => {
      busy($("#detectBtn"), true, pct ? `Downloading model ${pct}%` : msg);
    },
  });
  updateEnginePill("browser", d.backend);
  return d;
}

function renderDetect(d) {
  setGauge(d.score);
  $("#verdictText").textContent = (d.verdict || "").replace(/_/g, " ");
  $("#mEngine").textContent = d.engine;
  $("#mConf").textContent = d.confidence;
  const flagged = (d.spans || []).filter(s => s.flagged).length;
  $("#mFlag").textContent = (d.spans || []).length ? `${flagged}/${d.spans.length}` : "—";

  const cav = $("#caveats");
  if (d.caveats?.length) { cav.innerHTML = "<ul>" + d.caveats.map(c => `<li>${c}</li>`).join("") + "</ul>"; cav.hidden = false; }
  else cav.hidden = true;

  renderHeat(d.spans || []);
}

function renderHeat(spans) {
  const card = $("#heatCard"), heat = $("#heat");
  if (!spans.length) { card.hidden = true; return; }
  heat.innerHTML = spans.map(s => {
    const cls = s.score < 0.35 ? "h1" : s.score < 0.6 ? "h2" : "h3";
    const title = `${Math.round(s.score * 100)}% AI-like`;
    return `<span class="${cls}" title="${title}">${escapeHtml(s.text)}</span>`;
  }).join(" ");
  card.hidden = false;
}

function escapeHtml(s) { return s.replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

/* ---------- jump to humanize ---------- */
$("#toHumanize").addEventListener("click", () => {
  hInput.value = input.value; hInput.dispatchEvent(new Event("input"));
  $(".tab[data-mode=humanize]").click();
  hInput.scrollIntoView({ behavior: "smooth", block: "center" });
});

/* ---------- humanize (SSE) ---------- */
$("#humanizeBtn").addEventListener("click", async () => {
  const text = hInput.value.trim();
  if (countWords(text) < 5) return toast("Add some text first");
  busy($("#humanizeBtn"), true, "Humanizing…");
  const loopCard = $("#loopCard"), loopList = $("#loopList");
  loopList.innerHTML = ""; loopCard.hidden = false;
  $("#delta").hidden = true; $("#hOut").textContent = ""; $("#copyBtn").hidden = true;

  try {
    const res = await api("/humanize", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, strength: +$("#strength").value, max_rounds: +$("#rounds").value }),
    });
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const parts = buf.split("\n\n"); buf = parts.pop();
      for (const part of parts) {
        const ev = /event: (\w+)/.exec(part)?.[1];
        const data = /data: (.*)/s.exec(part)?.[1];
        if (!data) continue;
        const obj = JSON.parse(data);
        if (ev === "step") addLoopStep(obj);
        else if (ev === "result") showResult(obj);
      }
    }
  } catch (e) { toast("Error: " + e.message); }
  finally { busy($("#humanizeBtn"), false, "✨ Humanize"); }
});

function addLoopStep(step) {
  if (step.error) { toast("Rewriter error: " + step.error); return; }
  const li = document.createElement("li");
  const pct = Math.round((step.score ?? 0) * 100);
  li.innerHTML = `<span class="badge">round ${step.round}</span>
    <span>${step.verdict || ""}</span>
    <span class="bar"><i style="width:${pct}%"></i></span>
    <span>${pct}%</span>`;
  $("#loopList").appendChild(li);
}

function showResult(r) {
  $("#hOut").textContent = r.output;
  $("#copyBtn").hidden = false;
  const d = $("#delta"); d.hidden = false;
  $("#dBefore").textContent = Math.round(r.before.score * 100) + "% AI";
  $("#dAfter").textContent = Math.round(r.after.score * 100) + "% AI";
  $("#dSim").textContent = `meaning kept: ${Math.round(r.meaning_similarity * 100)}%`;
  const cav = $("#hCaveats");
  if (r.caveats?.length) { cav.innerHTML = "<ul>" + r.caveats.map(c => `<li>${c}</li>`).join("") + "</ul>"; cav.hidden = false; }
}

$("#copyBtn").addEventListener("click", async () => {
  await navigator.clipboard.writeText($("#hOut").textContent);
  toast("Copied to clipboard");
});

/* ---------- engine badge ---------- */
function updateEnginePill(kind, backendName) {
  const pill = $("#enginePill");
  if (kind === "backend") {
    pill.textContent = "engine: server";
    pill.title = "Detection runs on the Darren Ai backend.";
  } else {
    pill.textContent = `engine: on-device${backendName ? " (" + backendName + ")" : ""}`;
    pill.title = "Detection runs entirely in your browser — no server, your text never leaves this device.";
  }
}

(async () => {
  try {
    const d = await (await api("/health")).json();
    backendAvailable = true;
    $("#enginePill").textContent = `engine: ${d.engine}`;
    if (d.engine === "stub") $("#enginePill").title = "Stub engine active — deploy a real backend or use the on-device model.";
  } catch {
    backendAvailable = false;
    updateEnginePill("browser");
    $("#enginePill").title = "No backend — the on-device model runs in your browser.";
    // No server available: start warming the in-browser model immediately so
    // the first Detect click is instant. Files are cached after the first load.
    try {
      const be = await getBrowserEngine();
      be.warmup((msg, pct) => {
        const pill = $("#enginePill");
        if (pct != null && pct < 100) {
          pill.textContent = `engine: on-device — loading ${pct}%`;
        } else if (pct === 100 || msg === "Model ready") {
          updateEnginePill("browser", be.backend());
          pill.textContent = `engine: on-device (ready)`;
        }
      });
    } catch { /* ignore — model loads on first click instead */ }
  }
})();
