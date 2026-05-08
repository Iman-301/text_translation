const $ = (id) => document.getElementById(id);

function val(id, fallback = "") {
  const el = $(id);
  if (!el) return fallback;
  // Support inputs/selects/textareas and plain divs.
  return typeof el.value === "string" ? el.value : el.textContent || fallback;
}

function setStatus(msg, tone = "neutral") {
  const el = $("status");
  if (!el) return;
  el.textContent = msg;
  el.className =
    "text-xs " +
    (tone === "error"
      ? "text-rose-300"
      : tone === "ok"
      ? "text-emerald-300"
      : "text-slate-300");
}

function setOutput(text) {
  $("outputText").textContent = text || "";
}

function setMeta(text) {
  $("meta").textContent = text || "";
}

async function healthCheck() {
  const dot = $("healthDot");
  const text = $("healthText");
  try {
    const res = await fetch("/api/health", { cache: "no-store" });
    if (!res.ok) throw new Error("not ok");
    dot.className = "h-2 w-2 rounded-full bg-emerald-400";
    text.textContent = "Server online";
  } catch {
    dot.className = "h-2 w-2 rounded-full bg-rose-400";
    text.textContent = "Server offline";
  }
}

function readPayload() {
  const checkpointFallback = "seq2seq/models/checkpoints/best_model.pt";
  const vocabFallback = "seq2seq/data/vocab";
  return {
    text: val("inputText", ""),
    to: val("to", "fr"),
    // Simple UI defaults
    method: val("method", "greedy"),
    beam_width: Number(val("beamWidth", "5") || 5),
    max_length: Number(val("maxLength", "50") || 50),
    device: val("device", "cpu"),
    checkpoint_path: (val("checkpointPath", checkpointFallback) || checkpointFallback).trim(),
    vocab_dir: (val("vocabDir", vocabFallback) || vocabFallback).trim(),
  };
}

async function translate() {
  const btn = $("translateBtn");
  btn.disabled = true;
  setStatus("Translating…");

  try {
    const payload = readPayload();
    if (!payload.text || !payload.text.trim()) {
      setStatus("Please enter some English text.", "error");
      btn.disabled = false;
      return;
    }

    const res = await fetch("/api/translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json().catch(() => ({}));

    if (!res.ok || !data.ok) {
      setOutput("");
      setMeta("");
      setStatus(data.error || "Translation failed.", "error");
      btn.disabled = false;
      return;
    }

    setOutput(data.translation);
    setMeta(data.meta);
    setStatus("Done.", "ok");
  } catch (err) {
    setStatus(String(err || "Unexpected error"), "error");
  } finally {
    btn.disabled = false;
  }
}

function clearAll() {
  const input = $("inputText");
  if (input) input.value = "";
  setOutput("");
  setMeta("");
  setStatus("");
  if (input) input.focus();
}

async function copyOutput() {
  const text = $("outputText").textContent || "";
  if (!text.trim()) {
    setStatus("Nothing to copy yet.", "neutral");
    return;
  }
  try {
    await navigator.clipboard.writeText(text);
    setStatus("Copied.", "ok");
  } catch {
    setStatus("Copy failed (browser permissions).", "error");
  }
}

function updateBeamEnabled() {
  const methodEl = $("method");
  const beam = $("beamWidth");
  if (!methodEl || !beam) return;
  const method = methodEl.value;
  const disabled = method !== "beam_search";
  beam.disabled = disabled;
  beam.classList.toggle("opacity-60", disabled);
}

window.addEventListener("DOMContentLoaded", () => {
  healthCheck();
  updateBeamEnabled();

  const translateBtn = $("translateBtn");
  const clearBtn = $("clearBtn");
  const copyBtn = $("copyBtn");
  const method = $("method");
  const inputText = $("inputText");

  if (translateBtn) translateBtn.addEventListener("click", translate);
  if (clearBtn) clearBtn.addEventListener("click", clearAll);
  if (copyBtn) copyBtn.addEventListener("click", copyOutput);
  if (method) method.addEventListener("change", updateBeamEnabled);

  if (inputText)
    inputText.addEventListener("keydown", (e) => {
    const isMac = navigator.platform.toLowerCase().includes("mac");
    const combo = isMac ? e.metaKey : e.ctrlKey;
    if (combo && e.key === "Enter") {
      e.preventDefault();
      translate();
    }
  });
});

