const $ = (id) => document.getElementById(id);

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
  return {
    text: $("inputText").value,
    from_lang: $("from").value,
    to: $("to").value,
    method: $("method").value,
    beam_width: Number($("beamWidth").value || 5),
    max_length: Number($("maxLength").value || 50),
    device: $("device").value,
    checkpoint_path: $("checkpointPath").value.trim(),
    vocab_dir: $("vocabDir").value.trim(),
  };
}

async function translate() {
  const btn = $("translateBtn");
  btn.disabled = true;
  setStatus("Translating…");

  try {
    const payload = readPayload();
    if (!payload.text || !payload.text.trim()) {
      setStatus("Please enter some text.", "error");
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
  $("inputText").value = "";
  setOutput("");
  setMeta("");
  setStatus("");
  $("inputText").focus();
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
  const method = $("method").value;
  const beam = $("beamWidth");
  const disabled = method !== "beam_search";
  beam.disabled = disabled;
  beam.classList.toggle("opacity-60", disabled);
}

window.addEventListener("DOMContentLoaded", () => {
  healthCheck();
  updateBeamEnabled();

  $("translateBtn").addEventListener("click", translate);
  $("clearBtn").addEventListener("click", clearAll);
  $("copyBtn").addEventListener("click", copyOutput);
  $("method").addEventListener("change", updateBeamEnabled);
  $("from").addEventListener("change", () => setStatus(""));
  $("to").addEventListener("change", () => setStatus(""));

  $("inputText").addEventListener("keydown", (e) => {
    const isMac = navigator.platform.toLowerCase().includes("mac");
    const combo = isMac ? e.metaKey : e.ctrlKey;
    if (combo && e.key === "Enter") {
      e.preventDefault();
      translate();
    }
  });
});

