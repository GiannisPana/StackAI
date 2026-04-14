// The Archive — front-end for StackAI RAG
// Wires the upload + query UI to /ingest and /query.

(() => {
  const $ = (id) => document.getElementById(id);

  // Date stamp in the masthead
  const d = new Date();
  const months = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"];
  $("dateStamp").textContent = `${String(d.getDate()).padStart(2,"0")} ${months[d.getMonth()]} ${d.getFullYear()}`;

  // ---------- Ingest: file picker + drag/drop ----------
  const filesInput = $("files");
  const dropzone   = $("dropzone");
  const filelist   = $("filelist");
  const ingestBtn  = $("ingestBtn");
  let selected = [];

  const fmtSize = (n) => {
    if (n < 1024) return `${n} B`;
    if (n < 1024*1024) return `${(n/1024).toFixed(1)} KB`;
    return `${(n/1048576).toFixed(2)} MB`;
  };

  const renderFileList = () => {
    filelist.innerHTML = "";
    for (const f of selected) {
      const li = document.createElement("li");
      const name = document.createElement("span");
      name.className = "fname";
      name.textContent = f.name;
      const size = document.createElement("span");
      size.className = "fsize";
      size.textContent = fmtSize(f.size);
      li.append(name, size);
      filelist.appendChild(li);
    }
    ingestBtn.disabled = selected.length === 0;
  };

  const addFiles = (fileList) => {
    const incoming = Array.from(fileList).filter(f => f.type === "application/pdf" || f.name.toLowerCase().endsWith(".pdf"));
    // Dedup by name+size
    const key = (f) => `${f.name}::${f.size}`;
    const seen = new Set(selected.map(key));
    for (const f of incoming) {
      if (!seen.has(key(f))) { selected.push(f); seen.add(key(f)); }
    }
    renderFileList();
  };

  filesInput.addEventListener("change", (e) => addFiles(e.target.files));

  ["dragenter","dragover"].forEach(ev => dropzone.addEventListener(ev, (e) => {
    e.preventDefault(); dropzone.classList.add("is-hot");
  }));
  ["dragleave","drop"].forEach(ev => dropzone.addEventListener(ev, (e) => {
    e.preventDefault(); dropzone.classList.remove("is-hot");
  }));
  dropzone.addEventListener("drop", (e) => {
    if (e.dataTransfer?.files) addFiles(e.dataTransfer.files);
  });

  // ---------- Ingest: submit ----------
  ingestBtn.addEventListener("click", async () => {
    if (!selected.length) return;
    ingestBtn.disabled = true;
    const original = ingestBtn.querySelector(".btn__label").textContent;
    ingestBtn.querySelector(".btn__label").textContent = "Committing…";

    const fd = new FormData();
    for (const f of selected) fd.append("files", f);

    try {
      const r = await fetch("/ingest", { method: "POST", body: fd });
      const body = await r.json();
      renderIngestLog(body);
      selected = [];
      filesInput.value = "";
      renderFileList();
    } catch (err) {
      renderIngestLog({ failed: [{ filename: "(network)", reason: String(err) }] });
    } finally {
      ingestBtn.querySelector(".btn__label").textContent = original;
      ingestBtn.disabled = selected.length === 0;
    }
  });

  const renderIngestLog = (body) => {
    const log = $("ingestLog");
    const out = $("ingestOut");
    log.hidden = false;
    out.innerHTML = "";

    const push = (cls, tag, text) => {
      const row = document.createElement("div");
      row.className = `log-entry ${cls}`;
      const t = document.createElement("span");
      t.className = "tag";
      t.textContent = tag;
      const msg = document.createElement("span");
      msg.textContent = text;
      row.append(t, msg);
      out.appendChild(row);
    };

    for (const it of body.ingested || []) {
      push("ok", "OK", `${it.filename} — ${it.num_chunks ?? "?"} chunks, ${it.num_pages ?? "?"} pages`);
    }
    for (const it of body.skipped || []) {
      push("skip", "SKIP", `${it.filename} — ${it.reason || "duplicate"}`);
    }
    for (const it of body.failed || []) {
      push("err", "FAIL", `${it.filename} — ${it.reason || "unknown error"}`);
    }
    if (!out.children.length) push("skip", "—", "No changes reported.");
  };

  // ---------- Inquiry ----------
  const askForm    = $("askForm");
  const qInput     = $("q");
  const answerPane = $("answerPane");
  const placeholder = $("placeholder");
  const status     = $("answerStatus");
  const answerEl   = $("answer");
  const citationsEl = $("citations");
  const warningsEl = $("warnings");
  const intentTag  = $("folioIntent");
  const formatTag  = $("folioFormat");
  const latencyTag = $("folioLatency");

  const escapeHtml = (s) => s.replace(/[&<>"']/g, (c) => ({
    "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"
  }[c]));

  // Replace [1], [2,3] with styled superscript markers
  const renderAnswer = (text) => {
    const escaped = escapeHtml(text || "");
    return escaped.replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, (_, nums) => {
      const parts = nums.split(/\s*,\s*/).map(n => `<a class="cite" href="#cite-${n}">${n}</a>`);
      return parts.join("");
    });
  };

  askForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = qInput.value.trim();
    if (!query) return;

    placeholder.hidden = true;
    answerPane.hidden = false;
    status.textContent = "Consulting the holdings";
    status.classList.add("is-loading");
    answerEl.innerHTML = "";
    citationsEl.innerHTML = "";
    warningsEl.innerHTML = "";
    intentTag.textContent = "—";
    formatTag.textContent = "—";
    latencyTag.textContent = "—";

    const t0 = performance.now();
    try {
      const r = await fetch("/query", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ query }),
      });
      const body = await r.json();
      const dt = Math.round(performance.now() - t0);

      status.classList.remove("is-loading");
      status.textContent = body.refusal_reason ? "Refused" : "Response";

      intentTag.textContent  = `INTENT · ${(body.intent || "—").toString().toUpperCase()}`;
      formatTag.textContent  = `FORMAT · ${(body.format || "—").toString().toUpperCase()}`;
      latencyTag.textContent = `${dt} MS`;

      answerEl.innerHTML = renderAnswer(body.answer || body.refusal_reason || "(no answer)");

      for (const c of body.citations || []) {
        const li = document.createElement("li");
        li.id = `cite-${c.index}`;
        const head = document.createElement("div");
        head.className = "cite-head";
        const f = document.createElement("span");
        f.className = "cite-file";
        f.textContent = c.filename ?? "Unknown source";
        const m = document.createElement("span");
        m.className = "cite-meta";
        const score = (typeof c.score === "number") ? ` · score ${c.score.toFixed(3)}` : "";
        m.textContent = `p. ${c.page ?? "?"}${score}`;
        head.append(f, m);
        const txt = document.createElement("div");
        txt.className = "cite-text";
        txt.textContent = c.text ?? "";
        li.append(head, txt);
        citationsEl.appendChild(li);
      }

      for (const w of body.warnings || []) {
        const p = document.createElement("p");
        p.className = "warn";
        p.textContent = w;
        warningsEl.appendChild(p);
      }
    } catch (err) {
      status.classList.remove("is-loading");
      status.textContent = "Error";
      answerEl.textContent = `Request failed: ${err}`;
    }
  });
})();
