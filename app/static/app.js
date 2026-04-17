// The Archive - front-end for StackAI RAG.
// Wires the upload + query UI to /ingest and /query.

const createArchiveApp = (options = {}) => {
  const escapeHtml = (value) =>
    String(value ?? "").replace(/[&<>"']/g, (char) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    })[char]);

  const titleCase = (value) => {
    const text = String(value ?? "").trim();
    if (!text) {
      return "-";
    }
    return text
      .replace(/[_-]+/g, " ")
      .split(/\s+/)
      .map((part) => part.charAt(0).toUpperCase() + part.slice(1).toLowerCase())
      .join(" ");
  };

  const fileKey = (file) => `${file.name}::${file.size}`;

  const fmtSize = (size) => {
    if (size < 1024) return `${size} B`;
    if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
    return `${(size / 1048576).toFixed(2)} MB`;
  };

  const mergeSelectedFiles = (selected, fileList) => {
    const next = selected.slice();
    const seen = new Set(next.map(fileKey));
    for (const file of Array.from(fileList || [])) {
      const isPdf = file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf");
      if (!isPdf) {
        continue;
      }
      const key = fileKey(file);
      if (!seen.has(key)) {
        next.push(file);
        seen.add(key);
      }
    }
    return next;
  };

  const removeSelectedFile = (selected, keyToRemove) =>
    selected.filter((file) => fileKey(file) !== keyToRemove);

  const buildFileQueueRows = (selected) =>
    selected.map((file) => ({
      key: fileKey(file),
      name: file.name,
      size: fmtSize(file.size),
      removeLabel: "Remove file",
    }));

  const decorateCitationHtml = (html) =>
    html.replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, (_, numbers) => {
      const chips = numbers.split(/\s*,\s*/).map((number) =>
        `<a class="cite" href="#cite-${number}" data-cite="${number}">${number}</a>`,
      );
      return chips.join("");
    });

  const renderMarkdown = options.markdownToHtml || ((text) => {
    const input = String(text ?? "");
    if (typeof globalThis.marked?.setOptions === "function" && !globalThis.marked.__archiveConfigured) {
      globalThis.marked.setOptions({ breaks: true, gfm: true });
      globalThis.marked.__archiveConfigured = true;
    }
    if (typeof globalThis.marked?.parse === "function") {
      return globalThis.marked.parse(input);
    }
    return `<p>${escapeHtml(input)}</p>`;
  });

  const sanitizeHtml = options.sanitizeHtml || ((html) => {
    if (typeof globalThis.DOMPurify?.sanitize === "function") {
      return globalThis.DOMPurify.sanitize(html);
    }
    return html;
  });

  const renderAnswerHtml = (text) => {
    const input = String(text ?? "");
    try {
      const rendered = renderMarkdown(input);
      if (!options.sanitizeHtml && typeof globalThis.DOMPurify?.sanitize !== "function") {
        return decorateCitationHtml(`<p>${escapeHtml(input)}</p>`);
      }
      const safe = sanitizeHtml(rendered);
      return decorateCitationHtml(safe);
    } catch (error) {
      return decorateCitationHtml(`<p>${escapeHtml(input)}</p>`);
    }
  };

  const buildMetaDisplay = (body, durationMs) => ({
    intent: titleCase(body?.intent),
    format: titleCase(body?.format),
    latency: Number.isFinite(durationMs) ? `${durationMs} ms` : "-",
  });

  const REFUSAL_MESSAGES = {
    insufficient_evidence:
      "The retrieved passages do not contain enough evidence to answer this question.",
    personalized_advice:
      "This question asks for personalized advice that the archive will not generate.",
    pii: "The question appears to contain personal information.",
    pii_risk: "Answering would reveal personally identifiable information.",
    legal: "The question requests legal advice the archive will not provide.",
    medical: "The question requests medical advice the archive will not provide.",
  };

  const buildNotices = (body) => {
    const notices = [];

    const reason = body?.refusal_reason;
    if (reason) {
      const text = REFUSAL_MESSAGES[reason] || String(reason).replace(/_/g, " ");
      notices.push({ variant: "refusal", tag: "Refused", text });
    }

    const policy = body?.policy || {};
    if (policy.pii_masked) {
      const kinds = (policy.pii_entities || []).map(titleCase).filter(Boolean).join(", ");
      const text = kinds
        ? `Personal identifiers masked in the answer (${kinds}).`
        : "Personal identifiers were masked in the answer.";
      notices.push({ variant: "policy", tag: "PII Masked", text });
    }
    if (policy.disclaimer === "legal") {
      notices.push({
        variant: "policy",
        tag: "Legal Disclaimer",
        text: "This response is informational only and is not legal advice.",
      });
    } else if (policy.disclaimer === "medical") {
      notices.push({
        variant: "policy",
        tag: "Medical Disclaimer",
        text: "This response is informational only and is not medical advice.",
      });
    }

    const verification = body?.verification;
    if (verification && verification.all_supported === false) {
      const count = (verification.unsupported_sentences || []).length;
      const text = count
        ? `${count} sentence${count === 1 ? "" : "s"} may lack direct support from the cited passages.`
        : "Some sentences may lack direct support from the cited passages.";
      notices.push({ variant: "verification", tag: "Evidence Check", text });
    }

    return notices;
  };

  const buildCitationMap = (citations) => {
    const lookup = new Map();
    for (const citation of citations || []) {
      lookup.set(String(citation.index), citation);
    }
    return lookup;
  };

  const clamp = (value, min, max) => Math.min(Math.max(value, min), max);

  const syncComposerHeight = (textarea, form) => {
    if (!textarea) {
      return;
    }

    const dataset = textarea.dataset || (textarea.dataset = {});
    textarea.style.height = "auto";
    textarea.style.overflowY = "hidden";
    const nextHeight = textarea.scrollHeight || 0;
    textarea.style.height = `${nextHeight}px`;

    if (!dataset.baseHeight) {
      dataset.baseHeight = String(Math.min(nextHeight || 40, 40));
    }

    if (form?.classList?.toggle) {
      const baseHeight = Number(dataset.baseHeight || 0);
      form.classList.toggle("ask--expanded", nextHeight > baseHeight + 8);
    }
  };

  const init = (doc = document) => {
    if (!doc) {
      return;
    }

    const byId = (id) => doc.getElementById(id);

    const dateStamp = byId("dateStamp");
    if (dateStamp) {
      const now = new Date();
      const months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"];
      dateStamp.textContent = `${String(now.getDate()).padStart(2, "0")} ${months[now.getMonth()]} ${now.getFullYear()}`;
    }

    const filesInput = byId("files");
    const dropzone = byId("dropzone");
    const filelist = byId("filelist");
    const ingestBtn = byId("ingestBtn");
    const ingestLog = byId("ingestLog");
    const ingestOut = byId("ingestOut");
    const askForm = byId("askForm");
    const qInput = byId("q");
    const answerPane = byId("answerPane");
    const placeholder = byId("placeholder");
    const status = byId("answerStatus");
    const answerEl = byId("answer");
    const citationsEl = byId("citations");
    const warningsEl = byId("warnings");
    const noticesEl = byId("folioNotices");
    const intentTag = byId("folioIntent");
    const formatTag = byId("folioFormat");
    const latencyTag = byId("folioLatency");
    const tooltip = byId("citeTooltip");
    const tooltipFile = byId("citeTooltipFile");
    const tooltipMeta = byId("citeTooltipMeta");
    const tooltipText = byId("citeTooltipText");

    if (!filesInput || !dropzone || !filelist || !ingestBtn || !askForm || !qInput || !answerEl || !citationsEl) {
      return;
    }

    let selected = [];
    let citationMap = new Map();
    let highlightTimeout = null;

    const renderFileList = () => {
      filelist.innerHTML = "";
      for (const row of buildFileQueueRows(selected)) {
        const item = doc.createElement("li");

        const info = doc.createElement("div");
        info.className = "filelist__info";

        const name = doc.createElement("span");
        name.className = "fname";
        name.textContent = row.name;

        const size = doc.createElement("span");
        size.className = "fsize";
        size.textContent = row.size;

        const remove = doc.createElement("button");
        remove.type = "button";
        remove.className = "filelist__remove";
        remove.dataset.key = row.key;
        remove.textContent = row.removeLabel;
        remove.setAttribute("aria-label", `Remove ${row.name} from upload queue`);
        remove.addEventListener("click", () => {
          selected = removeSelectedFile(selected, remove.dataset.key);
          renderFileList();
        });

        info.append(name, size);
        item.append(info, remove);
        filelist.appendChild(item);
      }
      ingestBtn.disabled = selected.length === 0;
    };

    const addFiles = (fileList) => {
      selected = mergeSelectedFiles(selected, fileList);
      filesInput.value = "";
      renderFileList();
    };

    filesInput.addEventListener("change", (event) => addFiles(event.target.files));

    ["dragenter", "dragover"].forEach((eventName) => {
      dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropzone.classList.add("is-hot");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropzone.classList.remove("is-hot");
      });
    });

    dropzone.addEventListener("drop", (event) => {
      if (event.dataTransfer?.files) {
        addFiles(event.dataTransfer.files);
      }
    });

    const renderIngestLog = (body) => {
      ingestLog.hidden = false;
      ingestOut.innerHTML = "";

      const push = (variant, label, text) => {
        const row = doc.createElement("div");
        row.className = `log-entry ${variant}`;

        const tag = doc.createElement("span");
        tag.className = "tag";
        tag.textContent = label;

        const message = doc.createElement("span");
        message.textContent = text;

        row.append(tag, message);
        ingestOut.appendChild(row);
      };

      for (const item of body.ingested || []) {
        push("ok", "OK", `${item.filename} - ${item.num_chunks ?? "?"} chunks, ${item.num_pages ?? "?"} pages`);
      }
      for (const item of body.skipped || []) {
        push("skip", "SKIP", `${item.filename} - ${item.reason || "duplicate"}`);
      }
      for (const item of body.failed || []) {
        push("err", "FAIL", `${item.filename} - ${item.reason || "unknown error"}`);
      }
      if (!ingestOut.children.length) {
        push("skip", "-", "No changes reported.");
      }
    };

    ingestBtn.addEventListener("click", async () => {
      if (!selected.length) {
        return;
      }

      ingestBtn.disabled = true;
      const label = ingestBtn.querySelector(".btn__label");
      const originalLabel = label?.textContent || "Commit to Archive";
      if (label) {
        label.textContent = "Committing...";
      }

      const formData = new FormData();
      for (const file of selected) {
        formData.append("files", file);
      }

      try {
        const response = await fetch("/ingest", { method: "POST", body: formData });
        let body;
        try {
          body = await response.json();
        } catch (error) {
          body = { failed: [{ filename: "(response)", reason: "invalid response body" }] };
        }
        renderIngestLog(body);
        if (response.ok) {
          selected = [];
          filesInput.value = "";
          renderFileList();
        }
      } catch (error) {
        renderIngestLog({ failed: [{ filename: "(network)", reason: String(error) }] });
      } finally {
        if (label) {
          label.textContent = originalLabel;
        }
        ingestBtn.disabled = selected.length === 0;
      }
    });

    const setMetaDisplay = (body, durationMs) => {
      const meta = buildMetaDisplay(body, durationMs);
      intentTag.textContent = meta.intent;
      formatTag.textContent = meta.format;
      latencyTag.textContent = meta.latency;
    };

    const renderNotices = (body) => {
      if (!noticesEl) {
        return;
      }
      noticesEl.innerHTML = "";
      const notices = buildNotices(body);
      if (!notices.length) {
        noticesEl.hidden = true;
        return;
      }
      for (const notice of notices) {
        const row = doc.createElement("div");
        row.className = `notice notice--${notice.variant}`;

        const tag = doc.createElement("span");
        tag.className = "notice__tag";
        tag.textContent = notice.tag;

        const text = doc.createElement("div");
        text.className = "notice__body";
        text.textContent = notice.text;

        row.append(tag, text);
        noticesEl.appendChild(row);
      }
      noticesEl.hidden = false;
    };

    const hideTooltip = () => {
      if (!tooltip) {
        return;
      }
      tooltip.hidden = true;
      tooltip.classList.remove("is-visible");
    };

    const positionTooltip = (targetRect) => {
      if (!tooltip) {
        return;
      }

      const tooltipRect = tooltip.getBoundingClientRect();
      const viewportWidth = globalThis.innerWidth || doc.documentElement.clientWidth || 0;
      const preferredLeft = targetRect.left + (targetRect.width / 2) - (tooltipRect.width / 2);
      const left = clamp(preferredLeft, 12, Math.max(12, viewportWidth - tooltipRect.width - 12));

      let top = targetRect.top - tooltipRect.height - 12;
      if (top < 12) {
        top = targetRect.bottom + 12;
      }

      tooltip.style.left = `${left}px`;
      tooltip.style.top = `${top}px`;
    };

    const showTooltip = (chip) => {
      const citation = citationMap.get(chip.dataset.cite || "");
      if (!tooltip || !citation) {
        hideTooltip();
        return;
      }

      tooltipFile.textContent = citation.filename || "Unknown source";
      tooltipMeta.textContent = `Page ${citation.page ?? "?"}`;
      tooltipText.textContent = citation.text || "";
      tooltip.hidden = false;
      tooltip.classList.add("is-visible");
      positionTooltip(chip.getBoundingClientRect());
    };

    const flashCitation = (target) => {
      target.classList.remove("is-highlighted");
      // Force the removal to apply so repeated clicks still flash.
      void target.offsetWidth;
      target.classList.add("is-highlighted");
      if (highlightTimeout) {
        globalThis.clearTimeout(highlightTimeout);
      }
      highlightTimeout = globalThis.setTimeout(() => {
        target.classList.remove("is-highlighted");
      }, 700);
    };

    const jumpToCitation = (index) => {
      const target = byId(`cite-${index}`);
      if (!target) {
        return;
      }
      target.scrollIntoView({ behavior: "smooth", block: "nearest" });
      flashCitation(target);
    };

    const bindCitationInteractions = () => {
      hideTooltip();
      for (const chip of answerEl.querySelectorAll(".cite")) {
        chip.addEventListener("mouseenter", () => showTooltip(chip));
        chip.addEventListener("mouseleave", hideTooltip);
        chip.addEventListener("focus", () => showTooltip(chip));
        chip.addEventListener("blur", hideTooltip);
        chip.addEventListener("click", (event) => {
          event.preventDefault();
          jumpToCitation(chip.dataset.cite);
        });
      }
    };

    const renderCitations = (citations) => {
      citationsEl.innerHTML = "";
      citationMap = buildCitationMap(citations);

      for (const citation of citations || []) {
        const item = doc.createElement("li");
        item.id = `cite-${citation.index}`;

        const head = doc.createElement("div");
        head.className = "cite-head";

        const file = doc.createElement("span");
        file.className = "cite-file";
        file.textContent = citation.filename || "Unknown source";

        const meta = doc.createElement("span");
        meta.className = "cite-meta";
        meta.textContent = `Page ${citation.page ?? "?"}${typeof citation.score === "number" ? ` | score ${citation.score.toFixed(3)}` : ""}`;

        const text = doc.createElement("div");
        text.className = "cite-text";
        text.textContent = citation.text || "";

        head.append(file, meta);
        item.append(head, text);
        citationsEl.appendChild(item);
      }
    };

    const clearAnswerState = () => {
      answerEl.innerHTML = "";
      citationsEl.innerHTML = "";
      warningsEl.innerHTML = "";
      citationMap = new Map();
      setMetaDisplay({}, Number.NaN);
      renderNotices({});
      hideTooltip();
    };

    qInput.addEventListener("input", () => syncComposerHeight(qInput, askForm));
    qInput.addEventListener("paste", () => {
      globalThis.setTimeout(() => syncComposerHeight(qInput, askForm), 0);
    });
    qInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        askForm.requestSubmit();
      }
    });
    syncComposerHeight(qInput, askForm);

    askForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      const query = qInput.value.trim();
      if (!query) {
        return;
      }

      placeholder.hidden = true;
      answerPane.hidden = false;
      status.textContent = "Consulting the holdings";
      status.classList.add("is-loading");
      clearAnswerState();

      const startedAt = performance.now();
      try {
        const response = await fetch("/query", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ query }),
        });
        const body = await response.json();
        const durationMs = Math.round(performance.now() - startedAt);

        status.classList.remove("is-loading");
        status.textContent = body.refusal_reason ? "Refused" : "Response";
        setMetaDisplay(body, durationMs);
        renderNotices(body);
        answerEl.innerHTML = renderAnswerHtml(body.answer || body.refusal_reason || "(no answer)");

        renderCitations(body.citations || []);
        bindCitationInteractions();

        for (const warning of body.warnings || []) {
          const row = doc.createElement("p");
          row.className = "warn";
          row.textContent = warning;
          warningsEl.appendChild(row);
        }
      } catch (error) {
        status.classList.remove("is-loading");
        status.textContent = "Error";
        answerEl.textContent = `Request failed: ${error}`;
        citationMap = new Map();
        hideTooltip();
      }
    });

    if (typeof globalThis.addEventListener === "function") {
      globalThis.addEventListener("resize", hideTooltip);
      globalThis.addEventListener("scroll", hideTooltip, true);
    }

    renderFileList();
  };

  return {
    buildMetaDisplay,
    buildNotices,
    createArchiveApp,
    decorateCitationHtml,
    fileKey,
    fmtSize,
    init,
    buildFileQueueRows,
    mergeSelectedFiles,
    removeSelectedFile,
    renderAnswerHtml,
    syncComposerHeight,
  };
};

if (typeof module !== "undefined" && module.exports) {
  module.exports = { createArchiveApp };
}

if (typeof document !== "undefined") {
  const archiveApp = createArchiveApp();
  archiveApp.init(document);
}
