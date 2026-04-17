const assert = require("node:assert/strict");

const { createArchiveApp } = require("../../app/static/app.js");

const run = (name, fn) => {
  try {
    fn();
    process.stdout.write(`PASS ${name}\n`);
  } catch (error) {
    process.stderr.write(`FAIL ${name}\n${error.stack}\n`);
    process.exitCode = 1;
  }
};

run("mergeSelectedFiles keeps only PDFs and deduplicates by name and size", () => {
  const app = createArchiveApp();
  const merged = app.mergeSelectedFiles(
    [{ name: "alpha.pdf", size: 100, type: "application/pdf" }],
    [
      { name: "alpha.pdf", size: 100, type: "application/pdf" },
      { name: "beta.pdf", size: 250, type: "application/pdf" },
      { name: "notes.txt", size: 25, type: "text/plain" },
    ],
  );

  assert.deepEqual(
    merged.map((file) => [file.name, file.size]),
    [
      ["alpha.pdf", 100],
      ["beta.pdf", 250],
    ],
  );
});

run("removeSelectedFile drops only the matching queued file", () => {
  const app = createArchiveApp();
  const selected = [
    { name: "alpha.pdf", size: 100, type: "application/pdf" },
    { name: "beta.pdf", size: 250, type: "application/pdf" },
  ];

  const remaining = app.removeSelectedFile(selected, "alpha.pdf::100");

  assert.deepEqual(
    remaining.map((file) => file.name),
    ["beta.pdf"],
  );
});

run("buildFileQueueRows exposes an explicit remove action for each queued file", () => {
  const app = createArchiveApp();

  assert.deepEqual(
    app.buildFileQueueRows([
      { name: "alpha.pdf", size: 100, type: "application/pdf" },
    ]),
    [
      {
        key: "alpha.pdf::100",
        name: "alpha.pdf",
        size: "100 B",
        removeLabel: "Remove file",
      },
    ],
  );
});

run("renderAnswerHtml converts markdown output and decorates citation chips", () => {
  const app = createArchiveApp({
    markdownToHtml: () => "<p><strong>Policy</strong> summary [1, 2]</p>",
    sanitizeHtml: (html) => html,
  });

  const html = app.renderAnswerHtml("ignored");

  assert.match(html, /<strong>Policy<\/strong>/);
  assert.match(html, /data-cite="1"/);
  assert.match(html, /href="#cite-2"/);
});

run("buildMetaDisplay returns clear label values", () => {
  const app = createArchiveApp();

  assert.deepEqual(
    app.buildMetaDisplay({ intent: "search", format: "table" }, 842),
    {
      intent: "Search",
      format: "Table",
      latency: "842 ms",
    },
  );
});

run("buildNotices surfaces refusal, policy, and verification details", () => {
  const app = createArchiveApp();

  const notices = app.buildNotices({
    refusal_reason: "insufficient_evidence",
    policy: { pii_masked: true, pii_entities: ["email"], disclaimer: "legal" },
    verification: { all_supported: false, unsupported_sentences: [2, 4] },
  });

  assert.deepEqual(
    notices.map((notice) => notice.variant),
    ["refusal", "policy", "policy", "verification"],
  );
  assert.match(notices[0].text, /evidence/i);
  assert.match(notices[1].text, /Email/);
  assert.equal(notices[2].tag, "Legal Disclaimer");
  assert.match(notices[3].text, /2 sentences/);
});

run("buildNotices returns empty when no flags are set", () => {
  const app = createArchiveApp();

  assert.deepEqual(
    app.buildNotices({ verification: { all_supported: true }, policy: {} }),
    [],
  );
});

run("syncComposerHeight grows the textarea and marks the form as expanded", () => {
  const app = createArchiveApp();
  const classes = new Set();
  const textarea = {
    style: {},
    scrollHeight: 164,
  };
  const form = {
    classList: {
      toggle(name, enabled) {
        if (enabled) {
          classes.add(name);
        } else {
          classes.delete(name);
        }
      },
    },
  };

  app.syncComposerHeight(textarea, form);

  assert.equal(textarea.style.height, "164px");
  assert.equal(textarea.style.overflowY, "hidden");
  assert.ok(classes.has("ask--expanded"));
});
