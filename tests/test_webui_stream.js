"use strict";

const assert = require("node:assert/strict");
const { TextDecoder: NativeTextDecoder, TextEncoder: NativeTextEncoder } = require("node:util");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const APP_PATH = path.join(__dirname, "..", "bayesprobe", "webui_static", "app.js");

class Element {
  constructor(tagName) {
    this.tagName = tagName;
    this.children = [];
    this.className = "";
    this.dataset = {};
    this.disabled = false;
    this.hidden = false;
    this.textContent = "";
    this.value = "";
    this.checked = false;
    this.open = false;
    this._innerHTML = "";
    const classes = new Set();
    this.classList = {
      add(...names) {
        names.forEach((name) => classes.add(name));
      },
      remove(...names) {
        names.forEach((name) => classes.delete(name));
      },
      toggle(name, force) {
        const enabled = force === undefined ? !classes.has(name) : force;
        if (enabled) classes.add(name);
        else classes.delete(name);
        return enabled;
      },
      contains(name) {
        return classes.has(name);
      },
    };
  }

  get innerHTML() {
    return this._innerHTML;
  }

  set innerHTML(value) {
    this._innerHTML = String(value);
    this.children = [];
  }

  append(...children) {
    children.forEach((child) => this.appendChild(child));
  }

  appendChild(child) {
    this.children.push(child);
    return child;
  }

  querySelector(selector) {
    return findDescendant(this, selector);
  }

  addEventListener() {}
}

function findDescendant(root, selector) {
  for (const child of root.children) {
    if (
      selector === '.progress-item[data-state="active"]' &&
      child.className === "progress-item" &&
      child.dataset.state === "active"
    ) {
      return child;
    }
    if (selector === ".progress-status" && child.className === "progress-status") {
      return child;
    }
    const match = findDescendant(child, selector);
    if (match) return match;
  }
  return null;
}

class MockTextEncoder {
  constructor() {
    this.encoder = new NativeTextEncoder();
  }

  encode(value) {
    return this.encoder.encode(value);
  }
}

class MockTextDecoder {
  constructor() {
    this.decoder = new NativeTextDecoder();
    this.calls = [];
  }

  decode(value, options) {
    this.calls.push({ value, options });
    return this.decoder.decode(value, options);
  }
}

function loadApp({ fetch = () => Promise.reject(new Error("unexpected fetch")) } = {}) {
  const elements = new Map();
  const ids = [
    "run-form",
    "provider-kind",
    "provider-auth",
    "provider-note",
    "status-banner",
    "progress-list",
    "progress-state",
    "answer-panel",
    "answer-projection-state",
    "belief-panel",
    "trace-pane",
    "run-id",
    "run-button",
    "api-key",
    "base-url",
    "model-name",
    "timeout-seconds",
    "max-output-tokens",
    "question",
    "task-context",
    "context",
    "max-cycles",
    "max-probes",
    "stop-on-no-probes",
    "confidence-threshold",
    "posterior-delta-threshold",
  ];
  for (const id of ids) {
    elements.set(id, new Element("div"));
  }
  elements.get("provider-kind").value = "deterministic";

  const document = {
    createElement(tagName) {
      return new Element(tagName);
    },
    querySelector(selector) {
      return elements.get(selector.replace(/^#/, "")) || null;
    },
  };
  const context = vm.createContext({
    document,
    Error,
    JSON,
    Set,
    String,
    Number,
    ReadableStream,
    TextDecoder: MockTextDecoder,
    Uint8Array,
    fetch,
  });

  const source = `${fs.readFileSync(APP_PATH, "utf8")}
globalThis.__webuiTestExports = { consumeRunStream, handleProgressEvent, handleSubmit, syncProviderControls };`;
  vm.runInContext(source, context, { filename: APP_PATH });
  return {
    api: context.__webuiTestExports,
    elements,
  };
}

function streamFromText(
  text,
  splitAt,
  { onCancel = () => {}, stayOpen = false } = {}
) {
  const encoder = new MockTextEncoder();
  const chunks = splitAt == null
    ? [encoder.encode(text)]
    : [encoder.encode(text.slice(0, splitAt)), encoder.encode(text.slice(splitAt))];
  let chunkIndex = 0;
  const stream = new ReadableStream({
    pull(controller) {
      if (chunkIndex < chunks.length) {
        controller.enqueue(chunks[chunkIndex++]);
      } else if (stayOpen) {
        return new Promise(() => {});
      } else {
        controller.close();
      }
    },
    cancel(reason) {
      onCancel(reason);
    },
  });
  return stream;
}

function integratedCycleEvent() {
  return {
    event: "cycle_integrated",
    sequence: 1,
    cycle_id: "cycle-1",
    cycle_index: 1,
    data: {
      cycle_id: "cycle-1",
      signal_shape: "active_plus_passive",
      cycle: { boundary_status: "integrated" },
      probes: [],
      signals: [],
      evidence_events: [],
      belief_updates: [],
      hypothesis_evolutions: [],
      belief_state: {
        hypotheses: [
          {
            id: "H1",
            prior: 0.5,
            posterior: 0.75,
            statement: "H1 remains supported.",
          },
        ],
        posterior_summary: {
          total_active_posterior: 1,
          top_hypothesis: "H1",
          posterior_gap: 0.5,
        },
        uncertainty_summary: "More evidence may change the ranking.",
      },
      answer_projection: {
        current_best_hypothesis: "H1",
        answer: "H1 is currently favored.",
        posterior_summary: "H1=0.750",
        main_uncertainty: "More evidence may change the ranking.",
        weakest_assumption: "The evidence remains reliable.",
      },
    },
  };
}

test("consumes split NDJSON chunks and unlocks a completed stream", async () => {
  const { api } = loadApp();
  const events = [
    { event: "run_started", sequence: 1, run_id: "run-1", data: {} },
    { event: "run_completed", sequence: 2, data: {} },
  ];
  const payload = `${events.map((event) => JSON.stringify(event)).join("\n")}\n`;
  const stream = streamFromText(payload, payload.indexOf("run_completed") + 5);
  const received = [];

  await api.consumeRunStream({ body: stream }, (event) => received.push(event.event));

  assert.deepEqual(received, ["run_started", "run_completed"]);
  assert.equal(stream.locked, false);
});

test("accepts a tagged task admission outcome as a successful terminal event", async () => {
  const { api } = loadApp();
  const event = {
    event: "task_admission_completed",
    sequence: 1,
    run_id: "run-needs-reframing",
    cycle_id: null,
    cycle_index: null,
    data: {
      result_type: "needs_reframing",
      admission: {
        status: "needs_reframing",
        clarification_questions: ["What concrete answer should be evaluated?"],
      },
    },
  };
  const stream = streamFromText(`${JSON.stringify(event)}\n`);
  const received = [];

  await api.consumeRunStream({ body: stream }, (progress) => {
    received.push(progress.event);
  });

  assert.deepEqual(received, ["task_admission_completed"]);
  assert.equal(stream.locked, false);
});

test("cancels and unlocks a stream after a malformed progress event", async () => {
  const { api } = loadApp();
  let cancelCalls = 0;
  const stream = streamFromText(
    '{"event":"run_started"}\nnot-json\n',
    undefined,
    { onCancel: () => { cancelCalls += 1; }, stayOpen: true }
  );

  await assert.rejects(
    api.consumeRunStream({ body: stream }, () => {}),
    /invalid progress event/
  );

  assert.equal(cancelCalls, 1);
  assert.equal(stream.locked, false);
});

test("rejects incomplete streams after unlocking at EOF", async () => {
  const { api } = loadApp();
  const stream = streamFromText('{"event":"run_completed"');

  await assert.rejects(
    api.consumeRunStream({ body: stream }, () => {}),
    /incomplete event/
  );

  assert.equal(stream.locked, false);
});

test("preserves an integrated cycle when a sanitized terminal failure arrives", async () => {
  const { api, elements } = loadApp();
  const events = [
    integratedCycleEvent(),
    {
      event: "run_failed",
      sequence: 2,
      data: { error: { message: "provider request failed" } },
    },
  ];
  const stream = streamFromText(
    `${events.map((event) => JSON.stringify(event)).join("\n")}\n`
  );

  await assert.rejects(
    api.consumeRunStream({ body: stream }),
    /provider request failed/
  );

  assert.equal(stream.locked, false);
  assert.equal(elements.get("answer-panel").children.length, 5);
  assert.ok(elements.get("belief-panel").children.length > 0);
  assert.equal(elements.get("trace-pane").children.length, 1);
  assert.equal(elements.get("progress-list").children.at(-1).dataset.state, "failed");
});

test("keeps beliefs pending until task framing completes initialization", () => {
  const { api, elements } = loadApp();
  elements.get("belief-panel").textContent = "Run pending.";

  api.handleProgressEvent({
    event: "run_started",
    sequence: 1,
    run_id: "run-1",
    data: {},
  });
  api.handleProgressEvent({
    event: "task_framing_started",
    sequence: 2,
    run_id: "run-1",
    data: {},
  });
  api.handleProgressEvent({
    event: "task_framing_completed",
    sequence: 3,
    run_id: "run-1",
    data: { task_frame: { task_kind: "claim_verification" } },
  });

  assert.equal(elements.get("belief-panel").textContent, "Run pending.");
  assert.equal(
    elements.get("progress-list").children[1].children[1].textContent,
    "Framing task"
  );
  assert.equal(
    elements.get("progress-list").children[2].children[1].textContent,
    "Task framed"
  );

  api.handleProgressEvent({
    event: "initialization_completed",
    sequence: 4,
    run_id: "run-1",
    data: {
      belief_state: integratedCycleEvent().data.belief_state,
    },
  });

  assert.ok(elements.get("belief-panel").children.length > 0);
});

test("rejects out-of-order initialization without rendering a belief state", () => {
  const { api, elements } = loadApp();
  elements.get("belief-panel").textContent = "Run pending.";

  api.handleProgressEvent({
    event: "run_started",
    sequence: 1,
    run_id: "run-out-of-order",
    data: {},
  });

  assert.throws(
    () => api.handleProgressEvent({
      event: "initialization_completed",
      sequence: 2,
      run_id: "run-out-of-order",
      data: { belief_state: integratedCycleEvent().data.belief_state },
    }),
    /before task framing completed/
  );
  assert.equal(elements.get("belief-panel").textContent, "Run pending.");
  assert.equal(elements.get("belief-panel").children.length, 0);
});

test("resets task framing completion for each run", () => {
  const { api, elements } = loadApp();

  api.handleProgressEvent({
    event: "run_started",
    sequence: 1,
    run_id: "run-one",
    data: {},
  });
  api.handleProgressEvent({
    event: "task_framing_completed",
    sequence: 2,
    run_id: "run-one",
    data: { task_frame: { task_kind: "claim_verification" } },
  });
  api.handleProgressEvent({
    event: "run_started",
    sequence: 1,
    run_id: "run-two",
    data: {},
  });

  assert.throws(
    () => api.handleProgressEvent({
      event: "initialization_completed",
      sequence: 2,
      run_id: "run-two",
      data: { belief_state: integratedCycleEvent().data.belief_state },
    }),
    /before task framing completed/
  );
  assert.equal(elements.get("belief-panel").children.length, 0);
});

test("uses relation-neutral permanent belief terminology", () => {
  const indexPath = path.join(__dirname, "..", "bayesprobe", "webui_static", "index.html");
  const index = fs.readFileSync(indexPath, "utf8");
  const app = fs.readFileSync(APP_PATH, "utf8");

  assert.match(index, /<span class="eyebrow">Belief measure<\/span>/);
  assert.doesNotMatch(index, /<span class="eyebrow">Posterior<\/span>/);
  assert.match(app, /cycle_integrated: "Belief updated"/);
  assert.match(app, /relation === "independent" \? "credence" : "posterior"/);
});

test("renders independent beliefs as non-normalized credence", () => {
  const { api, elements } = loadApp();
  const event = integratedCycleEvent();
  event.data.belief_state = {
    task_frame: {
      hypothesis_frame: { relation: "independent" },
    },
    hypotheses: [
      { id: "H1", prior: 0.5, posterior: 0.8, statement: "First claim." },
      { id: "H2", prior: 0.5, posterior: 0.7, statement: "Second claim." },
    ],
    posterior_summary: {
      total_active_credence: 1.5,
      top_hypothesis: "H1",
      credence_gap: 0.1,
    },
    uncertainty_summary: "Independent hypotheses may coexist.",
  };

  api.handleProgressEvent(event);

  const beliefRows = elements.get("belief-panel").children;
  assert.equal(beliefRows[0].children[0].textContent, "Total credence (not normalized)");
  assert.equal(beliefRows[0].children[1].textContent, "1.500");
  assert.equal(beliefRows[1].children[0].textContent, "Top / credence gap");
  assert.match(beliefRows[3].children[1].children[0].textContent, /credence 0\.800/);
  assert.doesNotMatch(beliefRows[3].children[1].children[0].textContent, /posterior/);
  assert.equal(
    elements.get("answer-panel").children[2].children[0].textContent,
    "Belief summary"
  );
});

test("handleSubmit preserves integrated output after a terminal stream failure", async () => {
  const events = [
    integratedCycleEvent(),
    {
      event: "run_failed",
      sequence: 2,
      data: { error: { message: "provider request failed" } },
    },
  ];
  const stream = streamFromText(
    `${events.map((event) => JSON.stringify(event)).join("\n")}\n`
  );
  const requests = [];
  const { api, elements } = loadApp({
    fetch: async (...request) => {
      requests.push(request);
      return { ok: true, body: stream };
    },
  });
  const apiKeyField = elements.get("api-key");
  apiKeyField.value = "sk-session-only";
  let prevented = false;

  await api.handleSubmit({
    preventDefault() {
      prevented = true;
    },
  });

  assert.equal(prevented, true);
  assert.equal(requests[0][0], "/api/runs/autonomous/stream");
  assert.equal(stream.locked, false);
  assert.equal(elements.get("answer-panel").children.length, 5);
  assert.ok(elements.get("belief-panel").children.length > 0);
  assert.equal(elements.get("trace-pane").children.length, 1);
  assert.equal(elements.get("status-banner").textContent, "provider request failed");
  assert.equal(elements.get("run-button").disabled, false);
  assert.equal(elements.get("run-button").textContent, "Run autonomous loop");
  assert.equal(elements.get("provider-kind").disabled, false);
  assert.equal(elements.get("api-key").disabled, false);
  assert.equal(apiKeyField.value, "sk-session-only");
});

test("handleSubmit marks malformed stream progress as failed", async () => {
  const stream = streamFromText(
    '{"event":"run_started","sequence":1,"run_id":"run-1","data":{}}\nnot-json\n',
    undefined,
    { stayOpen: true }
  );
  const { api, elements } = loadApp({
    fetch: async () => ({ ok: true, body: stream }),
  });
  const apiKeyField = elements.get("api-key");
  apiKeyField.value = "sk-session-only";

  await api.handleSubmit({ preventDefault() {} });

  assert.equal(elements.get("progress-state").textContent, "Failed");
  assert.equal(elements.get("progress-list").children.at(-1).dataset.state, "failed");
  assert.equal(
    elements.get("status-banner").textContent,
    "Server returned an invalid progress event"
  );
  assert.equal(elements.get("status-banner").classList.contains("error"), true);
  assert.equal(elements.get("run-button").disabled, false);
  assert.equal(elements.get("provider-kind").disabled, false);
  assert.equal(elements.get("api-key").disabled, false);
  assert.equal(apiKeyField.value, "sk-session-only");
});

test("pending submit blocks re-entry and provider controls stay disabled", async () => {
  const pendingFetches = [];
  const { api, elements } = loadApp({
    fetch: (...request) => new Promise((resolve) => {
      pendingFetches.push({ request, resolve });
    }),
  });

  const firstSubmit = api.handleSubmit({ preventDefault() {} });
  await Promise.resolve();
  const secondSubmit = api.handleSubmit({ preventDefault() {} });
  elements.get("provider-kind").value = "openai_chat_completions";
  api.syncProviderControls();

  const activeState = {
    fetchCount: pendingFetches.length,
    runDisabled: elements.get("run-button").disabled,
    providerDisabled: elements.get("provider-kind").disabled,
    apiKeyDisabled: elements.get("api-key").disabled,
  };
  for (const pending of pendingFetches) {
    pending.resolve({
      ok: true,
      body: streamFromText('{"event":"run_completed","sequence":1,"data":{}}\n'),
    });
  }
  await Promise.all([firstSubmit, secondSubmit]);

  assert.equal(activeState.fetchCount, 1);
  assert.equal(activeState.runDisabled, true);
  assert.equal(activeState.providerDisabled, true);
  assert.equal(activeState.apiKeyDisabled, true);
  assert.equal(elements.get("run-button").disabled, false);
  assert.equal(elements.get("provider-kind").disabled, false);
  assert.equal(elements.get("api-key").disabled, false);
});

test("answer projection is current until run completion marks it final", () => {
  const { api, elements } = loadApp();

  api.handleProgressEvent({
    event: "run_started",
    sequence: 1,
    run_id: "run-1",
    data: {},
  });
  assert.equal(elements.get("answer-projection-state").textContent, "Current");

  api.handleProgressEvent(integratedCycleEvent());
  assert.equal(elements.get("answer-projection-state").textContent, "Current");

  api.handleProgressEvent({
    event: "run_completed",
    sequence: 3,
    data: {},
  });
  assert.equal(elements.get("answer-projection-state").textContent, "Final");
});

test("preflight HTTP failure restores controls and retains the API key", async () => {
  const { api, elements } = loadApp({
    fetch: async () => ({
      ok: false,
      async text() {
        return JSON.stringify({ error: { message: "question must not be empty" } });
      },
    }),
  });
  elements.get("api-key").value = "sk-session-only";

  await api.handleSubmit({ preventDefault() {} });

  assert.equal(elements.get("status-banner").textContent, "question must not be empty");
  assert.equal(elements.get("run-button").disabled, false);
  assert.equal(elements.get("provider-kind").disabled, false);
  assert.equal(elements.get("api-key").disabled, false);
  assert.equal(elements.get("api-key").value, "sk-session-only");
});
