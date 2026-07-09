const form = document.querySelector("#run-form");
const providerKind = document.querySelector("#provider-kind");
const providerAuth = document.querySelector("#provider-auth");
const providerNote = document.querySelector("#provider-note");
const apiKeyField = document.querySelector("#api-key");
const statusBanner = document.querySelector("#status-banner");
const answerPanel = document.querySelector("#answer-panel");
const beliefPanel = document.querySelector("#belief-panel");
const tracePane = document.querySelector("#trace-pane");
const runId = document.querySelector("#run-id");
const runButton = document.querySelector("#run-button");

const PROVIDER_NOTE_BY_KIND = {
  deterministic: "Deterministic mode runs locally without a key.",
  openai_responses: "OpenAI settings are used only for this request.",
  openai_chat_completions: "Chat Completions stays visible in v0.1 but is not supported.",
};

providerKind.addEventListener("change", syncProviderControls);
form.addEventListener("submit", handleSubmit);

syncProviderControls();

async function handleSubmit(event) {
  event.preventDefault();
  if (providerKind.value === "openai_chat_completions") {
    setStatus(PROVIDER_NOTE_BY_KIND.openai_chat_completions, "error");
    return;
  }

  setStatus("Running autonomous loop...", "busy");
  setBusy(true);

  try {
    const requestPayload = buildPayload();
    clearApiKey();
    const response = await fetch('/api/runs/autonomous', {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestPayload),
    });
    const payload = await parseJsonResponse(response);

    if (!response.ok) {
      throw new Error(payload.error?.message || "Run failed");
    }

    renderRun(payload);
    setStatus(`Stopped: ${payload.stop_reason}`, "ok");
  } catch (error) {
    setStatus(error.message || "Run failed", "error");
  } finally {
    clearApiKey();
    setBusy(false);
  }
}

function syncProviderControls() {
  const usesRemoteProvider = providerKind.value === "openai_responses";
  providerAuth.hidden = !usesRemoteProvider;
  runButton.disabled = providerKind.value === "openai_chat_completions";
  providerNote.textContent = PROVIDER_NOTE_BY_KIND[providerKind.value] || "";
  providerNote.classList.toggle(
    "is-warning",
    providerKind.value === "openai_chat_completions"
  );
  if (!usesRemoteProvider) {
    clearApiKey();
  }
}

function buildPayload() {
  const provider = { kind: providerKind.value };

  if (provider.kind === "openai_responses") {
    provider.api_key = valueOf("api-key");
    provider.base_url = valueOf("base-url") || null;
    provider.model = valueOf("model-name");
    provider.timeout_seconds = numberOrNull("timeout-seconds");
    provider.max_output_tokens = numberOrNull("max-output-tokens");
  }

  return {
    question: valueOf("question"),
    context: valueOf("context"),
    provider,
    runner: {
      max_cycles: numberOrNull("max-cycles"),
      max_probes_per_cycle: numberOrNull("max-probes"),
      stop_on_no_probes: document.querySelector("#stop-on-no-probes").checked,
      confidence_threshold: numberOrNull("confidence-threshold"),
      posterior_delta_threshold: numberOrNull("posterior-delta-threshold"),
    },
  };
}

async function parseJsonResponse(response) {
  const text = await response.text();
  if (!text) {
    return {};
  }
  try {
    return JSON.parse(text);
  } catch (error) {
    throw new Error("Server returned invalid JSON");
  }
}

function renderRun(payload) {
  runId.textContent = payload.run_id || "No run id";
  renderAnswer(payload.final_answer);
  renderBeliefs(payload.final_belief_state);
  renderTrace(payload.cycles || []);
}

function renderAnswer(answer) {
  answerPanel.innerHTML = "";

  if (!answer) {
    answerPanel.textContent = "No answer projection.";
    return;
  }

  answerPanel.appendChild(kv("Best hypothesis", answer.current_best_hypothesis));
  answerPanel.appendChild(kv("Answer", answer.answer));
  answerPanel.appendChild(kv("Posterior summary", answer.posterior_summary));
  answerPanel.appendChild(kv("Main uncertainty", answer.main_uncertainty));
  answerPanel.appendChild(kv("Weakest assumption", answer.weakest_assumption));
}

function renderBeliefs(beliefState) {
  beliefPanel.innerHTML = "";

  const hypotheses = beliefState?.hypotheses || [];
  if (!hypotheses.length) {
    beliefPanel.textContent = "No belief state yet.";
    return;
  }

  for (const hypothesis of hypotheses) {
    const row = document.createElement("div");
    row.className = "belief-row";

    const key = document.createElement("div");
    key.className = "belief-id";
    key.textContent = hypothesis.id || "Unknown";

    const value = document.createElement("div");
    value.className = "belief-copy";

    const metrics = document.createElement("div");
    metrics.className = "belief-metrics";
    metrics.textContent = [
      `prior ${formatNumber(hypothesis.prior)}`,
      `posterior ${formatNumber(hypothesis.posterior)}`,
    ].join(" | ");

    const statement = document.createElement("div");
    statement.className = "belief-statement";
    statement.textContent = hypothesis.statement || "";

    value.appendChild(metrics);
    value.appendChild(statement);
    row.appendChild(key);
    row.appendChild(value);
    beliefPanel.appendChild(row);
  }
}

function renderTrace(cycles) {
  tracePane.innerHTML = "";

  if (!cycles.length) {
    tracePane.innerHTML = '<div class="empty-state">Cycle details appear here after a run.</div>';
    return;
  }

  for (const cycle of cycles) {
    tracePane.appendChild(renderCycle(cycle));
  }
}

function renderCycle(cycle) {
  const details = document.createElement("details");
  details.className = "trace-item";
  details.open = true;

  const summary = document.createElement("summary");
  summary.textContent = `${cycle.cycle_id} (${cycle.signal_shape})`;
  details.appendChild(summary);

  const stack = document.createElement("div");
  stack.className = "trace-stack";
  stack.appendChild(block("Probes", cycle.probes));
  stack.appendChild(block("Signals", cycle.signals));
  stack.appendChild(block("Evidence", cycle.evidence_events));
  stack.appendChild(block("Belief updates", cycle.belief_updates));
  stack.appendChild(block("Hypothesis evolution", cycle.hypothesis_evolutions));
  stack.appendChild(block("Answer projection", cycle.answer_projection));
  details.appendChild(stack);

  return details;
}

function block(title, value) {
  const wrapper = document.createElement("section");
  const heading = document.createElement("h3");
  const pre = document.createElement("pre");

  heading.textContent = title;
  pre.textContent = JSON.stringify(value ?? null, null, 2);
  wrapper.appendChild(heading);
  wrapper.appendChild(pre);

  return wrapper;
}

function kv(label, value) {
  const row = document.createElement("div");
  row.className = "kv-row";

  const key = document.createElement("strong");
  const val = document.createElement("span");
  key.textContent = label;
  val.textContent = value ?? "";

  row.appendChild(key);
  row.appendChild(val);
  return row;
}

function valueOf(id) {
  return document.querySelector(`#${id}`).value.trim();
}

function numberOrNull(id) {
  const raw = valueOf(id);
  return raw === "" ? null : Number(raw);
}

function formatNumber(value) {
  return typeof value === "number" ? value.toFixed(3) : "n/a";
}

function setBusy(isBusy) {
  runButton.disabled =
    isBusy || providerKind.value === "openai_chat_completions";
  runButton.textContent = isBusy ? "Running..." : "Run autonomous loop";
}

function setStatus(message, tone) {
  statusBanner.textContent = message;
  statusBanner.classList.remove("busy", "ok", "error");
  if (tone) {
    statusBanner.classList.add(tone);
  }
}

function clearApiKey() {
  apiKeyField.value = "";
}
