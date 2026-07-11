const form = document.querySelector("#run-form");
const providerKind = document.querySelector("#provider-kind");
const providerAuth = document.querySelector("#provider-auth");
const providerNote = document.querySelector("#provider-note");
const statusBanner = document.querySelector("#status-banner");
const progressList = document.querySelector("#progress-list");
const progressState = document.querySelector("#progress-state");
const answerPanel = document.querySelector("#answer-panel");
const answerProjectionState = document.querySelector("#answer-projection-state");
const beliefPanel = document.querySelector("#belief-panel");
const tracePane = document.querySelector("#trace-pane");
const runId = document.querySelector("#run-id");
const runButton = document.querySelector("#run-button");

const PROVIDER_NOTE_BY_KIND = {
  deterministic: "Deterministic mode runs locally without a key.",
  openai_responses:
    "Responses-compatible providers only. Use Chat Completions for /chat/completions-compatible providers.",
  openai_chat_completions:
    "OpenAI-compatible Chat Completions settings are used only for this request. Check base URL, model, API key, and max output tokens.",
};
const OPENAI_COMPATIBLE_PROVIDER_KINDS = new Set([
  "openai_responses",
  "openai_chat_completions",
]);
const PROGRESS_LABEL_BY_EVENT = {
  run_started: "Run started",
  task_framing_started: "Framing task",
  task_framing_completed: "Task framed",
  initialization_completed: "Belief initialized",
  cycle_started: "Cycle started",
  probe_set_planned: "Probes planned",
  probe_execution_started: "Executing probes",
  signals_collected: "Signals collected",
  evidence_integration_started: "Integrating evidence",
  cycle_integrated: "Belief updated",
  run_completed: "Run completed",
  run_failed: "Run failed",
};
const providerControls = [
  providerKind,
  document.querySelector("#api-key"),
  document.querySelector("#base-url"),
  document.querySelector("#model-name"),
  document.querySelector("#timeout-seconds"),
  document.querySelector("#max-output-tokens"),
];
const streamedCycles = [];
let runInFlight = false;
let activeStreamRunId = null;
let taskFramingCompletedForRun = false;

providerKind.addEventListener("change", syncProviderControls);
form.addEventListener("submit", handleSubmit);

syncProviderControls();

async function handleSubmit(event) {
  event.preventDefault();
  if (runInFlight) return;

  runInFlight = true;
  setStatus("Running autonomous loop...", "busy");
  setBusy(true);
  clearRunOutput("running");
  resetRunProgress();

  try {
    const requestPayload = buildPayload();
    const response = await fetch('/api/runs/autonomous/stream', {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestPayload),
    });

    if (!response.ok) {
      const payload = await parseJsonResponse(response);
      throw new Error(payload.error?.message || "Run failed");
    }

    if (!response.body) {
      throw new Error("Server did not return a progress stream");
    }
    await consumeRunStream(response);
  } catch (error) {
    markProgressFailed();
    if (streamedCycles.length === 0) {
      clearRunOutput("failed");
    }
    setStatus(error.message || "Run failed", "error");
  } finally {
    runInFlight = false;
    setBusy(false);
  }
}

function syncProviderControls() {
  const usesRemoteProvider = OPENAI_COMPATIBLE_PROVIDER_KINDS.has(
    providerKind.value
  );
  providerAuth.hidden = !usesRemoteProvider;
  runButton.disabled = runInFlight;
  for (const control of providerControls) {
    control.disabled = runInFlight;
  }
  providerNote.textContent = PROVIDER_NOTE_BY_KIND[providerKind.value] || "";
  providerNote.classList.toggle("is-warning", false);
}

function buildPayload() {
  const provider = { kind: providerKind.value };

  if (
    provider.kind === "openai_responses" ||
    provider.kind === "openai_chat_completions"
  ) {
    provider.api_key = valueOf("api-key");
    provider.base_url = valueOf("base-url") || null;
    provider.model = valueOf("model-name");
    provider.timeout_seconds = numberOrNull("timeout-seconds");
    provider.max_output_tokens = numberOrNull("max-output-tokens");
  }

  return {
    question: valueOf("question"),
    task_context: valueOf("task-context"),
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

async function consumeRunStream(response, eventHandler = handleProgressEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let terminalEventSeen = false;
  let terminalFailure = null;
  let streamError = null;
  let cancelReader = false;
  let streamEnded = false;

  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (!line.trim()) continue;
        let event;
        try {
          event = JSON.parse(line);
        } catch (error) {
          throw new Error("Server returned an invalid progress event");
        }
        const isTerminalAdmissionOutcome =
          event.event === "task_admission_completed" &&
          ["needs_reframing", "out_of_scope"].includes(event.data?.result_type);
        const isTerminalEvent = ["run_completed", "run_failed"].includes(
          event.event
        ) || isTerminalAdmissionOutcome;
        if (isTerminalEvent) {
          terminalEventSeen = true;
          if (event.event === "run_failed") {
            terminalFailure = new Error(progressFailureMessage(event));
          }
        }
        eventHandler(event);
      }

      if (done) {
        streamEnded = true;
        break;
      }
    }

    if (buffer.trim()) {
      throw new Error("Progress stream ended with an incomplete event");
    }
    if (!terminalEventSeen) {
      throw new Error("Progress stream ended before the run completed");
    }
  } catch (error) {
    streamError = error;
    cancelReader = !streamEnded;
  } finally {
    if (cancelReader) {
      try {
        await reader.cancel();
      } catch (error) {
        streamError ||= error;
      }
    }
    reader.releaseLock();
  }

  if (streamError) {
    throw streamError;
  }
  if (terminalFailure) {
    throw terminalFailure;
  }
}

function handleProgressEvent(event) {
  const eventName = event.event;

  if (eventName === "run_started") {
    activeStreamRunId = event.run_id || null;
    taskFramingCompletedForRun = false;
  }
  if (
    eventName === "task_framing_completed" &&
    activeStreamRunId !== null &&
    event.run_id === activeStreamRunId
  ) {
    taskFramingCompletedForRun = true;
  }
  if (
    eventName === "initialization_completed" &&
    (
      activeStreamRunId === null ||
      event.run_id !== activeStreamRunId ||
      !taskFramingCompletedForRun
    )
  ) {
    throw new Error("Belief initialization arrived before task framing completed for this run");
  }

  renderProgressEvent(event);

  if (eventName === "run_started") {
    runId.textContent = event.run_id || "Run started";
    answerProjectionState.textContent = "Current";
  }
  if (eventName === "initialization_completed") {
    renderBeliefs(event.data?.belief_state);
  }
  if (eventName === "cycle_integrated") {
    upsertStreamedCycle(event.data);
    answerProjectionState.textContent = "Current";
    renderAnswer(event.data?.answer_projection);
    renderBeliefs(event.data?.belief_state);
    renderTrace(streamedCycles);
  }
  if (eventName === "run_completed") {
    answerProjectionState.textContent = "Final";
    renderRun(event.data || {});
    setStatus(
      `${event.data?.run?.status || "completed"}: ${event.data?.run?.regime || "autonomous"} / ${event.data?.stop_reason || "completed"}`,
      "ok"
    );
  }
  if (eventName === "run_failed") {
    answerProjectionState.textContent = streamedCycles.length
      ? "Current"
      : "Unavailable";
  }
}

function resetRunProgress() {
  streamedCycles.length = 0;
  activeStreamRunId = null;
  taskFramingCompletedForRun = false;
  progressList.innerHTML = "";
  progressState.textContent = "Running";
  answerProjectionState.textContent = "Current";
}

function markProgressFailed() {
  progressState.textContent = "Failed";
  const activeItem = progressList.querySelector(
    '.progress-item[data-state="active"]'
  );
  if (activeItem) {
    setProgressItemState(activeItem, "failed");
  }
  answerProjectionState.textContent = streamedCycles.length
    ? "Current"
    : "Unavailable";
}

function renderProgressEvent(event) {
  const activeItem = progressList.querySelector(
    '.progress-item[data-state="active"]'
  );
  if (activeItem) {
    setProgressItemState(activeItem, "complete");
  }

  const eventName = event.event;
  const isTerminalAdmissionOutcome =
    eventName === "task_admission_completed" &&
    ["needs_reframing", "out_of_scope"].includes(event.data?.result_type);
  const terminalState = eventName === "run_failed" ? "failed" :
    eventName === "run_completed" || isTerminalAdmissionOutcome
      ? "complete"
      : "active";
  const item = document.createElement("li");
  item.className = "progress-item";
  item.dataset.state = terminalState;

  const sequence = document.createElement("span");
  sequence.className = "progress-sequence";
  sequence.textContent = String(event.sequence ?? "-");

  const meta = document.createElement("span");
  meta.className = "progress-meta";
  const cycle = event.cycle_index != null ? ` | Cycle ${event.cycle_index}` : "";
  meta.textContent = `${PROGRESS_LABEL_BY_EVENT[eventName] || "Run update"}${cycle}`;

  const status = document.createElement("span");
  status.className = "progress-status";
  status.textContent = progressStatusLabel(terminalState);

  item.append(sequence, meta, status);
  progressList.appendChild(item);

  if (eventName === "run_completed" || isTerminalAdmissionOutcome) {
    progressState.textContent = "Completed";
  } else if (eventName === "run_failed") {
    progressState.textContent = "Failed";
  } else {
    progressState.textContent = "Running";
  }
}

function setProgressItemState(item, state) {
  item.dataset.state = state;
  item.querySelector(".progress-status").textContent = progressStatusLabel(state);
}

function progressStatusLabel(state) {
  if (state === "complete") return "Complete";
  if (state === "failed") return "Failed";
  return "Active";
}

function upsertStreamedCycle(cycle) {
  if (!cycle?.cycle_id) return;

  const existingIndex = streamedCycles.findIndex(
    (existing) => existing.cycle_id === cycle.cycle_id
  );
  if (existingIndex === -1) {
    streamedCycles.push(cycle);
  } else {
    streamedCycles[existingIndex] = cycle;
  }
}

function progressFailureMessage(event) {
  const message = event.data?.error?.message;
  return typeof message === "string" && message.trim() ? message : "Run failed";
}

function renderRun(payload) {
  runId.textContent = payload.run_id || "No run id";
  renderAnswer(payload.final_answer);
  renderBeliefs(payload.final_belief_state);
  renderTrace(payload.cycles || []);
}

function clearRunOutput(state = "running") {
  const failed = state === "failed";

  runId.textContent = failed ? "Last run failed." : "Run pending.";
  answerPanel.textContent = failed
    ? "Run failed. No answer projection."
    : "Run pending.";
  beliefPanel.textContent = failed
    ? "Run failed. No belief state."
    : "Run pending.";
  tracePane.innerHTML = `<div class="empty-state">${failed ? "Run failed. Cycle trace unavailable." : "Run pending."}</div>`;
}

function renderAnswer(answer) {
  answerPanel.innerHTML = "";

  if (!answer) {
    answerPanel.textContent = "No answer projection.";
    return;
  }

  answerPanel.appendChild(kv("Best answer / hypothesis", answer.current_best_hypothesis));
  answerPanel.appendChild(kv("Answer", answer.answer));
  answerPanel.appendChild(kv("Belief summary", answer.posterior_summary));
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

  const summary = beliefState?.posterior_summary || {};
  const relation = beliefState?.task_frame?.hypothesis_frame?.relation ||
    "exclusive_exhaustive";
  if (relation === "independent") {
    beliefPanel.appendChild(
      kv("Total credence (not normalized)", formatNumber(summary.total_active_credence))
    );
    beliefPanel.appendChild(
      kv(
        "Top / credence gap",
        `${summary.top_hypothesis || "n/a"} / ${formatNumber(summary.credence_gap)}`
      )
    );
  } else {
    beliefPanel.appendChild(
      kv("Posterior mass", formatNumber(summary.total_active_posterior))
    );
    beliefPanel.appendChild(
      kv(
        "Top / posterior gap",
        `${summary.top_hypothesis || "n/a"} / ${formatNumber(summary.posterior_gap)}`
      )
    );
  }
  beliefPanel.appendChild(
    kv("Current uncertainty", beliefState?.uncertainty_summary || "")
  );

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
      `${relation === "independent" ? "credence" : "posterior"} ${formatNumber(hypothesis.posterior)}`,
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
  const boundaryStatus = cycle.cycle?.boundary_status || "unknown";
  summary.textContent = `${cycle.cycle_id} (${cycle.signal_shape}) | ${boundaryStatus}`;
  details.appendChild(summary);

  const stack = document.createElement("div");
  stack.className = "trace-stack";
  stack.appendChild(
    block("Cycle lifecycle", {
      boundary_status: cycle.cycle?.boundary_status,
      started_at: cycle.cycle?.started_at,
      boundary_closed_at: cycle.cycle?.boundary_closed_at,
      completed_at: cycle.cycle?.completed_at,
    })
  );
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
  runButton.textContent = isBusy ? "Running..." : "Run autonomous loop";
  syncProviderControls();
}

function setStatus(message, tone) {
  statusBanner.textContent = message;
  statusBanner.classList.remove("busy", "ok", "error");
  if (tone) {
    statusBanner.classList.add(tone);
  }
}
