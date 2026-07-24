// ============================================================
// Intelligent Healthcare Diagnostic Assistant — frontend
// ============================================================

const state = {
  meta: null,
  demoPatients: [],
};

const DIAGNOSIS_LABELS = {
  covid19: "COVID-19",
  common_cold: "Common cold",
  cardiac_event: "Cardiac event",
  flu: "Flu",
  dengue: "Dengue",
  diabetes: "Diabetes",
  tuberculosis: "Tuberculosis",
  meningitis: "Meningitis",
  healthy: "Healthy",
};

function formatDiagnosis(raw) {
  if (!raw) return "Unknown";
  const key = String(raw).toLowerCase().trim();
  if (DIAGNOSIS_LABELS[key]) return DIAGNOSIS_LABELS[key];
  return key
    .replace(/_suspected$|_confirmed$/, "")
    .split(/[_\s]+/)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

function pct(value) {
  return `${Math.round(value * 100)}%`;
}

function el(tag, className, html) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (html !== undefined) node.innerHTML = html;
  return node;
}

// ---------------------------------------------------------
// Tabs
// ---------------------------------------------------------
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
    if (btn.dataset.tab === "evaluation") loadEvaluation();
  });
});

// ---------------------------------------------------------
// Agent status pill
// ---------------------------------------------------------
function setAgentBusy(busy) {
  const pill = document.getElementById("agentPill");
  const text = document.getElementById("agentPillText");
  pill.classList.toggle("busy", busy);
  text.textContent = busy ? "Agent reasoning\u2026" : "Agent idle";
}

// ---------------------------------------------------------
// Meta + demo patients + symptom grid
// ---------------------------------------------------------
async function loadMeta() {
  const res = await fetch("/api/meta");
  const data = await res.json();
  state.meta = data;

  const grid = document.getElementById("symptomGrid");
  grid.innerHTML = "";
  data.symptoms.forEach((s) => {
    const label = el("label", "symptom-item");
    label.innerHTML = `<input type="checkbox" value="${s.id}"> ${s.label}`;
    grid.appendChild(label);
  });

  renderPeas(data.peas);
  renderArchitecture(data.modules);
}

async function loadDemoPatients() {
  const res = await fetch("/api/demo-patients");
  const data = await res.json();
  state.demoPatients = data;

  const row = document.getElementById("demoRow");
  row.innerHTML = "";
  data.forEach((p, idx) => {
    const btn = el("button", "chip-btn", `${p.patient_id.replace("DEMO-", "Patient ")}`);
    btn.type = "button";
    btn.title = `Expected: ${formatDiagnosis(p.expected_diagnosis)}`;
    btn.addEventListener("click", () => fillFormFromPatient(p));
    row.appendChild(btn);
  });
}

function fillFormFromPatient(p) {
  document.getElementById("fName").value = p.name || "";
  document.getElementById("fGender").value = p.gender || "Unknown";
  document.getElementById("fAge").value = p.age;
  document.getElementById("fTemp").value = p.temperature;
  document.getElementById("fHr").value = p.heart_rate;
  document.getElementById("fBp").value = p.blood_pressure;
  document.getElementById("fExtra").value = "";

  const known = new Set((state.meta?.symptoms || []).map((s) => s.id));
  const checks = document.querySelectorAll("#symptomGrid input[type=checkbox]");
  const extras = [];
  checks.forEach((c) => (c.checked = p.symptoms.includes(c.value)));
  p.symptoms.forEach((s) => {
    if (!known.has(s)) extras.push(s);
  });
  document.getElementById("fExtra").value = extras.join(", ");

  document.getElementById("patientForm").dataset.patientId = p.patient_id;
  document.getElementById("patientForm").dataset.expected = p.expected_diagnosis || "";
}

// ---------------------------------------------------------
// Form submit -> /api/diagnose
// ---------------------------------------------------------
document.getElementById("patientForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const errorBox = document.getElementById("formError");
  errorBox.style.display = "none";

  const symptoms = Array.from(
    document.querySelectorAll("#symptomGrid input[type=checkbox]:checked")
  ).map((c) => c.value);

  const payload = {
    patient_id: e.target.dataset.patientId || undefined,
    expected_diagnosis: e.target.dataset.expected || undefined,
    name: document.getElementById("fName").value,
    gender: document.getElementById("fGender").value,
    age: document.getElementById("fAge").value,
    temperature: document.getElementById("fTemp").value,
    heart_rate: document.getElementById("fHr").value,
    blood_pressure: document.getElementById("fBp").value,
    symptoms,
    extra_symptoms: document.getElementById("fExtra").value,
  };

  const submitBtn = document.getElementById("submitBtn");
  const submitLabel = document.getElementById("submitLabel");
  submitBtn.disabled = true;
  submitLabel.innerHTML = '<span class="spinner"></span> Running pipeline\u2026';
  setAgentBusy(true);

  try {
    const res = await fetch("/api/diagnose", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) {
      errorBox.textContent = data.error || "Something went wrong.";
      errorBox.style.display = "block";
    } else {
      renderResults(data);
    }
  } catch (err) {
    errorBox.textContent = "Could not reach the diagnostic server.";
    errorBox.style.display = "block";
  } finally {
    submitBtn.disabled = false;
    submitLabel.textContent = "Run diagnostic agent";
    setAgentBusy(false);
  }
});

// ---------------------------------------------------------
// Results rendering
// ---------------------------------------------------------
function topEntries(pairs, n = 4) {
  return pairs
    .map(([k, v]) => [k, typeof v === "number" ? v : parseFloat(v)])
    .sort((a, b) => b[1] - a[1])
    .slice(0, n);
}

function barRows(pairs) {
  const max = Math.max(...pairs.map(([, v]) => v), 0.0001);
  return pairs
    .map(
      ([label, value]) => `
      <div class="bar-row">
        <div class="bar-label">${formatDiagnosis(label)}</div>
        <div class="bar-track"><div class="bar-fill" style="width:${(value / max) * 100}%"></div></div>
        <div class="bar-value">${pct(value)}</div>
      </div>`
    )
    .join("");
}

function moduleTopPairs(name, result) {
  if (name === "KnowledgeBase" && result.csv_scores) {
    return topEntries(Object.entries(result.csv_scores));
  }
  if (name === "BayesianNet" && result.ranked_diagnoses) {
    return topEntries(result.ranked_diagnoses);
  }
  if (name === "MLClassifier" && result.top5) {
    return topEntries(result.top5);
  }
  if (name === "NeuralNetwork" && result.all_probs) {
    return topEntries(Object.entries(result.all_probs));
  }
  return [];
}

const MODULE_ORDER = ["KnowledgeBase", "BayesianNet", "MLClassifier", "NeuralNetwork"];

function renderResults(report) {
  const col = document.getElementById("resultsColumn");
  col.innerHTML = "";

  const urgency = report.urgency || "LOW";

  // --- Diagnosis banner ---
  const banner = el("div", `diagnosis-banner banner-${urgency}`);
  banner.innerHTML = `
    <div>
      <div class="label">Aggregated diagnosis · Patient ${report.patient_id}</div>
      <div class="value">${formatDiagnosis(report.diagnosis)}</div>
      <div class="sub">Confidence ${pct(report.confidence)}${
    report.expected_diagnosis
      ? ` &middot; Expected: ${formatDiagnosis(report.expected_diagnosis)}`
      : ""
  } &middot; Next action: ${(report.next_action || "").replaceAll("_", " ")}</div>
    </div>
    <div class="urgency-badge urgency-${urgency}">${urgency} URGENCY</div>
  `;
  col.appendChild(banner);

  // --- Pipeline stepper ---
  const stepperCard = el("div", "card");
  const steps = ["Perceive", "Reason (4 modules)", "Aggregate", "Plan treatment"];
  const stepper = el("div", "stepper");
  steps.forEach((s, i) => {
    const step = el("div", "step", `<span class="num">${i + 1}</span> ${s}`);
    stepper.appendChild(step);
    if (i < steps.length - 1) stepper.appendChild(el("div", "step-connector"));
  });
  stepperCard.appendChild(el("div", "card-title", "Agent pipeline (perceive &rarr; think &rarr; act)"));
  stepperCard.appendChild(stepper);
  col.appendChild(stepperCard);

  // --- Module comparison grid ---
  const moduleCard = el("div", "card");
  moduleCard.appendChild(el("div", "card-title", "Diagnostic module comparison"));
  const grid = el("div", "module-grid");
  MODULE_ORDER.forEach((name) => {
    const result = report.module_results?.[name];
    if (!result) return;
    const meta = state.meta?.modules?.[name] || { label: name };
    const pairs = moduleTopPairs(name, result);
    const card = el("div", "module-card");
    card.innerHTML = `
      <div class="module-card-head">
        <div>
          <div class="module-name">${meta.label}</div>
          <div class="module-diagnosis">${formatDiagnosis(result.diagnosis)}</div>
        </div>
        <div class="module-confidence">${pct(result.confidence || 0)}</div>
      </div>
      ${barRows(pairs)}
    `;
    grid.appendChild(card);
  });
  moduleCard.appendChild(grid);
  col.appendChild(moduleCard);

  // --- Fuzzy severity gauge ---
  const fuzzy = report.severity || report.module_results?.Fuzzy;
  if (fuzzy && typeof fuzzy.severity_score === "number") {
    const gaugeCard = el("div", "card");
    gaugeCard.appendChild(el("div", "card-title", "Fuzzy severity assessment"));
    const wrap = el("div", "gauge-wrap");

    const readout = el("div", "gauge-readout");
    readout.innerHTML = `
      <div class="gauge-score">${fuzzy.severity_score.toFixed(1)}</div>
      <div class="gauge-label urgency-${fuzzy.severity_label}" style="border-radius:999px;padding:4px 10px;display:inline-block;">${fuzzy.severity_label}</div>
    `;
    wrap.appendChild(buildGaugeSvg(fuzzy.severity_score));
    wrap.appendChild(readout);

    const memberBars = el("div", "membership-bars");
    const memPairs = Object.entries(fuzzy.rule_strengths || {}).sort((a, b) => b[1] - a[1]);
    memberBars.innerHTML = `<div class="hint" style="margin-bottom:6px;">Rule activation strength</div>${barRows(memPairs)}`;
    wrap.appendChild(memberBars);

    gaugeCard.appendChild(wrap);
    col.appendChild(gaugeCard);
  }

  // --- Treatment plan ---
  const plan = report.treatment_plan;
  if (plan && Array.isArray(plan.plan)) {
    const planCard = el("div", "card");
    planCard.appendChild(el("div", "card-title", "STRIPS treatment plan"));
    const meta = el("div", "plan-meta");
    meta.innerHTML = `
      <div><b>${plan.steps}</b> steps</div>
      <div><b>${formatDiagnosis(plan.diagnosis)}</b> diagnosis</div>
      <div><b>${plan.urgency}</b> urgency</div>
    `;
    planCard.appendChild(meta);
    const list = el("ul", "timeline");
    plan.plan.forEach((step) => {
      const li = el("li");
      li.innerHTML = `
        <div class="step-index">${step.step}</div>
        <div class="step-name">${step.action.replace(/([a-z])([A-Z])/g, "$1 $2")}</div>
        <div class="step-duration">${step.duration}</div>
      `;
      list.appendChild(li);
    });
    planCard.appendChild(list);
    col.appendChild(planCard);
  }

  // --- Recommendations ---
  if (Array.isArray(report.recommendations) && report.recommendations.length) {
    const recCard = el("div", "card");
    recCard.appendChild(el("div", "card-title", "Recommendations"));
    const list = el("ul", "rec-list");
    report.recommendations.forEach((r) => list.appendChild(el("li", "", r)));
    recCard.appendChild(list);
    col.appendChild(recCard);
  }

  // --- Agent action log (collapsible) ---
  if (Array.isArray(report.action_log)) {
    const logCard = el("div", "card");
    const head = el("div", "collapsible-head open");
    head.innerHTML = `<div class="card-title" style="margin:0;">Agent action log</div><span class="chev">&#9662;</span>`;
    const body = el("div", "collapsible-body");
    const log = el("div", "log-panel");
    log.innerHTML = report.action_log.map((l) => `<div>${l}</div>`).join("");
    body.appendChild(log);
    head.addEventListener("click", () => {
      head.classList.toggle("open");
      body.classList.toggle("hidden");
    });
    logCard.appendChild(head);
    logCard.appendChild(body);
    col.appendChild(logCard);
  }
}

function buildGaugeSvg(score) {
  const clamped = Math.max(0, Math.min(100, score));
  const angle = (clamped / 100) * 180; // 0..180 degrees across the semicircle
  const rad = (Math.PI * (180 - angle)) / 180;
  const cx = 70, cy = 70, r = 58;
  const nx = cx + r * Math.cos(rad);
  const ny = cy - r * Math.sin(rad);

  const wrapper = document.createElement("div");
  wrapper.innerHTML = `
  <svg width="140" height="82" viewBox="0 0 140 82">
    <path d="M12 70 A58 58 0 0 1 34.6 24.6" stroke="#29875b" stroke-width="12" fill="none" stroke-linecap="round"/>
    <path d="M34.6 24.6 A58 58 0 0 1 70 12" stroke="#b58900" stroke-width="12" fill="none" stroke-linecap="round"/>
    <path d="M70 12 A58 58 0 0 1 105.4 24.6" stroke="#d8791f" stroke-width="12" fill="none" stroke-linecap="round"/>
    <path d="M105.4 24.6 A58 58 0 0 1 128 70" stroke="#cc3333" stroke-width="12" fill="none" stroke-linecap="round"/>
    <line x1="${cx}" y1="${cy}" x2="${nx}" y2="${ny}" stroke="#16262c" stroke-width="3" stroke-linecap="round"/>
    <circle cx="${cx}" cy="${cy}" r="5" fill="#16262c"/>
  </svg>`;
  return wrapper.firstElementChild;
}

// ---------------------------------------------------------
// Architecture tab
// ---------------------------------------------------------
function renderPeas(peas) {
  const grid = document.getElementById("peasGrid");
  grid.innerHTML = "";
  const rows = [
    ["Performance", peas.performance],
    ["Environment", peas.environment],
    ["Actuators", peas.actuators],
    ["Sensors", peas.sensors],
  ];
  rows.forEach(([title, text]) => {
    const card = el("div", "peas-card");
    card.innerHTML = `<h4>${title}</h4><p>${text}</p>`;
    grid.appendChild(card);
  });
}

function renderArchitecture(modules) {
  const grid = document.getElementById("archGrid");
  grid.innerHTML = "";
  Object.entries(modules).forEach(([key, m]) => {
    const card = el("div", `arch-card ${m.kind === "support" ? "support" : ""}`);
    card.innerHTML = `
      <span class="arch-tag">${m.tag}</span>
      <h4>${m.label}</h4>
      <p>${m.description}</p>
    `;
    grid.appendChild(card);
  });
}

// ---------------------------------------------------------
// Evaluation tab
// ---------------------------------------------------------
async function loadEvaluation() {
  const note = document.getElementById("evalNote");
  const content = document.getElementById("evalContent");
  note.textContent = "Loading evaluation report\u2026";
  const res = await fetch("/api/evaluation");
  const data = await res.json();

  if (!data.available) {
    note.textContent = "No evaluation report found yet.";
    content.innerHTML = "";
    return;
  }

  note.textContent = `Based on ${data.patients_evaluated} evaluated patients.`;
  const m = data.metrics || {};
  content.innerHTML = `
    <div class="metric-grid">
      <div class="metric-tile"><div class="val">${((m.accuracy || 0) * 100).toFixed(0)}%</div><div class="lab">Accuracy</div></div>
      <div class="metric-tile"><div class="val">${((m.precision || 0) * 100).toFixed(0)}%</div><div class="lab">Precision</div></div>
      <div class="metric-tile"><div class="val">${((m.recall || 0) * 100).toFixed(0)}%</div><div class="lab">Recall</div></div>
      <div class="metric-tile"><div class="val">${((m.f1_score || 0) * 100).toFixed(0)}%</div><div class="lab">F1 score</div></div>
    </div>
    <div class="image-grid" id="imageGrid"></div>
  `;

  const imgGrid = document.getElementById("imageGrid");
  const captions = {
    confusion_matrix: "Full-system confusion matrix",
    module_comparison: "Average confidence per module",
    ml_evaluation: "ML classifier evaluation",
    nn_training: "Neural network training curves",
  };
  Object.entries(data.images || {}).forEach(([key, src]) => {
    const card = el("div", "image-card");
    card.innerHTML = `<img src="${src}?t=${Date.now()}" alt="${key}"><div class="cap">${captions[key] || key}</div>`;
    imgGrid.appendChild(card);
  });
}

document.getElementById("regenBtn").addEventListener("click", async () => {
  const note = document.getElementById("evalNote");
  note.textContent = "Regenerating evaluation report from the 5 demo patients\u2026";
  const res = await fetch("/api/evaluation/regenerate", { method: "POST" });
  const data = await res.json();
  if (data.error) {
    note.textContent = data.error;
    return;
  }
  loadEvaluation();
});

// ---------------------------------------------------------
// Init
// ---------------------------------------------------------
loadMeta();
loadDemoPatients();
