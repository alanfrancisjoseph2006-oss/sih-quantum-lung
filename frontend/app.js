/* QuantumLung front-end - SIH 2026 PS 26139 - Team CodeNova
   Plain ES2020, no build step: one file, easy to audit during a demo. */

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s ?? "").replace(/[&<>"']/g, c =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = v => (v == null || Number.isNaN(v)) ? "—" : (100 * v).toFixed(1) + "%";
const num = (v, d = 3) => (v == null || Number.isNaN(v)) ? "—" : Number(v).toFixed(d);

const STATE = { meta: null, comparison: null, localization: null, education: null };

/* ---------------- navigation ---------------- */
$$("nav.tabs button").forEach(b => b.onclick = () => {
  $$("nav.tabs button").forEach(x => x.classList.toggle("active", x === b));
  $$(".page").forEach(p => p.classList.toggle("active", p.id === "page-" + b.dataset.page));
  window.scrollTo({ top: 0, behavior: "smooth" });
});

/* ---------------- boot ---------------- */
async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}

(async function boot() {
  try { STATE.meta = await getJSON("/api/meta"); } catch (e) { console.warn("meta", e); }
  try { STATE.comparison = await getJSON("/api/model-comparison"); } catch (e) { console.warn("cmp", e); }
  try { STATE.localization = await getJSON("/api/localization-validation"); } catch (e) { console.warn("loc", e); }
  try { STATE.education = await getJSON("/api/education"); } catch (e) { console.warn("edu", e); }
  try { STATE.lidc = await getJSON("/api/lidc-experiment"); } catch (e) { console.warn("lidc", e); }
  renderHome(); renderArchitecture(); renderModels(); renderQuantum(); renderAbout(); renderSteps();
})();

/* ================= HOME ================= */
function deployedRow() {
  const c = STATE.comparison, d = c?.deployed;
  if (!d) return null;
  const m = c.backbones?.[d.backbone]?.models?.[d.primary_model];
  return { d, m, bb: c.backbones?.[d.backbone] };
}

function renderHome() {
  const r = deployedRow();
  const t = r?.m?.test || {};
  const loc = STATE.localization?.results || {};
  const locKey = Object.keys(loc)[0];
  const stats = [
    { v: t.f1_macro != null ? pct(t.f1_macro) : "—", k: "Test macro-F1", cls: "" },
    { v: t.malignant_sensitivity != null ? pct(t.malignant_sensitivity) : "—", k: "Malignant sensitivity", cls: "ok" },
    { v: t.roc_auc_macro_ovr != null ? num(t.roc_auc_macro_ovr) : "—", k: "Test ROC-AUC (macro)", cls: "" },
    { v: (STATE.meta?.engine?.n_qubits ?? "—") + " qubits", k: "Deployed quantum circuit", cls: "q" },
  ];
  $("#home-stats").innerHTML = stats.map(s => `
    <div class="panel stat"><div class="v ${s.cls}">${esc(s.v)}</div><div class="k">${esc(s.k)}</div></div>`).join("");

  const qa = r?.bb?.quantum_advantage_test?.hybrid_vs_classical
    || r?.bb?.quantum_advantage_test?.quantum_vs_classical;
  const lk = locKey ? loc[locKey] : null;

  $("#home-findings").innerHTML = `
    <div class="notice warn"><b>Leakage is the real story in this dataset.</b><br>
      The IQ-OTH/NCCD slices come from only 110 patients but carry <i>no patient IDs</i>.
      Under a naive random split, <b>~55% of test images have a near-duplicate sibling in
      training</b>. We built pseudo-patient groups and split on them, so our numbers are lower
      than the 99%+ usually reported — and honest.</div>
    <div class="notice q"><b>No quantum advantage was demonstrated.</b><br>
      ${qa ? `Paired bootstrap on the test split: Δmacro-F1 = <b>${qa.mean_diff >= 0 ? "+" : ""}${num(qa.mean_diff)}</b>
      (95% CI ${num(qa.ci95_lo)} to ${num(qa.ci95_hi)}), <b>p = ${num(qa.p_value)}</b>.
      The quantum branch performs comparably to classical models, not better.`
      : "Significance testing was run; results are on the Model Comparison page."}</div>
    <div class="notice"><b>Attention maps were tested, not just displayed.</b><br>
      ${lk ? `Scored against ${lk.n_images} real lesion boxes recovered from DeepLesion overlays,
      Grad-CAM++ localisation was <b>no better than chance</b> (hit rate
      ${num(lk.pointing_game_hit_rate)}). That is exactly why we call them attention maps,
      never tumour boundaries.` : "Localisation was validated against real lesion annotations."}</div>
    ${STATE.lidc ? `<div class="notice warn"><b>We got more data. It made things worse.</b><br>
      Downloaded ${STATE.lidc.acquired.series_downloaded} LIDC-IDRI series
      (${STATE.lidc.acquired.gigabytes} GB) and added
      <b>${STATE.lidc.acquired.slices_extracted} labelled slices</b>, taking benign training
      images from ${STATE.lidc.training_set_change.benign.before} to
      <b>${STATE.lidc.training_set_change.benign.after}</b>. Macro-F1 on the unchanged test set
      <b>fell ${num(Math.abs(STATE.lidc.results_on_unchanged_iqoth_test.Hybrid_meta.delta))}</b>
      (p = ${num(STATE.lidc.results_on_unchanged_iqoth_test.Hybrid_meta.p_value)}). Domain shift
      beat the extra examples, so we did <b>not</b> deploy it.</div>` : ""}`;
}

/* ================= ARCHITECTURE DIAGRAM ================= */
function renderArchitecture() {
  const box = (x, y, w, h, label, sub, fill, stroke) => `
    <rect x="${x}" y="${y}" width="${w}" height="${h}" rx="9" fill="${fill}" stroke="${stroke}" stroke-width="1.3"/>
    <text x="${x + w / 2}" y="${y + (sub ? 22 : h / 2 + 4)}" text-anchor="middle" fill="#e8eef8"
      font-size="12.5" font-weight="600" font-family="Inter,sans-serif">${label}</text>
    ${sub ? `<text x="${x + w / 2}" y="${y + 39}" text-anchor="middle" fill="#8fa3c2"
      font-size="10.5" font-family="Inter,sans-serif">${sub}</text>` : ""}`;
  const arrow = (x1, y1, x2, y2, c = "#2a3a5c") =>
    `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${c}" stroke-width="1.6" marker-end="url(#ah)"/>`;

  $("#arch-diagram").innerHTML = `
  <div style="overflow-x:auto">
  <svg viewBox="0 0 1180 400" style="width:100%;min-width:900px;height:auto">
    <defs>
      <marker id="ah" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">
        <path d="M0,0 L0,6 L8,3 z" fill="#3d5480"/></marker>
    </defs>

    ${box(20, 158, 118, 54, "CT Image", "PNG · JPG · DICOM", "#111a2b", "#2a3a5c")}
    ${arrow(140, 185, 172, 185)}
    ${box(174, 158, 128, 54, "Preprocess", "denoise · CLAHE · letterbox", "#111a2b", "#2a3a5c")}
    ${arrow(304, 185, 336, 185)}
    ${box(338, 158, 132, 54, "CNN Backbone", "EfficientNet-B3", "#0e2033", "#2f6d8f")}
    ${arrow(472, 185, 504, 185)}
    ${box(506, 158, 118, 54, "Embedding", "256-d", "#0e2033", "#2f6d8f")}

    ${arrow(566, 156, 566, 122)} ${arrow(566, 214, 566, 248)}
    ${box(626, 60, 130, 52, "Grad-CAM++", "attention map", "#1c1630", "#6d4f9e")}
    ${arrow(566, 122, 624, 90)}

    ${box(506, 248, 118, 50, "PCA → 32", "train-fit only", "#111a2b", "#2a3a5c")}

    ${arrow(626, 273, 664, 232)} ${arrow(626, 273, 664, 314)}
    ${box(666, 200, 156, 62, "Classical Heads", "RF · SVM · XGB · LogReg", "#0e2033", "#2f6d8f")}
    ${box(666, 288, 156, 62, "Quantum Circuit", "angle enc · VQC · ⟨Z⟩", "#1c1630", "#8b5cf6")}

    ${arrow(824, 231, 866, 262)} ${arrow(824, 319, 866, 288)}
    ${box(868, 244, 132, 62, "Hybrid Fusion", "4 strategies", "#151f38", "#5b6ee0")}

    ${arrow(1002, 275, 1036, 275)}
    ${box(1038, 244, 122, 62, "Adaptive Router", "val-selected", "#111a2b", "#38bdf8")}

    ${arrow(1099, 242, 1099, 130)}
    ${box(1010, 68, 150, 62, "Report + XAI", "risk · region · limits", "#0d2a20", "#34d399")}
    ${arrow(756, 86, 1008, 92)}

    <text x="20" y="382" fill="#6b7f9e" font-size="10.5" font-family="Inter,sans-serif">
      Dashed capabilities intentionally absent: no staging model (no stage labels) · no supervised segmentation (no masks)
    </text>
  </svg></div>`;
}

/* ================= STEPS ================= */
const STEPS = ["Upload", "Decode", "Preprocess", "CNN", "Quantum", "Fusion", "Grad-CAM", "Report"];
function renderSteps(active = -1, done = -1) {
  $("#steps").innerHTML = STEPS.map((s, i) =>
    `<div class="step ${i <= done ? "done" : ""} ${i === active ? "active" : ""}">${esc(s)}</div>`).join("");
}

/* ================= UPLOAD ================= */
const drop = $("#drop"), fileInput = $("#fileInput");
drop.onclick = () => fileInput.click();
drop.ondragover = e => { e.preventDefault(); drop.classList.add("over"); };
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = e => {
  e.preventDefault(); drop.classList.remove("over");
  if (e.dataTransfer.files[0]) analyze(e.dataTransfer.files[0]);
};
fileInput.onchange = e => { if (e.target.files[0]) analyze(e.target.files[0]); };

async function analyze(file) {
  $("#results").innerHTML = "";
  $("#analyze-msg").innerHTML = `<div style="text-align:center"><div class="spinner"></div>
    <p class="muted" style="margin-top:10px">Analysing <b>${esc(file.name)}</b> …</p></div>`;

  let i = 0;
  renderSteps(0, -1);
  const timer = setInterval(() => { i = Math.min(i + 1, STEPS.length - 2); renderSteps(i, i - 1); }, 190);

  const fd = new FormData();
  fd.append("file", file);
  try {
    const res = await fetch("/api/analyze", { method: "POST", body: fd });
    clearInterval(timer);
    const data = await res.json();
    if (!res.ok) {
      renderSteps(-1, -1);
      $("#analyze-msg").innerHTML = `<div class="err"><b>Analysis failed (${res.status}).</b><br>${esc(data.detail || "Unknown error")}</div>`;
      return;
    }
    renderSteps(-1, STEPS.length - 1);
    $("#analyze-msg").innerHTML = `<p class="muted">Completed in ${num(data.runtime_seconds, 2)} s · ${esc(file.name)}</p>`;
    renderResults(data);
    $("#results").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) {
    clearInterval(timer); renderSteps(-1, -1);
    $("#analyze-msg").innerHTML = `<div class="err"><b>Could not reach the analysis server.</b><br>${esc(e.message)}</div>`;
  }
}

/* ================= RESULTS DASHBOARD ================= */
function probBars(probs, quantum = false) {
  return Object.entries(probs).map(([c, p]) => `
    <div class="kv"><span class="k">${esc(c)}</span>
      <span style="display:flex;align-items:center;gap:9px;flex:1;max-width:210px">
        <span class="bar ${quantum ? "q" : ""}" style="flex:1"><i style="width:${(100 * p).toFixed(1)}%"></i></span>
        <span class="v" style="min-width:52px">${pct(p)}</span></span></div>`).join("");
}

function branchCard(r, kind) {
  if (!r) return "";
  const q = kind === "quantum";
  return `<div class="panel tight">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
      <b style="font-size:.9rem">${esc(r.model)}</b>
      <span class="tag ${q ? "q" : "ok"}">${esc(r.predicted_class)}</span></div>
    ${probBars(r.probabilities, q)}</div>`;
}

function renderResults(d) {
  const fp = d.final_prediction;
  const band = { Low: "v-low", Intermediate: "v-mid", High: "v-high" }[fp.risk_band] || "v-mid";
  const ex = d.explainability, reg = ex.region || {}, mp = d.measured_parameters, ci = d.clinical_information;
  const perf = d.model_performance || {};

  const html = `
  <!-- 1-9 verdict -->
  <div class="panel">
    <div class="eyebrow">Final Selected Prediction</div>
    <div class="verdict">
      <div class="verdict-badge ${band}">
        ${esc(fp.display)}
        <div class="sub">${esc(fp.risk_band)} risk band · confidence ${pct(fp.confidence)}</div>
      </div>
      <div style="flex:1;min-width:280px">${probBars(fp.probabilities)}</div>
    </div>
    <div class="notice info" style="margin-top:14px">
      <b>Selected model: ${esc(fp.selected_model)}.</b> ${esc(fp.selection_reason)}</div>
    <div class="notice"><b>Branch agreement: ${pct(fp.branch_agreement)}.</b> ${esc(fp.branch_agreement_note)}</div>
    ${fp.decision_policy_applied ? `<div class="notice ${fp.decision_policy_changed_outcome ? "warn" : ""}">
      <b>Cost-sensitive decision policy ${fp.decision_policy_changed_outcome
        ? "changed this outcome" : "applied"}.</b>
      ${esc(fp.decision_policy_note)}
      ${fp.decision_policy_changed_outcome ? `<br>Raw model output was
        <b>${esc(fp.raw_model_class)}</b>; after the policy it is
        <b>${esc(fp.predicted_class)}</b>.` : ""}</div>` : ""}
  </div>

  <!-- 1,2 images -->
  <div class="panel">
    <div class="eyebrow">Explainability — ${esc(ex.method)}</div>
    <h3>${esc(ex.label)}</h3>
    <div class="notice warn">${esc(ex.caveat)}</div>
    <div class="grid g3" style="margin-top:14px">
      <div class="imgcard"><img src="${ex.images.original}" alt="Preprocessed CT">
        <div class="cap">1 · Preprocessed CT (model input)</div></div>
      <div class="imgcard"><img src="${ex.images.heatmap}" alt="Attention heatmap">
        <div class="cap">2 · Attention heatmap</div></div>
      <div class="imgcard"><img src="${ex.images.overlay}" alt="Overlay">
        <div class="cap">3 · Overlay — regions influencing the prediction</div></div>
    </div>
  </div>

  <!-- 6,7,8 branches -->
  <div class="panel">
    <div class="eyebrow">All Model Branches</div>
    <h3>Every branch, reported side by side</h3>
    <div class="grid g2">
      <div>
        <p class="muted" style="margin-bottom:8px">Classical deep learning</p>
        ${branchCard(d.branch_results.classical_deep_learning, "cnn")}
        <p class="muted" style="margin:14px 0 8px">Hybrid fusion</p>
        ${branchCard(d.branch_results.hybrid, "hybrid") || '<div class="notice">Hybrid branch unavailable.</div>'}
      </div>
      <div>
        <p class="muted" style="margin-bottom:8px">Quantum branch</p>
        ${branchCard(d.branch_results.quantum, "quantum")}
        <p class="muted" style="margin:14px 0 8px">Classical ML heads on CNN embedding</p>
        <div class="panel tight">
          <table class="data"><thead><tr><th>Model</th><th>Prediction</th><th>Conf.</th></tr></thead><tbody>
          ${d.branch_results.classical_ml.map(m => `<tr><td>${esc(m.model)}</td>
            <td>${esc(m.predicted_class)}</td><td class="num">${pct(m.confidence)}</td></tr>`).join("")}
          </tbody></table>
        </div>
      </div>
    </div>
  </div>

  <!-- 10,11 location + parameters -->
  <div class="grid g2">
    <div class="panel">
      <div class="eyebrow">Suspected Abnormality Location</div>
      <h3>${reg.available ? esc(reg.location_description) : "No focal attention region resolved"}</h3>
      ${reg.available ? `
      <div class="kv"><span class="k">Laterality</span><span class="v">${esc(reg.laterality)}</span></div>
      <div class="kv"><span class="k">Zone</span><span class="v">${esc(reg.zone)}</span></div>
      <div class="kv"><span class="k">Attention peak (x, y)</span><span class="v">${reg.peak_xy?.join(", ")}</span></div>
      <div class="kv"><span class="k">Attended-region box</span><span class="v">${reg.bbox_xyxy?.join(", ")}</span></div>
      <div class="kv"><span class="k">Attended area</span><span class="v">${pct(reg.attended_area_fraction)} of image</span></div>
      <div class="kv"><span class="k">Attention blobs</span><span class="v">${reg.n_attention_blobs}</span></div>` : ""}
      <div class="notice warn" style="margin-top:12px"><b>Not a tumour boundary.</b>
      Laterality follows radiological convention (patient's left appears on the image right).
      Lobe-level localisation is <b>not</b> attempted: the dataset provides no lobe labels, so any
      lobe claim would be guesswork.</div>
    </div>

    <div class="panel">
      <div class="eyebrow">Measured Image Parameters</div>
      <h3>Computed from pixels — no model involved</h3>
      <div class="kv"><span class="k">Source format</span><span class="v">${esc(d.input.source_format)}</span></div>
      <div class="kv"><span class="k">Original size</span><span class="v">${d.input.original_size?.join(" × ")}</span></div>
      <div class="kv"><span class="k">Windowing</span><span class="v">${esc(d.input.windowing_applied)}</span></div>
      <div class="kv"><span class="k">HU available</span><span class="v">${d.input.hu_available ? "yes (DICOM)" : "no (8-bit raster)"}</span></div>
      <div class="kv"><span class="k">Intensity mean ± sd</span><span class="v">${num(mp.intensity_mean,1)} ± ${num(mp.intensity_std,1)}</span></div>
      <div class="kv"><span class="k">Intensity p5 – p95</span><span class="v">${num(mp.intensity_p05,0)} – ${num(mp.intensity_p95,0)}</span></div>
      <div class="kv"><span class="k">Lung field detected</span><span class="v">${mp.lung_field_detected ? "yes" : "no"}</span></div>
      <div class="kv"><span class="k">Lung field area</span><span class="v">${pct(mp.lung_field_area_fraction)}</span></div>
      <div class="kv"><span class="k">Sharpness (Laplacian var)</span><span class="v">${num(mp.sharpness_laplacian_var,1)}</span></div>
      <div class="notice" style="margin-top:12px">${esc(d.measured_parameters_note)}</div>
      ${(d.input.notes || []).map(n => `<div class="notice info">${esc(n)}</div>`).join("")}
    </div>
  </div>

  <!-- 12 segmentation -->
  ${segmentationPanel(d)}

  <!-- 13 staging -->
  <div class="grid g2">
    <div class="panel">
      <div class="eyebrow">Localisation methods</div>
      <h3>Two independent views of "where"</h3>
      <ul class="clean">
        <li><b>Segmentation (above)</b> - a U-Net trained on radiologist-drawn nodule
          contours. Supervised, with measured Dice/IoU. This is the one that answers
          "where exactly".</li>
        <li><b>Grad-CAM attention</b> - shows which regions influenced the classifier.
          Coarse by construction (a 10x10 grid upsampled to 300x300) and measured to be
          no better than chance at pinpointing lesions out of domain.</li>
      </ul>
      <div class="notice info" style="margin-top:10px">They answer different questions,
        so both are shown. Where they agree, confidence is higher; where they disagree,
        the case deserves closer review.</div>
    </div>
    <div class="panel">
      <div class="eyebrow">Cancer Staging <span class="tag no">not available</span></div>
      <h3>Clinical stage is not predicted</h3>
      <p>${esc(d.staging.message)}</p>
    </div>
  </div>

  <!-- 14,15 clinical -->
  <div class="panel">
    <div class="eyebrow">Clinical Information &amp; Treatment Considerations</div>
    <h2>${esc(ci.headline)}</h2>
    <p>${esc(ci.summary)}</p>
    <div class="grid g2" style="margin-top:6px">
      <div>
        <h3>Possible clinical next steps</h3>
        <ul class="clean">${ci.possible_next_steps.map(s => `<li>${esc(s)}</li>`).join("")}</ul>
      </div>
      <div>
        <h3>Treatment approaches clinicians may consider</h3>
        ${ci.treatment_considerations.length ? `<ul class="clean">${ci.treatment_considerations
          .map(t => `<li><b>${esc(t.modality)}</b> — ${esc(t.description)}</li>`).join("")}</ul>`
          : `<p class="muted">No treatment modalities are listed for this risk category.</p>`}
      </div>
    </div>
    <div class="notice warn" style="margin-top:14px"><b>${esc(ci.treatment_disclaimer)}</b></div>
    <div class="notice">${esc(ci.confidence_caveat)}</div>
  </div>

  <!-- 16,17 limitations -->
  <div class="panel">
    <div class="eyebrow">AI Limitations</div>
    <h3>What this model cannot do</h3>
    <ul class="clean">${(d.ai_limitations || []).map(l => `<li>${esc(l)}</li>`).join("")}</ul>
    <div class="notice info" style="margin-top:14px">
      <b>Deployed model performance</b> (held-out test set, n = ${perf.test_metrics?.n ?? "—"}):
      macro-F1 ${pct(perf.test_metrics?.f1_macro)} ·
      malignant sensitivity ${pct(perf.test_metrics?.malignant_sensitivity)} ·
      ROC-AUC ${num(perf.test_metrics?.roc_auc_macro_ovr)} ·
      backbone ${esc(perf.backbone)} · primary ${esc(perf.primary_model)}.
    </div>
    <div class="disclaimer" style="margin:14px 0 0"><span class="ic">⚕️</span>
      <span><b>${esc(d.disclaimer)}</b></span></div>
  </div>`;

  $("#results").innerHTML = html;
}


/* Segmentation panel. Deliberately leads with what the model earned (Dice/IoU on
   held-out patients) and what it cannot claim, before showing the pretty picture. */
function segmentationPanel(d) {
  const s = d.segmentation || {};
  if (!s.available) {
    return `<div class="panel">
      <div class="eyebrow">Segmentation <span class="tag no">not available</span></div>
      <h3>No supervised segmentation is produced</h3>
      <p>${esc(s.message || "Segmentation model unavailable.")}</p></div>`;
  }
  const perf = s.in_domain_performance || {};
  const it = s.interpretation || {};
  const found = (s.n_lesions || 0) > 0;

  return `<div class="panel">
    <div class="eyebrow">Tumour Segmentation <span class="tag ${found ? "ok" : "no"}">
      ${found ? `${s.n_lesions} region${s.n_lesions > 1 ? "s" : ""} detected` : "no region detected"}</span></div>
    <h2>Predicted nodule location</h2>
    <div class="notice warn"><b>${esc(it.what_this_is_not || "")}</b></div>
    ${it.concordance ? `<div class="notice ${it.concordance.state === "agree" ? "info" : "warn"}">
      <b>${it.concordance.state === "agree" ? "Models agree" : "Models disagree"}
      &mdash; classifier said &ldquo;${esc(it.concordance.classifier_class)}&rdquo;,
      segmenter found ${it.concordance.segmenter_regions} region(s).</b><br>
      ${esc(it.concordance.note)}</div>` : ""}

    ${found ? `
    <div class="grid g2" style="margin-top:14px">
      <div class="imgcard"><img src="${s.images.overlay}" alt="Segmentation overlay">
        <div class="cap">Predicted nodule outline over the CT</div></div>
      <div class="imgcard"><img src="${s.images.mask}" alt="Binary mask">
        <div class="cap">Binary segmentation mask</div></div>
    </div>

    <h3 style="margin-top:18px">Detected regions</h3>
    <div class="tbl-wrap"><table class="data">
      <thead><tr><th>#</th><th>Location</th><th>Area (px)</th><th>Equiv. diameter</th>
        <th>% of lung</th><th>Model confidence</th></tr></thead><tbody>
      ${s.lesions.map((l, i) => `<tr${i === 0 ? ' class="hl"' : ""}>
        <td>${i + 1}</td><td>${esc(l.location)}</td>
        <td class="num">${l.area_px}</td>
        <td class="num">${l.equivalent_diameter_px} px</td>
        <td class="num">${l.area_fraction_of_lung != null ? (100 * l.area_fraction_of_lung).toFixed(3) + "%" : "&mdash;"}</td>
        <td class="num">${pct(l.mean_confidence)}</td></tr>`).join("")}
      </tbody></table></div>` : `
    <div class="notice warn">The segmentation model found no nodule-like region in the
      detected lung field. That is emphatically <b>not</b> a clearance: on held-out
      patients this model detected only
      <b>${perf.detection_rate != null ? pct(perf.detection_rate) : "about half"}</b>
      of annotated nodules, and it was trained on a different dataset from the image you
      uploaded.</div>`}

    <div class="grid g2" style="margin-top:16px">
      <div>
        <h3>What this model earned</h3>
        <div class="kv"><span class="k">Architecture</span><span class="v">${esc(s.model)}</span></div>
        <div class="kv"><span class="k">Trained on</span><span class="v">${esc(s.trained_on)}</span></div>
        <div class="kv"><span class="k">Detection rate</span>
          <span class="v">${perf.detection_rate != null ? pct(perf.detection_rate) : "&mdash;"}</span></div>
        <div class="kv"><span class="k">Dice <i>when it detects</i></span>
          <span class="v">${num(perf.dice_when_detected)}</span></div>
        <div class="kv"><span class="k">Centre error when detected</span>
          <span class="v">${perf.median_centroid_error_mm != null ? "~" + num(perf.median_centroid_error_mm, 2) + " mm" : "&mdash;"}</span></div>
        <div class="kv"><span class="k">Dice over all slices</span><span class="v">${num(perf.dice)}</span></div>
        <div class="kv"><span class="k">IoU over all slices</span><span class="v">${num(perf.iou)}</span></div>
        <div class="kv"><span class="k">False positives / slice</span>
          <span class="v">${num(perf.false_positive_blobs_per_slice, 2)}</span></div>
        <div class="kv"><span class="k">Operating threshold</span><span class="v">${num(s.threshold, 2)}</span></div>
        <div class="kv"><span class="k">Lung-field filter</span><span class="v">${s.lung_field_detected ? "applied" : "no lung field found"}</span></div>
      </div>
      <div>
        <h3>What it cannot claim</h3>
        <div class="notice warn">${esc(it.domain_caveat || "")}</div>
        ${it.firing_ratio_note ? `<div class="notice info">${esc(it.firing_ratio_note)}</div>` : ""}
        ${it.sensitivity_note ? `<div class="notice warn">${esc(it.sensitivity_note)}</div>` : ""}
        ${it.false_positive_note ? `<div class="notice warn">${esc(it.false_positive_note)}</div>` : ""}
      </div>
    </div>
  </div>`;
}

/* ================= MODEL COMPARISON ================= */
function renderModels() {
  const c = STATE.comparison;
  const el = $("#model-content");
  if (!c?.backbones) { el.innerHTML = `<div class="err">Model comparison unavailable. Run <code>scripts/06_model_selection.py</code>.</div>`; return; }

  let html = "";
  for (const [bb, b] of Object.entries(c.backbones)) {
    const rows = Object.entries(b.models).sort((x, y) => y[1].test.f1_macro - x[1].test.f1_macro);
    const dep = c.deployed?.backbone === bb ? c.deployed.primary_model : null;
    html += `
    <h3 style="margin-top:26px">Backbone: <code>${esc(bb)}</code>
      ${c.deployed?.backbone === bb ? '<span class="tag ok">deployed</span>' : ''}</h3>
    <div class="tbl-wrap"><table class="data">
      <thead><tr><th>Model</th><th>Type</th><th>Accuracy</th><th>Precision</th><th>Recall</th>
      <th>F1 (macro)</th><th>Test F1 95% CI</th><th>ROC-AUC</th><th>Mal. sens.</th><th>ECE</th></tr></thead><tbody>
      ${rows.map(([nm, m]) => {
        const t = m.test, type = nm.startsWith("Quantum") ? "Quantum"
          : nm.startsWith("Hybrid") ? "Hybrid" : nm.startsWith("CNN") ? "Deep learning" : "Classical ML";
        const tag = type === "Quantum" ? "q" : type === "Hybrid" ? "warn" : "no";
        return `<tr class="${nm === dep ? "hl" : ""}">
          <td><b>${esc(nm)}</b>${nm === dep ? ' <span class="tag ok">primary</span>' : ''}</td>
          <td><span class="tag ${tag}">${type}</span></td>
          <td class="num">${pct(t.accuracy)}</td><td class="num">${pct(t.precision_macro)}</td>
          <td class="num">${pct(t.recall_macro)}</td><td class="num"><b>${pct(t.f1_macro)}</b></td>
          <td class="num">${num(m.test_ci?.lo95)}–${num(m.test_ci?.hi95)}</td>
          <td class="num">${num(t.roc_auc_macro_ovr)}</td>
          <td class="num">${pct(t.malignant_sensitivity)}</td>
          <td class="num">${num(m.test_calibration?.ece)}</td></tr>`;
      }).join("")}
      </tbody></table></div>
      ${barChart(rows)}
      ${curvePanels(b, rows)}
      <div class="notice info" style="margin-top:14px">
        <b>Selection:</b> argmax on validation was <code>${esc(b.argmax_validation)}</code>, but
        <b>${b.statistically_tied.length} of ${rows.length}</b> candidates were statistically
        indistinguishable from it (paired bootstrap, p ≥ 0.05). The clinical tie-break then chose
        <code>${esc(b.primary_model)}</code>. ${esc(b.selection_policy)}</div>
      ${qaBlock(b.quantum_advantage_test)}`;
  }
  el.innerHTML = html + `
    <div class="notice warn" style="margin-top:20px"><b>Why these numbers are lower than published
    IQ-OTH/NCCD results.</b> Papers reporting 99–100% typically use a random split, which leaks
    near-duplicate slices of the same patient across train and test. Under our group-aware split
    that leakage path is removed, so these figures are a more realistic estimate of generalisation.</div>
    <div class="notice"><b>Note on cross-validation scores.</b> The 5-fold CV scores recorded in
    <code>reports/experiments/</code> are computed on embeddings produced by a CNN that was itself
    trained on those same training images, so they are optimistically biased and are not used for
    model selection.</div>`;
}

function barChart(rows) {
  const W = 1100, rowH = 26, H = rows.length * rowH + 46;
  const max = 1.0;
  const bars = rows.map(([nm, m], i) => {
    const v = m.test.f1_macro, lo = m.test_ci?.lo95 ?? v, hi = m.test_ci?.hi95 ?? v;
    const x0 = 260, w = (W - x0 - 90);
    const col = nm.startsWith("Quantum") ? "#c084fc" : nm.startsWith("Hybrid") ? "#fbbf24"
      : nm.startsWith("CNN") ? "#38bdf8" : "#5b7fd0";
    const y = 30 + i * rowH;
    return `
      <text x="${x0 - 10}" y="${y + 12}" text-anchor="end" fill="#9fb0cb" font-size="11"
        font-family="Inter,sans-serif">${esc(nm)}</text>
      <rect x="${x0}" y="${y + 3}" width="${(w * v / max).toFixed(1)}" height="13" rx="3" fill="${col}" opacity=".85"/>
      <line x1="${x0 + w * lo / max}" y1="${y + 9.5}" x2="${x0 + w * hi / max}" y2="${y + 9.5}"
        stroke="#e8eef8" stroke-width="1.2" opacity=".55"/>
      <line x1="${x0 + w * lo / max}" y1="${y + 5}" x2="${x0 + w * lo / max}" y2="${y + 14}" stroke="#e8eef8" stroke-width="1.2" opacity=".55"/>
      <line x1="${x0 + w * hi / max}" y1="${y + 5}" x2="${x0 + w * hi / max}" y2="${y + 14}" stroke="#e8eef8" stroke-width="1.2" opacity=".55"/>
      <text x="${x0 + w * v / max + 8}" y="${y + 13}" fill="#e8eef8" font-size="11"
        font-family="ui-monospace,monospace">${pct(v)}</text>`;
  }).join("");
  const grid = [0, .25, .5, .75, 1].map(g => {
    const x = 260 + (W - 350) * g;
    return `<line x1="${x}" y1="24" x2="${x}" y2="${H - 16}" stroke="#1e2b45" stroke-width="1"/>
      <text x="${x}" y="${H - 4}" text-anchor="middle" fill="#6b7f9e" font-size="10">${(100*g)}%</text>`;
  }).join("");
  return `<div style="overflow-x:auto;margin-top:16px"><svg viewBox="0 0 ${W} ${H}"
    style="width:100%;min-width:820px;height:auto">${grid}${bars}
    <text x="260" y="14" fill="#6b7f9e" font-size="11" font-family="Inter,sans-serif">
      Test macro-F1 with 95% bootstrap CI</text></svg>
    <div class="legend"><span><i style="background:#38bdf8"></i>Deep learning</span>
    <span><i style="background:#5b7fd0"></i>Classical ML</span>
    <span><i style="background:#c084fc"></i>Quantum</span>
    <span><i style="background:#fbbf24"></i>Hybrid</span></div></div>`;
}

/* ROC + PR curves and confusion matrix for the top model of a backbone */
function curvePanels(b, rows) {
  const primary = b.primary_model;
  const cur = b.curves_test?.[primary];
  const m = b.models[primary];
  if (!cur && !m) return "";
  return `<div class="grid g3" style="margin-top:18px">
    <div class="panel tight"><h3 style="font-size:.95rem">ROC — ${esc(primary)}</h3>
      ${cur ? curveSvg(cur.roc, "fpr", "tpr", "auc", "False positive rate", "True positive rate", true) : "—"}</div>
    <div class="panel tight"><h3 style="font-size:.95rem">Precision–Recall — ${esc(primary)}</h3>
      ${cur ? curveSvg(cur.pr, "recall", "precision", "ap", "Recall", "Precision", false) : "—"}</div>
    <div class="panel tight"><h3 style="font-size:.95rem">Confusion matrix — ${esc(primary)}</h3>
      ${confusionSvg(m?.test)}</div></div>`;
}

const CURVE_COLORS = { benign: "#fbbf24", malignant: "#f87171", normal: "#34d399" };

function curveSvg(obj, xk, yk, sk, xlab, ylab, diag) {
  if (!obj) return "—";
  const S = 250, pad = 34, W = S, H = S;
  const X = v => pad + v * (W - pad - 10), Y = v => H - pad - v * (H - pad - 16);
  let paths = "", leg = "";
  for (const [cls, d] of Object.entries(obj)) {
    const xs = d[xk], ys = d[yk];
    if (!xs?.length) continue;
    const col = CURVE_COLORS[cls] || "#38bdf8";
    // PR curves come back descending in recall; sort so the polyline is monotone.
    const pts = xs.map((x, i) => [x, ys[i]]).sort((a, c) => a[0] - c[0]);
    paths += `<polyline fill="none" stroke="${col}" stroke-width="1.8" points="${
      pts.map(([x, y]) => `${X(x).toFixed(1)},${Y(y).toFixed(1)}`).join(" ")}"/>`;
    leg += `<span><i style="background:${col}"></i>${esc(cls)} ${num(d[sk], 3)}</span>`;
  }
  const grid = [0, .5, 1].map(g => `
    <line x1="${X(g)}" y1="${Y(0)}" x2="${X(g)}" y2="${Y(1)}" stroke="#1e2b45"/>
    <line x1="${X(0)}" y1="${Y(g)}" x2="${X(1)}" y2="${Y(g)}" stroke="#1e2b45"/>
    <text x="${X(g)}" y="${H - 16}" text-anchor="middle" fill="#6b7f9e" font-size="9">${g}</text>
    <text x="${pad - 6}" y="${Y(g) + 3}" text-anchor="end" fill="#6b7f9e" font-size="9">${g}</text>`).join("");
  return `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto">
    ${grid}${diag ? `<line x1="${X(0)}" y1="${Y(0)}" x2="${X(1)}" y2="${Y(1)}"
      stroke="#2a3a5c" stroke-dasharray="3 3"/>` : ""}${paths}
    <text x="${W / 2}" y="${H - 3}" text-anchor="middle" fill="#6b7f9e" font-size="9">${xlab}</text>
    <text x="9" y="${H / 2}" fill="#6b7f9e" font-size="9" transform="rotate(-90 9 ${H / 2})"
      text-anchor="middle">${ylab}</text></svg>
    <div class="legend" style="margin-top:6px">${leg}</div>`;
}

function confusionSvg(t) {
  const cm = t?.confusion_matrix, labs = t?.confusion_matrix_labels;
  if (!cm) return "—";
  const n = cm.length, cell = 56, pad = 74, W = pad + n * cell + 10, H = 40 + n * cell + 26;
  const max = Math.max(...cm.flat(), 1);
  let s = "";
  for (let i = 0; i < n; i++) {
    s += `<text x="${pad - 7}" y="${44 + i * cell + cell / 2}" text-anchor="end" fill="#9fb0cb"
      font-size="10">${esc(labs[i])}</text>`;
    s += `<text x="${pad + i * cell + cell / 2}" y="34" text-anchor="middle" fill="#9fb0cb"
      font-size="10">${esc(labs[i])}</text>`;
    for (let j = 0; j < n; j++) {
      const v = cm[i][j], a = v / max;
      const col = i === j ? `rgba(52,211,153,${.14 + .66 * a})` : `rgba(248,113,113,${.12 + .6 * a})`;
      s += `<rect x="${pad + j * cell}" y="${44 + i * cell}" width="${cell - 3}" height="${cell - 3}"
        rx="4" fill="${col}" stroke="#1e2b45"/>
        <text x="${pad + j * cell + (cell - 3) / 2}" y="${44 + i * cell + (cell - 3) / 2 + 4}"
        text-anchor="middle" fill="#e8eef8" font-size="13" font-weight="600">${v}</text>`;
    }
  }
  return `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto">
    <text x="6" y="16" fill="#6b7f9e" font-size="9">rows = true class · columns = predicted</text>${s}</svg>`;
}

function qaBlock(qa) {
  if (!qa) return "";
  return Object.entries(qa).map(([k, v]) => {
    const sig = v["significant_at_0.05"];
    return `<div class="notice ${sig ? "q" : "warn"}" style="margin-top:10px">
      <b>Quantum advantage test — ${esc(k.replace(/_/g, " "))}:</b>
      ${esc(v.quantum_model || v.hybrid_model)} vs ${esc(v.classical_model)} ·
      Δmacro-F1 = <b>${v.mean_diff >= 0 ? "+" : ""}${num(v.mean_diff)}</b>
      (95% CI ${num(v.ci95_lo)} … ${num(v.ci95_hi)}), p = <b>${num(v.p_value)}</b> →
      <b>${sig ? "statistically significant" : "NOT significant — no quantum advantage demonstrated"}</b>.</div>`;
  }).join("");
}

/* ================= QUANTUM PAGE ================= */
function renderQuantum() {
  const q = STATE.meta?.quantum || {}, ci = q.circuit || {};
  const nq = ci.n_qubits || STATE.meta?.engine?.n_qubits || 4;
  const nl = ci.n_variational_layers || q.variational_layers || 3;

  $("#quantum-content").innerHTML = `
    <p>The quantum branch is a <b>variational quantum circuit (VQC)</b> simulated with PennyLane.
    It does not replace the CNN — it consumes the CNN's learned representation and offers an
    alternative, entanglement-based decision boundary over a small number of features.</p>

    <div class="grid g4" style="margin:18px 0">
      <div class="panel stat"><div class="v q">${nq}</div><div class="k">Qubits (deployed)</div></div>
      <div class="panel stat"><div class="v">${nl}</div><div class="k">Variational layers</div></div>
      <div class="panel stat"><div class="v">${ci.circuit_depth ?? "—"}</div><div class="k">Circuit depth</div></div>
      <div class="panel stat"><div class="v">${ci.gate_count ?? "—"}</div><div class="k">Total gates</div></div>
    </div>

    ${quantumDiagram(nq, nl)}

    <div class="grid g2" style="margin-top:20px">
      <div class="panel tight">
        <h3>Circuit specification</h3>
        <div class="kv"><span class="k">Encoding</span><span class="v">${esc(ci.encoding || "Angle (RY)")}</span></div>
        <div class="kv"><span class="k">Entanglement</span><span class="v">${esc(ci.entanglement || "CNOT ring")}</span></div>
        <div class="kv"><span class="k">Measurement</span><span class="v">${esc(ci.measurement || "Pauli-Z ⟨Z⟩")}</span></div>
        <div class="kv"><span class="k">Backend</span><span class="v">${esc(ci.backend || "default.qubit")}</span></div>
        <div class="kv"><span class="k">Differentiation</span><span class="v">${esc(ci.differentiation || "backprop")}</span></div>
        <div class="kv"><span class="k">Trainable circuit params</span><span class="v">${ci.trainable_circuit_params ?? "—"}</span></div>
        <div class="kv"><span class="k">Gate breakdown</span><span class="v">${esc(JSON.stringify(ci.gate_types || {}))}</span></div>
        <div class="kv"><span class="k">Qubit sweep run</span><span class="v">${(q.qubit_sweep || []).join(" · ")}</span></div>
      </div>
      <div class="panel tight">
        <h3>Design choices, and why</h3>
        <ul class="clean">
          <li><b>Simulator, not hardware.</b> Statevector simulation on <code>default.qubit</code>.
          No real QPU was used, so no hardware-noise claims are made.</li>
          <li><b>Few qubits (4/6/8).</b> Swept experimentally; 4 qubits performed best on
          validation. More qubits did not help — with 735 training samples they simply overfit.</li>
          <li><b>Shallow depth (${nl} layers).</b> Deeper ansätze hit barren plateaus and train
          poorly at this data scale.</li>
          <li><b>PCA to 32, then a learned projection to ${nq}.</b> The projection is trainable, so
          the network chooses which directions deserve the limited qubit budget.</li>
          <li><b>Angle encoding.</b> One feature per qubit as an RY rotation — cheap to simulate and
          keeps the encoding interpretable.</li>
        </ul>
      </div>
    </div>

    <div class="notice warn" style="margin-top:18px"><b>Honest conclusion.</b> Across both backbones,
    the quantum and hybrid branches were <b>not</b> significantly better than the best classical
    model on the held-out test split (paired bootstrap, p ≈ 0.69–0.71). The quantum branch is a
    genuine, working, trainable component — but on this dataset it delivers <i>comparable</i>, not
    superior, performance. We report that rather than claiming an advantage we did not measure.</div>`;
}

function quantumDiagram(nq, nl) {
  const W = 1100, top = 44, gap = 40, H = top + nq * gap + 54;
  const x0 = 150, xEnd = W - 90;
  let s = "";
  for (let w = 0; w < nq; w++) {
    const y = top + w * gap;
    s += `<line x1="${x0}" y1="${y}" x2="${xEnd}" y2="${y}" stroke="#2a3a5c" stroke-width="1.4"/>
      <text x="${x0 - 12}" y="${y + 4}" text-anchor="end" fill="#9fb0cb" font-size="12"
        font-family="ui-monospace,monospace">|0⟩ q${w}</text>`;
    s += gate(x0 + 30, y, "RY(xᵢ)", "#38bdf8");
    for (let l = 0; l < nl; l++) {
      const bx = x0 + 130 + l * 210;
      s += gate(bx, y, "RY(θ)", "#c084fc") + gate(bx + 62, y, "RZ(φ)", "#c084fc");
      const y2 = top + ((w + 1) % nq) * gap;
      const cx = bx + 140;
      s += `<circle cx="${cx}" cy="${y}" r="3.6" fill="#8b5cf6"/>
        <line x1="${cx}" y1="${y}" x2="${cx}" y2="${y2}" stroke="#8b5cf6" stroke-width="1.3" opacity=".65"/>
        <circle cx="${cx}" cy="${y2}" r="7" fill="none" stroke="#8b5cf6" stroke-width="1.3"/>
        <line x1="${cx - 7}" y1="${y2}" x2="${cx + 7}" y2="${y2}" stroke="#8b5cf6" stroke-width="1.3"/>`;
    }
    s += `<rect x="${xEnd - 52}" y="${y - 13}" width="46" height="26" rx="5" fill="#0d2a20" stroke="#34d399"/>
      <text x="${xEnd - 29}" y="${y + 4}" text-anchor="middle" fill="#86efc7" font-size="11"
        font-family="ui-monospace,monospace">⟨Z⟩</text>`;
  }
  const labels = `
    <text x="${x0 + 30}" y="26" text-anchor="middle" fill="#38bdf8" font-size="10.5"
      font-family="Inter,sans-serif" font-weight="600">ANGLE ENCODING</text>
    <text x="${x0 + 300}" y="26" text-anchor="middle" fill="#c084fc" font-size="10.5"
      font-family="Inter,sans-serif" font-weight="600">${nl} × VARIATIONAL LAYER (RY·RZ + CNOT RING)</text>
    <text x="${xEnd - 29}" y="26" text-anchor="middle" fill="#34d399" font-size="10.5"
      font-family="Inter,sans-serif" font-weight="600">MEASURE</text>`;
  return `<div style="overflow-x:auto"><svg viewBox="0 0 ${W} ${H}"
    style="width:100%;min-width:900px;height:auto">${labels}${s}
    <text x="${x0}" y="${H - 12}" fill="#6b7f9e" font-size="10.5" font-family="Inter,sans-serif">
      CNN embedding (256-d) → StandardScaler → PCA (32) → learned linear projection → tanh·π → the ${nq} rotation angles above
    </text></svg></div>`;
}
function gate(x, y, label, col) {
  return `<rect x="${x - 28}" y="${y - 13}" width="56" height="26" rx="5" fill="#131c30" stroke="${col}" stroke-width="1.2"/>
    <text x="${x}" y="${y + 4}" text-anchor="middle" fill="${col}" font-size="10.5"
      font-family="ui-monospace,monospace">${label}</text>`;
}

/* ================= ABOUT ================= */
function renderAbout() {
  const ed = STATE.education?.cancer_education, m = STATE.meta;
  const loc = STATE.localization?.results || {};
  const lk = Object.keys(loc)[0] ? loc[Object.keys(loc)[0]] : null;

  $("#about-content").innerHTML = `
  <div class="panel">
    <div class="eyebrow">Problem Statement 26139</div>
    <h1>Hybrid Quantum Machine Learning Platform for Early Disease Detection</h1>
    <p>Team <b>CodeNova</b> · Smart India Hackathon 2026. Focus: early lung cancer detection from
    chest CT imaging using a hybrid quantum-classical architecture with explainable outputs.</p>
    <div class="grid g3" style="margin-top:14px">
      <div class="panel tight"><h3>Backend</h3><p class="muted">Python · FastAPI · PyTorch ${m?.hardware?.torch || ""}
        ${m?.hardware?.cuda ? "· CUDA on " + esc(m.hardware.gpu_name) : "· CPU"}</p></div>
      <div class="panel tight"><h3>Quantum</h3><p class="muted">PennyLane variational circuits,
        statevector simulation, ${m?.quantum?.qubit_sweep?.join("/") || "4/6/8"} qubit sweep</p></div>
      <div class="panel tight"><h3>Vision</h3><p class="muted">timm · EfficientNet-B3, DenseNet-121 ·
        Grad-CAM++ · OpenCV · pydicom</p></div>
    </div>
  </div>

  <div class="panel">
    <div class="eyebrow">Dataset</div>
    <h2>What the data actually supports</h2>
    <div class="grid g2">
      <div>
        <h3>IQ-OTH/NCCD (training + evaluation)</h3>
        <div class="kv"><span class="k">Images on disk</span><span class="v">1,097 JPEG</span></div>
        <div class="kv"><span class="k">Classes</span><span class="v">malignant 561 · normal 416 · benign 120</span></div>
        <div class="kv"><span class="k">After cleaning</span><span class="v">1,048 (43 exact dups, 6 label-ambiguous excluded)</span></div>
        <div class="kv"><span class="k">Split</span><span class="v">735 / 157 / 156 · group-aware · seed 42</span></div>
        <div class="kv"><span class="k">Patient IDs</span><span class="v">absent — pseudo-patient grouping used</span></div>
      </div>
      <div>
        <h3>Capability matrix</h3>
        <div class="kv"><span class="k">3-class classification</span><span class="v"><span class="tag ok">supported</span></span></div>
        <div class="kv"><span class="k">Attention localisation</span><span class="v"><span class="tag warn">weak — validated</span></span></div>
        <div class="kv"><span class="k">Tumour segmentation</span><span class="v"><span class="tag no">no masks in data</span></span></div>
        <div class="kv"><span class="k">Cancer staging</span><span class="v"><span class="tag no">no stage labels</span></span></div>
        <div class="kv"><span class="k">Histological subtype</span><span class="v"><span class="tag no">no subtype labels</span></span></div>
      </div>
    </div>
    <div class="notice info" style="margin-top:14px"><b>Second dataset discovered during the audit.</b>
    The 197 PNGs in <code>Test cases/</code> follow the DeepLesion naming convention and contain
    burned-in green lesion bounding boxes. They have no class labels, so they are never used for
    classification. Instead we inpaint the boxes out, run Grad-CAM, and score the attention peak
    against the recovered box — turning them into a genuine localisation validation set.</div>
  </div>

  <div class="panel">
    <div class="eyebrow">Methodology</div>
    <h2>Pipeline</h2>
    <ol class="clean">
      <li><b>Forensic audit</b> — formats, corruption, duplicates, resolution, class balance, leakage risk.</li>
      <li><b>Cleaning</b> — flag-and-quarantine (never delete); exact duplicates and label-ambiguous images excluded.</li>
      <li><b>Preprocessing</b> — bilateral denoise, CLAHE, aspect-preserving letterbox to 300×300. DICOM path applies rescale slope/intercept and a true lung window.</li>
      <li><b>Augmentation</b> — train only: ±12° rotation, ±6% translation, 0.92–1.08 scale, mild brightness/contrast/noise/gamma. No vertical flips or elastic warps.</li>
      <li><b>Splitting</b> — group-aware on pseudo-patient clusters; verified zero group overlap.</li>
      <li><b>Classical branch</b> — two-stage fine-tuning, focal loss, balanced sampling, bf16 mixed precision.</li>
      <li><b>Quantum branch</b> — PCA(32) → learned projection → angle encoding → VQC → ⟨Z⟩.</li>
      <li><b>Fusion</b> — four strategies; chosen on validation.</li>
      <li><b>Selection</b> — paired bootstrap significance testing plus a clinical tie-break.</li>
      <li><b>Explainability</b> — Grad-CAM++, quantitatively validated against real lesion boxes.</li>
    </ol>
  </div>

  ${STATE.lidc ? `<div class="panel">
    <div class="eyebrow">External Data Experiment</div>
    <h2>We went and got more data. It did not help.</h2>
    <p>${esc(STATE.lidc.question)} <b>${esc(STATE.lidc.answer)}</b></p>
    <div class="grid g4" style="margin:14px 0">
      <div class="panel stat"><div class="v">${STATE.lidc.acquired.series_downloaded}</div>
        <div class="k">LIDC series downloaded</div></div>
      <div class="panel stat"><div class="v">${STATE.lidc.acquired.slices_extracted}</div>
        <div class="k">Labelled slices added</div></div>
      <div class="panel stat"><div class="v ok">${STATE.lidc.training_set_change.benign.multiplier}x</div>
        <div class="k">More benign training data</div></div>
      <div class="panel stat"><div class="v" style="color:var(--danger)">${num(STATE.lidc.results_on_unchanged_iqoth_test.Hybrid_meta.delta)}</div>
        <div class="k">Macro-F1 change</div></div>
    </div>
    <div class="grid g2">
      <div>
        <h3>How it was kept honest</h3>
        <ul class="clean">
          <li><b>Annotations parsed before downloading imaging</b>, so the 50 series were chosen
            by nodule yield rather than at random.</li>
          <li><b>Balanced selection.</b> Ranking purely by benign yield would have made every LIDC
            image benign, letting the model learn "looks like LIDC = benign". We took 32
            benign-rich plus 18 malignant-rich series instead.</li>
          <li><b>LIDC entered training only.</b> Validation and test stayed byte-identical and
            100% IQ-OTH, so a dataset shortcut could not inflate the numbers.</li>
          <li><b>Indeterminate nodules excluded.</b>
            ${STATE.lidc.acquired.nodules_indeterminate_excluded} nodules with a mean malignancy
            rating of exactly 3 were dropped, never silently assigned.</li>
          <li><b>No normals taken from LIDC</b> - every LIDC patient was scanned for suspected
            pathology, so an unannotated slice is not evidence of a healthy lung.</li>
        </ul>
      </div>
      <div>
        <h3>What happened</h3>
        <div class="kv"><span class="k">Source-identity probe ROC-AUC</span>
          <span class="v">${num(STATE.lidc.dataset_identity_probe.source_prediction_roc_auc, 4)}</span></div>
        <div class="kv"><span class="k">CNN macro-F1</span><span class="v">0.917 &rarr; 0.804</span></div>
        <div class="kv"><span class="k">Hybrid macro-F1</span><span class="v">0.925 &rarr; 0.831 (p=0.009)</span></div>
        <div class="kv"><span class="k">Normal recall</span><span class="v">0.950 &rarr; 0.817</span></div>
        <div class="notice warn" style="margin-top:12px">${esc(STATE.lidc.mitigation_attempted)}</div>
        <div class="notice info">${esc(STATE.lidc.one_genuine_benefit)}</div>
      </div>
    </div>
    <div class="notice q" style="margin-top:14px"><b>Lesson.</b> ${esc(STATE.lidc.lesson)}</div>
    <div class="notice"><b>Decision.</b> ${esc(STATE.lidc.decision)}</div>
  </div>` : ""}

  <div class="panel">
    <div class="eyebrow">Limitations</div>
    <h2>Stated plainly</h2>
    <ul class="clean">
      ${(ed?.ai_limitations || []).map(l => `<li>${esc(l)}</li>`).join("")}
      ${lk ? `<li><b>Measured localisation limitation:</b> against ${lk.n_images} real DeepLesion
      lesion boxes, Grad-CAM++ pointing-game hit rate was ${num(lk.pointing_game_hit_rate)}
      (chance ≈ ${num(lk.expected_chance_hit_rate, 4)}), and only
      ${num(lk.mean_attention_energy_in_box, 4)} of attention energy fell inside the lesion box.
      On out-of-domain data the attention maps do <b>not</b> reliably localise lesions.</li>` : ""}
      <li><b>Pseudo-patient grouping is a surrogate</b> for true patient IDs; residual leakage cannot be fully excluded.</li>
    </ul>
  </div>

  ${ed ? `<div class="panel">
    <div class="eyebrow">Cancer Information</div>
    <h2>Educational reference</h2>
    <p class="muted">Separate from any model output above.</p>
    <h3>What lung cancer is</h3><p>${esc(ed.what_is_it)}</p>
    <h3>Major types</h3>
    <ul class="clean">${ed.major_types.map(([t, d]) => `<li><b>${esc(t)}</b> — ${esc(d)}</li>`).join("")}</ul>
    <div class="grid g2" style="margin-top:12px">
      <div><h3>Risk factors</h3><ul class="clean">${ed.risk_factors.map(r => `<li>${esc(r)}</li>`).join("")}</ul></div>
      <div>
        <h3>Why early detection matters</h3><p>${esc(ed.why_early_detection)}</p>
        <h3>How CT is used</h3><p>${esc(ed.how_ct_is_used)}</p>
      </div>
    </div>
    <h3>What AI analysis means</h3><p>${esc(ed.what_ai_means)}</p>
    <h3>Why pathology confirmation matters</h3><p>${esc(ed.why_pathology)}</p>
  </div>` : ""}

  <div class="panel">
    <div class="eyebrow">Research Disclaimer</div>
    <div class="disclaimer" style="margin:0"><span class="ic">⚕️</span><span>
      <b>This platform is a research prototype built for a hackathon.</b> It has not been
      clinically validated, has no regulatory clearance, and must not be used to make real
      patient-care decisions. AI-assisted research and clinical decision-support tool —
      not a replacement for professional medical diagnosis.</span></div>
  </div>`;
}
