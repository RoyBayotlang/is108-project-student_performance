/* ================================================================
   app.js  –  IS 108 Student Performance Prediction
   Handles all API calls, chart rendering, and UI logic.
================================================================ */

// ── Chart instances (kept so we can destroy on re-render) ────────
const charts = {};

// ── Utility: destroy a chart if it exists ───────────────────────
function destroyChart(id) {
  if (charts[id]) { charts[id].destroy(); delete charts[id]; }
}

// ── Fetch helper ─────────────────────────────────────────────────
async function apiFetch(url, opts = {}) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || res.statusText);
  }
  return res.json();
}

// ─────────────────────────────────────────────────────────────────
//  DASHBOARD
// ─────────────────────────────────────────────────────────────────
async function loadDashboard() {
  try {
    const data = await apiFetch('/api/dataset/info');

    document.getElementById('d-total').textContent   = data.rows;
    document.getElementById('d-pass').textContent    = data.pass_count;
    document.getElementById('d-fail').textContent    = data.fail_count;
    document.getElementById('d-features').textContent = (data.columns - 1); // minus target G3

    // ── Doughnut: Pass / Fail ─────────────────────────────────────
    destroyChart('chartDist');
    const ctx1 = document.getElementById('chartDist').getContext('2d');
    charts['chartDist'] = new Chart(ctx1, {
      type: 'doughnut',
      data: {
        labels: ['Pass', 'Fail'],
        datasets: [{ data: [data.pass_count, data.fail_count],
          backgroundColor: ['rgba(16,185,129,.7)', 'rgba(239,68,68,.7)'],
          borderColor:     ['#10b981', '#ef4444'],
          borderWidth: 2 }]
      },
      options: {
        responsive: true, maintainAspectRatio: true,
        plugins: {
          legend: { position: 'bottom', labels: { color: '#94a3b8', font: { size: 12 } } },
          tooltip: { callbacks: { label: ctx => ` ${ctx.label}: ${ctx.raw} students` } }
        },
        cutout: '65%'
      }
    });

    // ── Bar: G3 histogram from raw data ──────────────────────────
    await loadG3Chart();

  } catch (e) {
    console.error('Dashboard error:', e);
  }
}

async function loadG3Chart() {
  try {
    const data = await apiFetch('/api/dataset');
    const g3Idx = data.columns.indexOf('G3');
    const counts = Array(21).fill(0);
    data.rows.forEach(r => { const g = parseInt(r[g3Idx]); if (g >= 0 && g <= 20) counts[g]++; });
    const colors = counts.map((_, i) => i >= 10 ? 'rgba(16,185,129,.75)' : 'rgba(239,68,68,.75)');
    const borders = counts.map((_, i) => i >= 10 ? '#10b981' : '#ef4444');

    destroyChart('chartG3');
    const ctx2 = document.getElementById('chartG3').getContext('2d');
    charts['chartG3'] = new Chart(ctx2, {
      type: 'bar',
      data: {
        labels: counts.map((_, i) => i),
        datasets: [{ label: 'Students', data: counts, backgroundColor: colors, borderColor: borders, borderWidth: 1, borderRadius: 4 }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false }, tooltip: { callbacks: { title: ctx => `G3 = ${ctx[0].label}`, label: ctx => ` ${ctx.raw} students` } } },
        scales: {
          x: { grid: { color: 'rgba(255,255,255,.06)' }, ticks: { color: '#94a3b8' }, title: { display: true, text: 'Final Grade (G3)', color: '#94a3b8' } },
          y: { grid: { color: 'rgba(255,255,255,.06)' }, ticks: { color: '#94a3b8' }, title: { display: true, text: 'Count', color: '#94a3b8' } }
        }
      }
    });
  } catch (e) { console.error('G3 chart error:', e); }
}

// ─────────────────────────────────────────────────────────────────
//  DATASET
// ─────────────────────────────────────────────────────────────────
async function loadDataset() {
  try {
    const [ds, info] = await Promise.all([
      apiFetch('/api/dataset'),
      apiFetch('/api/dataset/info')
    ]);

    document.getElementById('ds-rows').textContent   = info.rows;
    document.getElementById('ds-cols').textContent   = info.columns;
    document.getElementById('ds-missing').textContent = Object.values(info.missing).reduce((a, b) => a + b, 0);
    document.getElementById('ds-g3mean').textContent = info.g3_mean;
    document.getElementById('ds-badge').textContent  = `Showing ${ds.rows.length} of ${info.rows} rows`;

    // Build table headers
    const thead = document.getElementById('ds-thead');
    thead.innerHTML = '<tr>' + ds.columns.map(c => `<th>${c}</th>`).join('') + '</tr>';

    // Build rows (colour G3 cell)
    const g3Idx = ds.columns.indexOf('G3');
    const tbody = document.getElementById('ds-tbody');
    tbody.innerHTML = ds.rows.map(row =>
      '<tr>' + row.map((cell, i) => {
        let cls = '';
        if (i === g3Idx) cls = parseInt(cell) >= 10 ? 'text-success fw-semibold' : 'text-danger fw-semibold';
        return `<td class="${cls}">${cell}</td>`;
      }).join('') + '</tr>'
    ).join('');

  } catch (e) {
    console.error('Dataset load error:', e);
  }
}

// ─────────────────────────────────────────────────────────────────
//  PREPROCESSING
// ─────────────────────────────────────────────────────────────────
async function loadPreprocess() {
  try {
    const data = await apiFetch('/api/preprocess');

    document.getElementById('pr-total').textContent    = data.total_rows;
    document.getElementById('pr-train').textContent    = data.train_rows;
    document.getElementById('pr-test').textContent     = data.test_rows;
    document.getElementById('pr-features').textContent = data.feature_count;
    document.getElementById('pr-missing').textContent  = data.missing_total;
    document.getElementById('cat-count').textContent   = `Encoding ${data.categorical_cols.length} categorical columns → integers`;
    document.getElementById('split-info').textContent  = data.split;

    // Doughnut: train/test split
    destroyChart('chartSplit');
    const ctx = document.getElementById('chartSplit').getContext('2d');
    charts['chartSplit'] = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: ['Train', 'Test'],
        datasets: [{ data: [data.train_rows, data.test_rows],
          backgroundColor: ['rgba(59,130,246,.7)', 'rgba(245,158,11,.7)'],
          borderColor: ['#3b82f6', '#f59e0b'], borderWidth: 2 }]
      },
      options: {
        responsive: true, maintainAspectRatio: true,
        plugins: { legend: { position: 'bottom', labels: { color: '#94a3b8', font: { size: 12 } } } },
        cutout: '60%'
      }
    });

  } catch (e) { console.error('Preprocess error:', e); }
}

// ─────────────────────────────────────────────────────────────────
//  TRAINING
// ─────────────────────────────────────────────────────────────────
async function trainModels() {
  const btn     = document.getElementById('btn-train');
  const spinner = document.getElementById('train-spinner');
  const results = document.getElementById('train-results');

  btn.disabled = true;
  spinner.classList.remove('d-none');
  results.classList.add('d-none');

  try {
    const data = await apiFetch('/api/train', { method: 'POST' });
    renderTrainCards(data.metrics);
    results.classList.remove('d-none');
  } catch (e) {
    alert('Training error: ' + e.message);
  } finally {
    btn.disabled = false;
    spinner.classList.add('d-none');
  }
}

function renderTrainCards(metrics) {
  const colors = { KNN: 'primary', SVM: 'warning', ANN: 'success' };
  const emojis = { KNN: '🔵', SVM: '🟡', ANN: '🟢' };
  const container = document.getElementById('train-metric-cards');

  container.innerHTML = Object.entries(metrics).map(([name, m]) => `
    <div class="col-md-4">
      <div class="card p-3">
        <h5 class="text-${colors[name]}">${emojis[name]} ${name}</h5>
        <div class="d-flex justify-content-between py-1 border-bottom border-secondary">
          <span class="text-muted" style="font-size:.82rem;">Accuracy</span>
          <span class="fw-bold text-${colors[name]}">${m.accuracy}%</span>
        </div>
        <div class="d-flex justify-content-between py-1 border-bottom border-secondary">
          <span class="text-muted" style="font-size:.82rem;">F1 Score</span>
          <span style="font-size:.82rem;">${m.f1}%</span>
        </div>
        <div class="d-flex justify-content-between py-1">
          <span class="text-muted" style="font-size:.82rem;">Train Time</span>
          <span style="font-size:.82rem;">${m.train_time}s</span>
        </div>
      </div>
    </div>`).join('');
}

// ─────────────────────────────────────────────────────────────────
//  EVALUATION
// ─────────────────────────────────────────────────────────────────
async function loadEvaluation() {
  try {
    const data = await apiFetch('/api/evaluate');
    renderEvalCards(data);
    renderCompareChart(data);
  } catch (e) {
    document.getElementById('eval-cards').innerHTML =
      `<div class="col-12"><div class="alert alert-warning"><i class="bi bi-exclamation-triangle me-2"></i>${e.message}</div></div>`;
  }
}

function renderCompareChart(metrics) {
  const names  = Object.keys(metrics);
  const accs   = names.map(n => metrics[n].accuracy);
  const precs  = names.map(n => metrics[n].precision);
  const recalls = names.map(n => metrics[n].recall);
  const f1s    = names.map(n => metrics[n].f1);

  destroyChart('chartCompare');
  const ctx = document.getElementById('chartCompare').getContext('2d');
  charts['chartCompare'] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: names,
      datasets: [
        { label: 'Accuracy',  data: accs,    backgroundColor: 'rgba(59,130,246,.7)',  borderColor: '#3b82f6', borderWidth:1, borderRadius:4 },
        { label: 'Precision', data: precs,   backgroundColor: 'rgba(16,185,129,.7)',  borderColor: '#10b981', borderWidth:1, borderRadius:4 },
        { label: 'Recall',    data: recalls, backgroundColor: 'rgba(245,158,11,.7)',  borderColor: '#f59e0b', borderWidth:1, borderRadius:4 },
        { label: 'F1 Score',  data: f1s,     backgroundColor: 'rgba(139,92,246,.7)',  borderColor: '#8b5cf6', borderWidth:1, borderRadius:4 },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: '#94a3b8' } } },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,.06)' }, ticks: { color: '#94a3b8' } },
        y: { min: 0, max: 100, grid: { color: 'rgba(255,255,255,.06)' }, ticks: { color: '#94a3b8', callback: v => v + '%' } }
      }
    }
  });
}

function renderEvalCards(metrics) {
  const best = Object.entries(metrics).reduce((a, b) => b[1].accuracy > a[1].accuracy ? b : a)[0];
  const colors = { KNN: 'primary', SVM: 'warning', ANN: 'success' };
  const emojis = { KNN: '🔵', SVM: '🟡', ANN: '🟢' };

  document.getElementById('eval-cards').innerHTML = Object.entries(metrics).map(([name, m]) => {
    const cm = m.confusion_matrix;
    const isBest = name === best;
    return `
    <div class="col-md-4">
      <div class="card p-3 ${isBest ? 'best-model' : ''}">
        <div class="d-flex align-items-center justify-content-between mb-3">
          <h5 class="mb-0 text-${colors[name]}">${emojis[name]} ${name}</h5>
          ${isBest ? '<span class="badge bg-primary">⭐ Best</span>' : ''}
        </div>
        <!-- Metrics row -->
        <div class="row g-2 mb-3 text-center">
          <div class="col-6"><div class="metric-value text-${colors[name]}">${m.accuracy}%</div><div class="metric-label">Accuracy</div></div>
          <div class="col-6"><div class="metric-value">${m.f1}%</div><div class="metric-label">F1 Score</div></div>
          <div class="col-6"><div class="metric-value">${m.precision}%</div><div class="metric-label">Precision</div></div>
          <div class="col-6"><div class="metric-value">${m.recall}%</div><div class="metric-label">Recall</div></div>
        </div>
        <!-- Confusion matrix -->
        <div class="text-center mb-1" style="font-size:.75rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:.5px;">Confusion Matrix</div>
        ${cm ? `
        <div class="cm-grid mx-auto">
          <div class="cm-cell cm-tn">${cm[0][0]}</div>
          <div class="cm-cell cm-fp">${cm[0][1]}</div>
          <div class="cm-cell cm-fn">${cm[1][0]}</div>
          <div class="cm-cell cm-tp">${cm[1][1]}</div>
        </div>
        <div class="cm-label">TN · FP<br>FN · TP</div>` : '<p class="text-muted text-center" style="font-size:.8rem;">N/A</p>'}
      </div>
    </div>`;
  }).join('');
}

// ─────────────────────────────────────────────────────────────────
//  PREDICTION
// ─────────────────────────────────────────────────────────────────
async function runPrediction() {
  // Gather form data
  const form = document.getElementById('predict-form');
  const formData = {};
  form.querySelectorAll('input, select').forEach(el => {
    if (el.name) formData[el.name] = el.value;
  });

  // Fill defaults for columns not in the simplified form
  const defaults = {
    school:'GP', Pstatus:'T', Medu:2, Fedu:2,
    Mjob:'other', Fjob:'other', reason:'course', guardian:'mother',
    traveltime:1, schoolsup:'no', famsup:'yes', paid:'no',
    activities:'no', nursery:'yes', romantic:'no',
    famrel:4, goout:3, Dalc:1, Walc:1
  };
  Object.entries(defaults).forEach(([k, v]) => { if (!(k in formData)) formData[k] = v; });

  document.getElementById('predict-placeholder').classList.add('d-none');
  document.getElementById('predict-results').classList.add('d-none');

  try {
    const data = await apiFetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(formData)
    });

    // Majority verdict
    const isPass    = data.majority === 'Pass';
    const verdictEl = document.getElementById('majority-label');
    const cardEl    = document.getElementById('verdict-card');
    verdictEl.textContent = isPass ? '✅ Pass' : '❌ Fail';
    verdictEl.className   = `display-5 fw-bold mb-1 ${isPass ? 'text-success' : 'text-danger'}`;
    cardEl.className      = `card mb-3 ${isPass ? 'result-pass' : 'result-fail'}`;

    // Per-model cards
    const colors = { KNN: 'primary', SVM: 'warning', ANN: 'success' };
    const emojis = { KNN: '🔵', SVM: '🟡', ANN: '🟢' };
    document.getElementById('model-results').innerHTML = Object.entries(data.models).map(([name, r]) => {
      const pass = r.prediction === 'Pass';
      return `
      <div class="col-4">
        <div class="result-card ${pass ? 'result-pass' : 'result-fail'}">
          <div style="font-size:.75rem;color:var(--text-muted);">${emojis[name]} ${name}</div>
          <div class="result-label ${pass ? 'text-success' : 'text-danger'}">${r.prediction}</div>
          <div style="font-size:.75rem;color:var(--text-muted);">${r.confidence}% confidence</div>
          <div class="mt-1">
            <div class="progress" style="height:4px;">
              <div class="progress-bar bg-${pass ? 'success' : 'danger'}" style="width:${r.pass_prob}%"></div>
            </div>
            <div style="font-size:.68rem;color:var(--text-muted);margin-top:2px;">Pass prob: ${r.pass_prob}%</div>
          </div>
        </div>
      </div>`;
    }).join('');

    document.getElementById('predict-results').classList.remove('d-none');
  } catch (e) {
    alert('Prediction error: ' + e.message);
    document.getElementById('predict-placeholder').classList.remove('d-none');
  }
}

// ─────────────────────────────────────────────────────────────────
//  TAB SWITCH LISTENERS  – lazy-load each tab on first visit
// ─────────────────────────────────────────────────────────────────
const loaded = {};

document.querySelectorAll('a[data-bs-toggle="tab"]').forEach(tab => {
  tab.addEventListener('shown.bs.tab', e => {
    const target = e.target.getAttribute('href');
    if (loaded[target]) return;
    loaded[target] = true;

    if (target === '#pane-dataset')    loadDataset();
    if (target === '#pane-preprocess') loadPreprocess();
    if (target === '#pane-evaluate')   loadEvaluation();
  });
});

// ── Auto-load dashboard on page ready ───────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loaded['#pane-dashboard'] = true;
  loadDashboard();
});
