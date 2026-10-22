// Logique interactive SynapseBI — Multi-Datasets & Agents Live Sync

let activeDataset = 'telco';
let charts = {};

async function initDatasetSelector() {
  const status = await fetchJSON('/api/status');
  const selector = document.getElementById('dataset-selector');
  if (status && status.datasets && selector) {
    selector.innerHTML = status.datasets.map(d => `
      <option value="${d}" ${d === activeDataset ? 'selected' : ''}>${d}.csv</option>
    `).join('');
  }
}

function changeActiveDataset(name) {
  activeDataset = name;
  loadDashboardData();
  loadExecutionData();
  loadXAIData();
  loadGovernanceData();
  loadWelcomeData();
}

async function handleFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const statusMsg = document.getElementById('upload-status-msg');
  if (statusMsg) statusMsg.innerText = `Téléversement de ${file.name}...`;

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/upload', {
      method: 'POST',
      body: formData
    });
    const data = await res.json();
    if (res.ok) {
      if (statusMsg) statusMsg.innerHTML = `<span class="text-emerald-600 font-bold">✓ Fichier ${data.filename} enregistré. Lancement de l'analyse...</span>`;
      
      const datasetName = data.filename.replace(/\.csv$/i, '');
      activeDataset = datasetName;
      
      // Déclencher le pipeline
      await fetch(`/api/pipeline/run?source_filename=${encodeURIComponent(data.filename)}&preset=none&skip_xai=true`, {
        method: 'POST'
      });
      
      showView('view-execution');
      triggerPipelineRun(data.filename);
      await initDatasetSelector();
    } else {
      if (statusMsg) statusMsg.innerText = `Erreur: ${data.detail || 'Échec upload'}`;
    }
  } catch (err) {
    if (statusMsg) statusMsg.innerText = `Erreur réseau : ${err.message}`;
  }
}

async function fetchJSON(url) {
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    return await res.json();
  } catch (e) {
    console.error("Fetch error:", e);
    return null;
  }
}

function showView(viewId) {
  document.querySelectorAll('.view-section').forEach(el => el.classList.add('hidden'));
  const target = document.getElementById(viewId);
  if (target) target.classList.remove('hidden');

  // Sidebar highlight
  document.querySelectorAll('.nav-item').forEach(el => {
    el.classList.remove('bg-blue-600', 'text-white', 'font-medium');
    el.classList.add('text-slate-400', 'hover:bg-slate-900', 'hover:text-slate-200');
  });
  const navBtn = document.getElementById(`nav-${viewId}`);
  if (navBtn) {
    navBtn.classList.remove('text-slate-400', 'hover:bg-slate-900', 'hover:text-slate-200');
    navBtn.classList.add('bg-blue-600', 'text-white', 'font-medium');
  }

  // Load view content
  if (viewId === 'view-dashboard') loadDashboardData();
  if (viewId === 'view-execution') loadExecutionData();
  if (viewId === 'view-xai') loadXAIData();
  if (viewId === 'view-governance') loadGovernanceData();
  if (viewId === 'view-welcome') loadWelcomeData();
}

// 1. WELCOME DATA
async function loadWelcomeData() {
  const kpiData = await fetchJSON(`/api/reports/${activeDataset}/kpi`);
  const xaiData = await fetchJSON(`/api/reports/${activeDataset}/xai`);

  if (kpiData && kpiData.kpis) {
    const churn = kpiData.kpis.find(k => k.id === 'churn_rate');
    if (churn) document.getElementById('welcome-stat-churn').innerText = `${churn.value}%`;
    const loss = kpiData.kpis.find(k => k.id === 'monthly_lost_revenue');
    if (loss) document.getElementById('welcome-stat-loss').innerText = `$${(Number(loss.value)/1000).toFixed(1)}k`;
  }
  if (xaiData && xaiData.model_metrics) {
    document.getElementById('welcome-stat-auc').innerText = xaiData.model_metrics.auc_roc.toFixed(3);
  }
}

// 2. DASHBOARD DATA
async function loadDashboardData() {
  const kpiData = await fetchJSON(`/api/reports/${activeDataset}/kpi`);
  const analystData = await fetchJSON(`/api/reports/${activeDataset}/analyst`);

  if (kpiData && kpiData.kpis) {
    kpiData.kpis.forEach(k => {
      if (k.id === 'customer_count') document.getElementById('dash-customer-count').innerText = Number(k.value).toLocaleString();
      if (k.id === 'churn_rate') document.getElementById('dash-churn-rate').innerText = `${k.value}%`;
      if (k.id === 'arpu_monthly') document.getElementById('dash-arpu').innerText = `$${k.value}`;
      if (k.id === 'total_revenue') document.getElementById('dash-revenue').innerText = `$${(Number(k.value) / 1000000).toFixed(1)}m`;
    });
  }

  if (analystData) {
    if (analystData.insights && analystData.insights.length > 0) {
      document.getElementById('dash-insight-title').innerText = analystData.insights[0].title;
      document.getElementById('dash-insight-body').innerText = analystData.insights[0].observation;
    }
    if (analystData.recommendations && analystData.recommendations.length > 0) {
      document.getElementById('dash-rec-body').innerText = analystData.recommendations[0].action;
    }
  }

  initDashboardCharts();
}

function initDashboardCharts() {
  const ctxChurn = document.getElementById('chart-churn-trend');
  if (ctxChurn && !charts['churnTrend']) {
    charts['churnTrend'] = new Chart(ctxChurn, {
      type: 'line',
      data: {
        labels: ['Avr', 'Mai', 'Juin', 'Juil', 'Août', 'Sept'],
        datasets: [{
          label: 'Taux de Churn (%)',
          data: [2.5, 2.4, 2.3, 2.2, 2.1, 2.0],
          borderColor: '#2563eb',
          backgroundColor: 'rgba(37, 99, 235, 0.05)',
          tension: 0.3,
          fill: true
        }]
      },
      options: { responsive: true, maintainAspectRatio: false }
    });
  }

  const ctxPlan = document.getElementById('chart-churn-plan');
  if (ctxPlan && !charts['churnPlan']) {
    charts['churnPlan'] = new Chart(ctxPlan, {
      type: 'bar',
      data: {
        labels: ['Month-to-month', 'One year', 'Two year'],
        datasets: [{
          label: 'Taux de Churn (%)',
          data: [42.71, 11.27, 2.83],
          backgroundColor: '#2563eb',
          borderRadius: 4
        }]
      },
      options: { indexAxis: 'y', responsive: true, maintainAspectRatio: false }
    });
  }
}

// 3. PIPELINE EXECUTION DATA
async function loadExecutionData() {
  const orchData = await fetchJSON(`/api/reports/${activeDataset}/orchestrator`);
  const container = document.getElementById('pipeline-timeline-container');
  if (!container) return;

  if (orchData && orchData.steps) {
    document.getElementById('exec-total-duration').innerText = `${orchData.total_duration_seconds}s`;
    document.getElementById('exec-status-badge').innerText = orchData.status.toUpperCase();

    container.innerHTML = orchData.steps.map(step => `
      <div class="flex items-center justify-between p-4 bg-white border border-slate-200 rounded-lg hover:border-blue-400 transition">
        <div class="flex items-center space-x-3">
          <div class="w-8 h-8 rounded-full ${step.status === 'success' ? 'bg-emerald-100 text-emerald-600' : 'bg-rose-100 text-rose-600'} flex items-center justify-center font-bold">
            ${step.status === 'success' ? '✓' : '!'}
          </div>
          <div>
            <h4 class="font-semibold text-slate-800 capitalize">${step.agent_name.replace('_', ' ')} Agent</h4>
            <p class="text-xs text-slate-500">${step.message || 'Exécuté avec succès'}</p>
          </div>
        </div>
        <div class="flex items-center space-x-4">
          <span class="font-mono text-xs text-slate-500">${step.duration_seconds}s</span>
          <span class="px-2.5 py-1 text-xs font-medium rounded-full ${step.status === 'success' ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' : 'bg-rose-50 text-rose-700'}">
            ${step.status}
          </span>
        </div>
      </div>
    `).join('');
  }
}

async function triggerPipelineRun() {
  const btn = document.getElementById('btn-run-pipeline');
  if (btn) {
    btn.disabled = true;
    btn.innerText = 'Exécution en cours...';
  }
  await fetch('/api/pipeline/run', { method: 'POST' });

  const interval = setInterval(async () => {
    const status = await fetchJSON('/api/status');
    if (status && !status.pipeline_state.is_running) {
      clearInterval(interval);
      if (btn) {
        btn.disabled = false;
        btn.innerText = 'Relancer le Pipeline';
      }
      loadExecutionData();
    }
  }, 2000);
}

// 4. AUDIT & XAI DATA
async function loadXAIData() {
  const xaiData = await fetchJSON(`/api/reports/${activeDataset}/xai`);
  if (!xaiData) return;

  if (xaiData.model_metrics) {
    const m = xaiData.model_metrics;
    document.getElementById('xai-accuracy').innerText = `${(m.accuracy * 100).toFixed(1)}%`;
    document.getElementById('xai-recall').innerText = `${(m.recall * 100).toFixed(1)}%`;
    document.getElementById('xai-f1').innerText = `${(m.f1 * 100).toFixed(1)}%`;
    document.getElementById('xai-auc').innerText = m.auc_roc.toFixed(3);
  }

  if (xaiData.shap_summary && xaiData.shap_summary.top_features) {
    const top = xaiData.shap_summary.top_features;
    const ctxShap = document.getElementById('chart-shap-global');
    if (ctxShap && !charts['shapGlobal']) {
      charts['shapGlobal'] = new Chart(ctxShap, {
        type: 'bar',
        data: {
          labels: top.map(t => t.feature),
          datasets: [{
            label: 'Impact Moyen (|SHAP value|)',
            data: top.map(t => t.mean_abs_shap),
            backgroundColor: '#2563eb',
            borderRadius: 4
          }]
        },
        options: { indexAxis: 'y', responsive: true, maintainAspectRatio: false }
      });
    }
  }

  const findingsContainer = document.getElementById('xai-findings-container');
  if (findingsContainer && xaiData.audit_findings) {
    findingsContainer.innerHTML = xaiData.audit_findings.map(f => `
      <div class="p-3 bg-slate-50 border border-slate-200 rounded-md">
        <div class="flex items-center justify-between mb-1">
          <span class="font-mono text-xs text-blue-600 font-semibold">${f.id}</span>
          <span class="text-xs px-2 py-0.5 rounded bg-blue-100 text-blue-800 uppercase font-bold">${f.severity}</span>
        </div>
        <p class="text-sm font-medium text-slate-800">${f.title}</p>
        <p class="text-xs text-slate-600 mt-1">${f.detail}</p>
      </div>
    `).join('');
  }
}

// 5. GOVERNANCE DATA
async function loadGovernanceData() {
  const etlData = await fetchJSON(`/api/reports/${activeDataset}/etl`);
  const container = document.getElementById('governance-rules-container');
  if (container && etlData && etlData.actions) {
    container.innerHTML = etlData.actions.map((act, i) => `
      <tr class="border-b border-slate-100 text-xs">
        <td class="py-2.5 font-mono text-slate-600">RULE-00${i+1}</td>
        <td class="py-2.5 font-medium text-slate-800">${act.rule}</td>
        <td class="py-2.5 font-mono text-blue-600">${act.column || 'Dataset'}</td>
        <td class="py-2.5">${act.rows_affected} lignes</td>
        <td class="py-2.5"><span class="px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">Corrigé</span></td>
      </tr>
    `).join('');
  }
}

document.addEventListener('DOMContentLoaded', () => {
  initDatasetSelector();
  showView('view-welcome');
});
