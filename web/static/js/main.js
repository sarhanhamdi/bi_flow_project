// Logique interactive SynapseBI — Multi-Datasets & Agents Live Sync

let activeDataset = '';
let activeFilename = '';
let charts = {};

async function initDatasetSelector() {
  const status = await fetchJSON('/api/status');
  const selector = document.getElementById('dataset-selector');
  if (status && selector) {
    if (status.available_files && status.available_files.length > 0) {
      selector.innerHTML = `
        <option value="">-- Sélectionner un dataset --</option>
        ${status.available_files.map(f => `
          <option value="${f.id}" data-file="${f.filename}" ${f.id === activeDataset ? 'selected' : ''}>
            ${f.label}
          </option>
        `).join('')}
      `;
    } else {
      selector.innerHTML = `<option value="">-- Aucun dataset trouvé --</option>`;
    }
  }
}

function changeActiveDataset(name) {
  const selector = document.getElementById('dataset-selector');
  const selectedOpt = selector ? selector.options[selector.selectedIndex] : null;
  activeFilename = selectedOpt && selectedOpt.dataset.file ? selectedOpt.dataset.file : (name ? `${name}.csv` : '');
  activeDataset = name;

  if (!activeDataset) {
    resetAllViews();
    return;
  }

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
      if (statusMsg) statusMsg.innerHTML = `<span class="text-emerald-600 font-bold">✓ Fichier ${data.filename} enregistré. Lancement du pipeline...</span>`;
      
      const datasetName = data.filename.replace(/\.csv$/i, '');
      activeDataset = datasetName;
      activeFilename = data.filename;
      
      resetAllViews();

      await fetch(`/api/pipeline/run?source_filename=${encodeURIComponent(data.filename)}&preset=none&skip_xai=false`, {
        method: 'POST'
      });
      
      showView('view-execution');
      triggerPipelineRun();
      await initDatasetSelector();
    } else {
      if (statusMsg) statusMsg.innerText = `Erreur: ${data.detail || 'Échec upload'}`;
    }
  } catch (err) {
    if (statusMsg) statusMsg.innerText = `Erreur réseau : ${err.message}`;
  }
}

function resetAllViews() {
  resetDashboardView();
  resetGovernanceView();
  resetXAIView();
  resetExecutionView();
  resetWelcomeView();
}

function resetGovernanceView() {
  const container = document.getElementById('governance-rules-container');
  if (container) {
    container.innerHTML = `<tr><td colspan="5" class="py-6 text-center text-slate-400 text-xs">Aucun rapport de gouvernance disponible. Sélectionnez un dataset et lancez le pipeline.</td></tr>`;
  }
}

function resetXAIView() {
  document.getElementById('xai-accuracy').innerText = '--';
  document.getElementById('xai-recall').innerText = '--';
  document.getElementById('xai-f1').innerText = '--';
  document.getElementById('xai-auc').innerText = '--';
  if (charts['shapGlobal']) { charts['shapGlobal'].destroy(); charts['shapGlobal'] = null; }
  const findings = document.getElementById('xai-findings-container');
  if (findings) {
    findings.innerHTML = `<p class="text-xs text-slate-400 italic">Sélectionnez un dataset et lancez le pipeline pour générer l'audit XAI.</p>`;
  }
}

function resetExecutionView() {
  const duration = document.getElementById('exec-total-duration');
  if (duration) duration.innerText = '--';
  const badge = document.getElementById('exec-status-badge');
  if (badge) {
    badge.innerText = 'EN ATTENTE';
    badge.className = 'px-3 py-1 bg-slate-200 text-slate-700 rounded-full font-bold text-xs uppercase';
  }
  const container = document.getElementById('pipeline-timeline-container');
  if (container) {
    container.innerHTML = `
      <div class="p-8 text-center bg-white border border-slate-200 rounded-lg text-slate-500 text-xs">
        Téléversez un dataset ou cliquez sur « Lancer le Pipeline » pour voir les agents s'exécuter.
      </div>
    `;
  }
}

function resetWelcomeView() {
  const c = document.getElementById('welcome-stat-churn');
  if (c) c.innerText = '--';
  const l = document.getElementById('welcome-stat-loss');
  if (l) l.innerText = '--';
  const a = document.getElementById('welcome-stat-auc');
  if (a) a.innerText = '--';
}

function resetDashboardView() {
  document.getElementById('dash-customer-count').innerText = '--';
  document.getElementById('dash-churn-rate').innerText = '--';
  document.getElementById('dash-arpu').innerText = '--';
  document.getElementById('dash-revenue').innerText = '--';
  document.getElementById('dash-insight-title').innerText = 'En attente d\'analyse';
  document.getElementById('dash-insight-body').innerText = 'Sélectionnez un dataset et lancez le pipeline pour générer les insights.';
  document.getElementById('dash-rec-title').innerText = 'En attente de recommandations';
  document.getElementById('dash-rec-body').innerText = 'Les actions d\'optimisation prioritaires s\'afficheront ici après exécution.';

  if (charts['churnTrend']) { charts['churnTrend'].destroy(); charts['churnTrend'] = null; }
  if (charts['churnPlan']) { charts['churnPlan'].destroy(); charts['churnPlan'] = null; }
  if (charts['shapGlobal']) { charts['shapGlobal'].destroy(); charts['shapGlobal'] = null; }
}

async function fetchJSON(url) {
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    return await res.json();
  } catch (e) {
    return null;
  }
}

function showView(viewId) {
  document.querySelectorAll('.view-section').forEach(el => el.classList.add('hidden'));
  const target = document.getElementById(viewId);
  if (target) target.classList.remove('hidden');

  document.querySelectorAll('.nav-item').forEach(el => {
    el.classList.remove('bg-blue-600', 'text-white', 'font-medium');
    el.classList.add('text-slate-400', 'hover:bg-slate-900', 'hover:text-slate-200');
  });
  const navBtn = document.getElementById(`nav-${viewId}`);
  if (navBtn) {
    navBtn.classList.remove('text-slate-400', 'hover:bg-slate-900', 'hover:text-slate-200');
    navBtn.classList.add('bg-blue-600', 'text-white', 'font-medium');
  }

  if (viewId === 'view-dashboard') loadDashboardData();
  if (viewId === 'view-execution') loadExecutionData();
  if (viewId === 'view-xai') loadXAIData();
  if (viewId === 'view-governance') loadGovernanceData();
  if (viewId === 'view-welcome') loadWelcomeData();
}

// 1. WELCOME DATA
async function loadWelcomeData() {
  if (!activeDataset) return;
  const kpiData = await fetchJSON(`/api/reports/${activeDataset}/kpi`);
  const xaiData = await fetchJSON(`/api/reports/${activeDataset}/xai`);

  if (kpiData && kpiData.kpis) {
    const primaryRate = kpiData.kpis.find(k => k.id === 'churn_rate' || k.id === 'cdr_incident_rate' || k.id === 'ticket_resolution_rate');
    if (primaryRate && primaryRate.value !== null) {
      document.getElementById('welcome-stat-churn').innerText = `${primaryRate.value}%`;
    }
    const loss = kpiData.kpis.find(k => k.id === 'monthly_revenue_lost' || k.id === 'total_revenue' || k.id === 'ticket_count');
    if (loss && loss.value !== null) {
      document.getElementById('welcome-stat-loss').innerText = typeof loss.value === 'number' && loss.value > 1000 ? `$${(loss.value/1000).toFixed(1)}k` : `${loss.value}`;
    }
  }
  if (xaiData && xaiData.model_metrics && xaiData.model_metrics.auc_roc !== undefined) {
    document.getElementById('welcome-stat-auc').innerText = xaiData.model_metrics.auc_roc.toFixed(3);
  }
}

// 2. DASHBOARD DATA
async function loadDashboardData() {
  if (!activeDataset) {
    resetDashboardView();
    return;
  }
  const kpiData = await fetchJSON(`/api/reports/${activeDataset}/kpi`);
  const analystData = await fetchJSON(`/api/reports/${activeDataset}/analyst`);
  const dashReport = await fetchJSON(`/api/reports/${activeDataset}/dashboard`);

  if (kpiData && kpiData.kpis && kpiData.kpis.length > 0) {
    const kpis = kpiData.kpis;
    // Carte 1
    const kpi1 = kpis.find(k => k.id === 'customer_count' || k.id === 'ticket_count') || kpis[0];
    if (kpi1) {
      document.getElementById('dash-label-1').innerText = kpi1.name;
      document.getElementById('dash-customer-count').innerText = Number(kpi1.value).toLocaleString();
    }
    // Carte 2
    const kpi2 = kpis.find(k => k.id === 'churn_rate' || k.id === 'cdr_incident_rate' || k.id === 'ticket_resolution_rate' || k.id === 'dispute_rate') || kpis[1] || kpis[0];
    if (kpi2) {
      document.getElementById('dash-label-2').innerText = kpi2.name;
      document.getElementById('dash-churn-rate').innerText = kpi2.unit === '%' ? `${kpi2.value}%` : `${kpi2.value}`;
    }
    // Carte 3
    const kpi3 = kpis.find(k => k.id === 'arpu' || k.id === 'avg_throughput' || k.id === 'avg_mos_voice' || k.id === 'avg_csat' || k.id === 'avg_latency') || kpis[2] || kpis[0];
    if (kpi3) {
      document.getElementById('dash-label-3').innerText = kpi3.name;
      document.getElementById('dash-arpu').innerText = `${kpi3.value} ${kpi3.unit || ''}`.trim();
    }
    // Carte 4
    const kpi4 = kpis.find(k => k.id === 'monthly_revenue_lost' || k.id === 'total_revenue' || k.id === 'network_availability' || k.id === 'timely_response_rate') || kpis[3] || kpis[0];
    if (kpi4) {
      document.getElementById('dash-label-4').innerText = kpi4.name;
      if (kpi4.unit === '%') {
        document.getElementById('dash-revenue').innerText = `${kpi4.value}%`;
      } else if (Number(kpi4.value) >= 1000000) {
        document.getElementById('dash-revenue').innerText = `$${(Number(kpi4.value) / 1000000).toFixed(1)}m`;
      } else {
        document.getElementById('dash-revenue').innerText = `${kpi4.value} ${kpi4.unit || ''}`.trim();
      }
    }
  } else {
    resetDashboardView();
  }

  if (analystData) {
    if (analystData.insights && analystData.insights.length > 0) {
      document.getElementById('dash-insight-title').innerText = analystData.insights[0].title;
      document.getElementById('dash-insight-body').innerText = analystData.insights[0].observation;
    } else {
      document.getElementById('dash-insight-title').innerText = 'Analyse effectuée';
      document.getElementById('dash-insight-body').innerText = analystData.llm_narrative || 'Aucune anomalie critique détectée.';
    }
    if (analystData.recommendations && analystData.recommendations.length > 0) {
      document.getElementById('dash-rec-title').innerText = analystData.recommendations[0].title || 'Recommandation Prioritaire';
      document.getElementById('dash-rec-body').innerText = analystData.recommendations[0].action;
    } else {
      document.getElementById('dash-rec-title').innerText = 'Recommandation';
      document.getElementById('dash-rec-body').innerText = 'Maintenir la surveillance des indicateurs de référence.';
    }
  }

  initDashboardCharts(dashReport, kpiData);
}

function initDashboardCharts(dashReport, kpiData) {
  let chart1Title = 'Tendance / Répartition';
  let chart1Labels = ['Période 1', 'Période 2', 'Période 3', 'Période 4'];
  let chart1Values = [0, 0, 0, 0];

  let chart2Title = 'Ventilation par Segment';
  let chart2Labels = ['Segment A', 'Segment B', 'Segment C'];
  let chart2Values = [0, 0, 0];

  if (dashReport && dashReport.visuals) {
    const barVisuals = dashReport.visuals.filter(v => v.type === 'bar' && Array.isArray(v.data) && v.data.length > 0);
    if (barVisuals.length > 0) {
      const v1 = barVisuals[0];
      chart2Title = v1.title || 'Ventilation observée';
      chart2Labels = v1.data.map(d => d.segment);
      chart2Values = v1.data.map(d => d.value !== undefined ? d.value : d.rows);

      if (barVisuals.length > 1) {
        const v2 = barVisuals[1];
        chart1Title = v2.title || 'Répartition secondaire';
        chart1Labels = v2.data.map(d => d.segment);
        chart1Values = v2.data.map(d => d.value !== undefined ? d.value : d.rows);
      }
    }
  } else if (kpiData && kpiData.kpis) {
    const kpiWithBreakdown = kpiData.kpis.find(k => k.breakdowns && Object.keys(k.breakdowns).length > 0);
    if (kpiWithBreakdown) {
      const firstDim = Object.keys(kpiWithBreakdown.breakdowns)[0];
      const segs = kpiWithBreakdown.breakdowns[firstDim];
      chart2Title = `${kpiWithBreakdown.name} par ${firstDim}`;
      chart2Labels = segs.map(s => s.segment);
      chart2Values = segs.map(s => s.value || 0);
    }
  }

  const title1El = document.getElementById('chart-title-1');
  if (title1El) title1El.innerText = chart1Title;
  const title2El = document.getElementById('chart-title-2');
  if (title2El) title2El.innerText = chart2Title;

  const ctxPlan = document.getElementById('chart-churn-plan');
  if (ctxPlan) {
    if (charts['churnPlan']) charts['churnPlan'].destroy();
    charts['churnPlan'] = new Chart(ctxPlan, {
      type: 'bar',
      data: {
        labels: chart2Labels,
        datasets: [{
          label: 'Valeur observée',
          data: chart2Values,
          backgroundColor: '#2563eb',
          borderRadius: 4
        }]
      },
      options: { indexAxis: 'y', responsive: true, maintainAspectRatio: false }
    });
  }

  const ctxTrend = document.getElementById('chart-churn-trend');
  if (ctxTrend) {
    if (charts['churnTrend']) charts['churnTrend'].destroy();
    charts['churnTrend'] = new Chart(ctxTrend, {
      type: 'line',
      data: {
        labels: chart1Labels,
        datasets: [{
          label: 'Valeur observée',
          data: chart1Values,
          borderColor: '#2563eb',
          backgroundColor: 'rgba(37, 99, 235, 0.05)',
          tension: 0.3,
          fill: true
        }]
      },
      options: { responsive: true, maintainAspectRatio: false }
    });
  }
}

// 3. PIPELINE EXECUTION DATA
async function loadExecutionData() {
  if (!activeDataset) return;
  const orchData = await fetchJSON(`/api/reports/${activeDataset}/orchestrator`);
  const container = document.getElementById('pipeline-timeline-container');
  if (!container) return;

  if (orchData && orchData.steps) {
    document.getElementById('exec-total-duration').innerText = `${orchData.total_duration_seconds}s`;
    const badge = document.getElementById('exec-status-badge');
    badge.innerText = orchData.status.toUpperCase();
    badge.className = orchData.status === 'success' 
      ? 'px-3 py-1 bg-emerald-200 text-emerald-900 rounded-full font-bold text-xs uppercase'
      : 'px-3 py-1 bg-amber-200 text-amber-900 rounded-full font-bold text-xs uppercase';

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
  const targetFile = activeFilename || (activeDataset ? `${activeDataset}.csv` : '');
  if (!targetFile) return;

  const btn = document.getElementById('btn-run-pipeline');
  if (btn) {
    btn.disabled = true;
    btn.innerText = 'Exécution en cours...';
  }
  
  resetDashboardView();
  resetGovernanceView();
  resetXAIView();

  await fetch(`/api/pipeline/run?source_filename=${encodeURIComponent(targetFile)}&preset=none&skip_xai=false`, { method: 'POST' });

  const interval = setInterval(async () => {
    const status = await fetchJSON('/api/status');
    if (status && !status.pipeline_state.is_running) {
      clearInterval(interval);
      if (btn) {
        btn.disabled = false;
        btn.innerText = 'Relancer le Pipeline';
      }
      loadExecutionData();
      loadDashboardData();
      loadXAIData();
      loadGovernanceData();
      loadWelcomeData();
    }
  }, 2000);
}

// 4. AUDIT & XAI DATA
async function loadXAIData() {
  if (!activeDataset) {
    resetXAIView();
    return;
  }
  const xaiData = await fetchJSON(`/api/reports/${activeDataset}/xai`);
  if (!xaiData) {
    resetXAIView();
    return;
  }

  if (xaiData.model_metrics) {
    const m = xaiData.model_metrics;
    document.getElementById('xai-accuracy').innerText = m.accuracy !== undefined ? `${(m.accuracy * 100).toFixed(1)}%` : '--';
    document.getElementById('xai-recall').innerText = m.recall !== undefined ? `${(m.recall * 100).toFixed(1)}%` : '--';
    document.getElementById('xai-f1').innerText = m.f1 !== undefined ? `${(m.f1 * 100).toFixed(1)}%` : '--';
    document.getElementById('xai-auc').innerText = m.auc_roc !== undefined ? m.auc_roc.toFixed(3) : '--';
  } else {
    resetXAIView();
  }

  if (xaiData.shap_summary && xaiData.shap_summary.top_features && xaiData.shap_summary.top_features.length > 0) {
    const top = xaiData.shap_summary.top_features;
    const ctxShap = document.getElementById('chart-shap-global');
    if (ctxShap) {
      if (charts['shapGlobal']) charts['shapGlobal'].destroy();
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
  if (findingsContainer && xaiData.audit_findings && xaiData.audit_findings.length > 0) {
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
  } else if (findingsContainer) {
    findingsContainer.innerHTML = `<p class="text-xs text-slate-400 italic">Aucune anomalie d'audit relevée pour ce modèle.</p>`;
  }
}

// 5. GOVERNANCE DATA
async function loadGovernanceData() {
  if (!activeDataset) {
    resetGovernanceView();
    return;
  }
  const etlData = await fetchJSON(`/api/reports/${activeDataset}/etl`);
  const container = document.getElementById('governance-rules-container');
  if (container) {
    if (etlData && etlData.actions && etlData.actions.length > 0) {
      container.innerHTML = etlData.actions.map((act, i) => `
        <tr class="border-b border-slate-100 text-xs">
          <td class="py-2.5 font-mono text-slate-600">RULE-00${i+1}</td>
          <td class="py-2.5 font-medium text-slate-800">${act.rule}</td>
          <td class="py-2.5 font-mono text-blue-600">${act.column || 'Dataset'}</td>
          <td class="py-2.5">${act.rows_affected} lignes</td>
          <td class="py-2.5"><span class="px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">Corrigé</span></td>
        </tr>
      `).join('');
    } else {
      resetGovernanceView();
    }
  }
}

document.addEventListener('DOMContentLoaded', () => {
  resetAllViews();
  initDatasetSelector();
  showView('view-welcome');
});
