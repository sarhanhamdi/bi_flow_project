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
      if (!activeDataset && status.output_datasets && status.output_datasets.length > 0) {
        const defaultOne = status.available_files.find(f => status.output_datasets.includes(f.id));
        if (defaultOne) {
          selector.value = defaultOne.id;
          changeActiveDataset(defaultOne.id);
        }
      }
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
  const recTitleEl = document.getElementById('dash-rec-title');
  if (recTitleEl) recTitleEl.innerText = 'En attente de recommandations';
  const recBodyEl = document.getElementById('dash-rec-body');
  if (recBodyEl) recBodyEl.innerText = 'Les actions d\'optimisation prioritaires s\'afficheront ici après exécution.';

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
    // Ne garder que les KPIs effectivement calculés (valeur non null)
    const validKPIs = kpiData.kpis.filter(k => k.value !== null && k.status !== 'not_computable');

    if (validKPIs.length >= 4) {
      setKPICard(1, validKPIs[0]);
      setKPICard(2, validKPIs[1]);
      setKPICard(3, validKPIs[2]);
      setKPICard(4, validKPIs[3]);
    } else if (validKPIs.length > 0) {
      validKPIs.forEach((k, idx) => setKPICard(idx + 1, k));
      for (let i = validKPIs.length + 1; i <= 4; i++) {
        document.getElementById(`dash-label-${i}`).innerText = '--';
        document.getElementById(i === 1 ? 'dash-customer-count' : (i === 2 ? 'dash-churn-rate' : (i === 3 ? 'dash-arpu' : 'dash-revenue'))).innerText = '--';
      }
    } else {
      resetDashboardView();
    }
  } else {
    resetDashboardView();
  }

function setKPICard(cardIndex, k) {
  const lbl = document.getElementById(`dash-label-${cardIndex}`);
  const valId = cardIndex === 1 ? 'dash-customer-count' : (cardIndex === 2 ? 'dash-churn-rate' : (cardIndex === 3 ? 'dash-arpu' : 'dash-revenue'));
  const valEl = document.getElementById(valId);
  if (!lbl || !valEl) return;

  lbl.innerText = k.name;
  if (k.unit === '%') {
    valEl.innerText = `${k.value}%`;
  } else if (k.unit === 'devise' && Number(k.value) >= 1000000) {
    valEl.innerText = `$${(Number(k.value) / 1000000).toFixed(1)}m`;
  } else if (typeof k.value === 'number') {
    valEl.innerText = `${k.value.toLocaleString()} ${k.unit || ''}`.trim();
  } else {
    valEl.innerText = `${k.value} ${k.unit || ''}`.trim();
  }
}

  if (analystData) {
    const inTitle = document.getElementById('dash-insight-title');
    const inBody = document.getElementById('dash-insight-body');
    if (analystData.insights && analystData.insights.length > 0) {
      if (inTitle) inTitle.innerText = analystData.insights[0].title || 'Insight Identifié';
      if (inBody) inBody.innerText = analystData.insights[0].finding || analystData.insights[0].observation || analystData.insights[0].evidence || '';
    } else {
      if (inTitle) inTitle.innerText = 'Analyse effectuée';
      if (inBody) inBody.innerText = analystData.llm_narrative || 'Aucune anomalie critique détectée.';
    }
    const recTitle = document.getElementById('dash-rec-title');
    const recBody = document.getElementById('dash-rec-body');
    if (analystData.recommendations && analystData.recommendations.length > 0) {
      if (recTitle) recTitle.innerText = analystData.recommendations[0].title || analystData.recommendations[0].target || 'Recommandation Prioritaire';
      if (recBody) recBody.innerText = analystData.recommendations[0].action || analystData.recommendations[0].rationale || '';
    } else {
      if (recTitle) recTitle.innerText = 'Recommandation';
      if (recBody) recBody.innerText = 'Maintenir la surveillance des indicateurs de référence.';
    }
  }

  initDashboardCharts(dashReport, kpiData);
}

function initDashboardCharts(dashReport, kpiData) {
  let chart1Title = 'Tendance / Répartition';
  let chart1Labels = ['Période 1', 'Période 2', 'Période 3', 'Période 4'];
  let chart1Values = [0, 0, 0, 0];
  let chart1Type = 'bar';

  let chart2Title = 'Ventilation par Segment';
  let chart2Labels = ['Segment A', 'Segment B', 'Segment C'];
  let chart2Values = [0, 0, 0];
  let chart2Type = 'bar';

  if (dashReport && dashReport.visuals) {
    const validVisuals = dashReport.visuals.filter(v => (v.type === 'bar' || v.type === 'line') && Array.isArray(v.data) && v.data.length > 0);
    if (validVisuals.length > 0) {
      const v1 = validVisuals[0];
      chart1Title = v1.title || 'Répartition principale';
      chart1Labels = v1.data.map(d => d.segment || d.period || d.label || String(d.x || ''));
      chart1Values = v1.data.map(d => d.value !== undefined ? d.value : (d.rows !== undefined ? d.rows : (d.y || 0)));
      chart1Type = v1.type === 'line' ? 'line' : 'bar';

      if (validVisuals.length > 1) {
        const v2 = validVisuals[1];
        chart2Title = v2.title || 'Ventilation secondaire';
        chart2Labels = v2.data.map(d => d.segment || d.period || d.label || String(d.x || ''));
        chart2Values = v2.data.map(d => d.value !== undefined ? d.value : (d.rows !== undefined ? d.rows : (d.y || 0)));
        chart2Type = v2.type === 'line' ? 'line' : 'bar';
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

  const ctxTrend = document.getElementById('chart-churn-trend');
  if (ctxTrend) {
    if (charts['churnTrend']) {
      charts['churnTrend'].destroy();
      charts['churnTrend'] = null;
    }
    charts['churnTrend'] = new Chart(ctxTrend, {
      type: chart1Type,
      data: {
        labels: chart1Labels,
        datasets: [{
          label: 'Valeur observée',
          data: chart1Values,
          backgroundColor: '#3b82f6',
          borderColor: '#2563eb',
          borderRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: { beginAtZero: false }
        }
      }
    });
  }

  const ctxPlan = document.getElementById('chart-churn-plan');
  if (ctxPlan) {
    if (charts['churnPlan']) {
      charts['churnPlan'].destroy();
      charts['churnPlan'] = null;
    }
    charts['churnPlan'] = new Chart(ctxPlan, {
      type: chart2Type,
      data: {
        labels: chart2Labels,
        datasets: [{
          label: 'Valeur observée',
          data: chart2Values,
          backgroundColor: '#0ea5e9',
          borderRadius: 4
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false
      }
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
