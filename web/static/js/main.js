// Logique interactive SynapseBI — Multi-Datasets & Agents Live Sync

let activeDataset = '';
let activeFilename = '';
let charts = {};
let currentAuditRole = 'business';

function switchAuditRole(role) {
  currentAuditRole = role;
  const subBusiness = document.getElementById('audit-subview-business');
  const subTechnical = document.getElementById('audit-subview-technical');
  const btnBusiness = document.getElementById('btn-role-business');
  const btnTechnical = document.getElementById('btn-role-technical');
  const badge = document.getElementById('audit-role-badge');

  if (role === 'business') {
    if (subBusiness) subBusiness.classList.remove('hidden');
    if (subTechnical) subTechnical.classList.add('hidden');
    if (btnBusiness) {
      btnBusiness.className = 'flex items-center space-x-2 px-3.5 py-2 rounded-lg bg-white text-blue-700 font-semibold shadow-sm transition';
    }
    if (btnTechnical) {
      btnTechnical.className = 'flex items-center space-x-2 px-3.5 py-2 rounded-lg text-slate-600 hover:text-slate-900 transition';
    }
    if (badge) {
      badge.innerText = '• Vue Finance & ROI';
      badge.className = 'px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200';
    }
  } else {
    if (subBusiness) subBusiness.classList.add('hidden');
    if (subTechnical) subTechnical.classList.remove('hidden');
    if (btnTechnical) {
      btnTechnical.className = 'flex items-center space-x-2 px-3.5 py-2 rounded-lg bg-white text-blue-700 font-semibold shadow-sm transition';
    }
    if (btnBusiness) {
      btnBusiness.className = 'flex items-center space-x-2 px-3.5 py-2 rounded-lg text-slate-600 hover:text-slate-900 transition';
    }
    if (badge) {
      badge.innerText = '• Vue Data Science & MLOps';
      badge.className = 'px-2.5 py-0.5 rounded-full text-xs font-semibold bg-blue-50 text-blue-700 border border-blue-200';
    }
    if (charts['shapGlobal']) {
      setTimeout(() => charts['shapGlobal'].resize(), 60);
    }
  }
}

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
  // 1. Vue Responsable Financier
  const finRisk = document.getElementById('fin-risk-exposure');
  if (finRisk) finRisk.innerText = '--';
  const finNet = document.getElementById('fin-net-savings');
  if (finNet) finNet.innerText = '--';
  const finRoi = document.getElementById('fin-roi-pct');
  if (finRoi) finRoi.innerText = '--';
  const finWaste = document.getElementById('fin-waste-reduction');
  if (finWaste) finWaste.innerText = '--';
  const finBudget = document.getElementById('fin-budget-spent');
  if (finBudget) finBudget.innerText = '--';
  const finGross = document.getElementById('fin-gross-saved');
  if (finGross) finGross.innerText = '--';

  const finLevers = document.getElementById('fin-marketing-levers-container');
  if (finLevers) {
    finLevers.innerHTML = `<p class="text-xs text-slate-400 italic">Sélectionnez un dataset pour extraire les leviers marketing optimaux.</p>`;
  }
  const finFindings = document.getElementById('fin-findings-container');
  if (finFindings) {
    finFindings.innerHTML = `<p class="text-xs text-slate-400 italic">En attente de l'évaluation financière et ROI.</p>`;
  }

  // 2. Vue Responsable Technique
  const acc = document.getElementById('xai-accuracy');
  if (acc) acc.innerText = '--';
  const rec = document.getElementById('xai-recall');
  if (rec) rec.innerText = '--';
  const f1 = document.getElementById('xai-f1');
  if (f1) f1.innerText = '--';
  const auc = document.getElementById('xai-auc');
  if (auc) auc.innerText = '--';

  const tp = document.getElementById('tech-tp');
  if (tp) tp.innerText = '--';
  const fp = document.getElementById('tech-fp');
  if (fp) fp.innerText = '--';
  const fn = document.getElementById('tech-fn');
  if (fn) fn.innerText = '--';
  const tn = document.getElementById('tech-tn');
  if (tn) tn.innerText = '--';

  if (charts['shapGlobal']) { charts['shapGlobal'].destroy(); charts['shapGlobal'] = null; }
  const techFindings = document.getElementById('tech-findings-container');
  if (techFindings) {
    techFindings.innerHTML = `<p class="text-xs text-slate-400 italic">En attente de l'audit technique et MLOps.</p>`;
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

// 4. AUDIT & XAI DATA (Dual Role: Responsable Financier / Marketing & Responsable Technique / MLOps)
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

  const m = xaiData.model_metrics;
  const fAudit = xaiData.financial_audit;
  const tAudit = xaiData.technical_audit;
  const findings = xaiData.audit_findings || [];

  // ================= 1. VUE RESPONSABLE FINANCIER & MARKETING =================
  let cur = fAudit ? fAudit.currency : '$';
  let riskExposure = fAudit ? fAudit.total_risk_exposure : 0;
  let netSavings = fAudit ? fAudit.projected_net_savings : 0;
  let roiPct = fAudit ? fAudit.campaign_roi_pct : 0;
  let wasteReduction = fAudit ? fAudit.waste_reduction_pct : 0;
  let unitVal = fAudit ? fAudit.unit_customer_value : 650;
  let campaignCost = fAudit ? fAudit.unit_campaign_cost : 25;
  let costRatio = fAudit ? fAudit.cost_inaction_ratio : 26.0;
  let budgetSpent = fAudit ? fAudit.campaign_budget_spent : 0;
  let grossSaved = fAudit ? fAudit.gross_saved_value : 0;
  let levers = fAudit && fAudit.marketing_levers ? fAudit.marketing_levers : [];

  // Fallback de calcul automatique si le rapport JSON provient d'un run antérieur
  if (!fAudit && m) {
    const pos = Math.round(m.n_test * m.positive_rate_test);
    const neg = m.n_test - pos;
    const tp = Math.round(pos * m.recall);
    const fn = pos - tp;
    const fp = m.precision > 0 ? Math.max(0, Math.round(tp / m.precision) - tp) : 0;
    const tn = Math.max(0, neg - fp);

    unitVal = 650;
    campaignCost = 25;
    costRatio = 26.0;
    riskExposure = (pos) * unitVal;
    grossSaved = tp * unitVal * 0.35;
    budgetSpent = (tp + fp) * campaignCost;
    netSavings = grossSaved - budgetSpent;
    roiPct = budgetSpent > 0 ? Number(((netSavings / budgetSpent) * 100).toFixed(1)) : 320.0;
    wasteReduction = Number(((tn / Math.max(1, tn + fp)) * 100).toFixed(1));
  }

  const elRisk = document.getElementById('fin-risk-exposure');
  if (elRisk) elRisk.innerText = `${cur}${Math.round(riskExposure).toLocaleString()}`;
  const elNet = document.getElementById('fin-net-savings');
  if (elNet) elNet.innerText = `+${cur}${Math.round(netSavings).toLocaleString()}`;
  const elRoi = document.getElementById('fin-roi-pct');
  if (elRoi) elRoi.innerText = `${roiPct}%`;
  const elWaste = document.getElementById('fin-waste-reduction');
  if (elWaste) elWaste.innerText = `${wasteReduction}%`;

  const elUnit = document.getElementById('fin-unit-val');
  if (elUnit) elUnit.innerText = `${cur}${unitVal.toFixed(2)}`;
  const elCost = document.getElementById('fin-campaign-cost');
  if (elCost) elCost.innerText = `${cur}${campaignCost.toFixed(2)}`;
  const elRatio = document.getElementById('fin-cost-ratio');
  if (elRatio) elRatio.innerText = `${costRatio}x`;
  const elBudget = document.getElementById('fin-budget-spent');
  if (elBudget) elBudget.innerText = `${cur}${Math.round(budgetSpent).toLocaleString()}`;
  const elGross = document.getElementById('fin-gross-saved');
  if (elGross) elGross.innerText = `${cur}${Math.round(grossSaved).toLocaleString()}`;

  // Plan d'Action & Leviers Marketing issus de SHAP
  const leversContainer = document.getElementById('fin-marketing-levers-container');
  if (leversContainer) {
    if (levers.length > 0) {
      leversContainer.innerHTML = levers.map(lev => `
        <div class="p-3 bg-slate-50 border border-slate-200 rounded-lg space-y-1">
          <div class="flex items-center justify-between">
            <span class="font-mono text-xs font-bold text-blue-700">${lev.feature}</span>
            <span class="px-2 py-0.5 rounded bg-blue-100 text-blue-800 text-[10px] font-bold">Levier #${lev.impact_rank}</span>
          </div>
          <p class="text-xs font-semibold text-slate-800">${lev.business_meaning}</p>
          <p class="text-xs text-slate-600"><span class="font-medium text-slate-700">Action :</span> ${lev.recommended_action}</p>
          <p class="text-[11px] text-emerald-700 font-medium">✓ ${lev.expected_gain}</p>
        </div>
      `).join('');
    } else if (xaiData.shap_summary && xaiData.shap_summary.top_features) {
      leversContainer.innerHTML = xaiData.shap_summary.top_features.slice(0, 3).map((f, i) => `
        <div class="p-3 bg-slate-50 border border-slate-200 rounded-lg space-y-1">
          <div class="flex items-center justify-between">
            <span class="font-mono text-xs font-bold text-blue-700">${f.feature}</span>
            <span class="px-2 py-0.5 rounded bg-blue-100 text-blue-800 text-[10px] font-bold">Levier #${i+1}</span>
          </div>
          <p class="text-xs font-semibold text-slate-800">Facteur décisif identifié par l'analyse SHAP (Impact : ${f.mean_abs_shap})</p>
          <p class="text-xs text-slate-600">Action recommandée : cibler les actions de rétention et d'arbitrage sur ce paramètre clé.</p>
        </div>
      `).join('');
    }
  }

  // Constats d'audit pour le Responsable Financier
  const finFindingsContainer = document.getElementById('fin-findings-container');
  if (finFindingsContainer) {
    const bizFindings = findings.filter(f => f.role === 'business' || f.role === 'all' || (f.id && f.id.includes('FIN')));
    const displayList = bizFindings.length > 0 ? bizFindings : findings;
    finFindingsContainer.innerHTML = displayList.map(f => `
      <div class="p-3 bg-slate-50 border border-slate-200 rounded-md">
        <div class="flex items-center justify-between mb-1">
          <span class="font-mono text-xs text-blue-600 font-semibold">${f.id}</span>
          <span class="text-xs px-2 py-0.5 rounded bg-emerald-100 text-emerald-800 font-bold">${f.financial_impact || 'Impact Stratégique'}</span>
        </div>
        <p class="text-sm font-semibold text-slate-900">${f.title}</p>
        <p class="text-xs text-slate-700 mt-1">${f.detail}</p>
        <p class="text-xs text-slate-500 mt-1"><span class="font-semibold text-slate-700">Recommandation CODIR :</span> ${f.recommendation}</p>
      </div>
    `).join('');
  }

  // ================= 2. VUE RESPONSABLE TECHNIQUE & MLOPS =================
  if (m) {
    const acc = document.getElementById('xai-accuracy');
    if (acc) acc.innerText = m.accuracy !== undefined ? `${(m.accuracy * 100).toFixed(1)}%` : '--';
    const rec = document.getElementById('xai-recall');
    if (rec) rec.innerText = m.recall !== undefined ? `${(m.recall * 100).toFixed(1)}%` : '--';
    const f1 = document.getElementById('xai-f1');
    if (f1) f1.innerText = m.f1 !== undefined ? `${(m.f1 * 100).toFixed(1)}%` : '--';
    const auc = document.getElementById('xai-auc');
    if (auc) auc.innerText = m.auc_roc !== undefined ? m.auc_roc.toFixed(3) : '--';
  }

  // Matrice de confusion
  let cm = tAudit ? tAudit.confusion_matrix : (m && m.confusion_matrix ? m.confusion_matrix : null);
  if (!cm && m) {
    const pos = Math.round(m.n_test * m.positive_rate_test);
    const tp = Math.round(pos * m.recall);
    const fn = pos - tp;
    const fp = m.precision > 0 ? Math.max(0, Math.round(tp / m.precision) - tp) : 0;
    const tn = Math.max(0, (m.n_test - pos) - fp);
    cm = { true_positives: tp, false_positives: fp, false_negatives: fn, true_negatives: tn };
  }

  if (cm) {
    const elTp = document.getElementById('tech-tp');
    if (elTp) elTp.innerText = Number(cm.true_positives).toLocaleString();
    const elFp = document.getElementById('tech-fp');
    if (elFp) elFp.innerText = Number(cm.false_positives).toLocaleString();
    const elFn = document.getElementById('tech-fn');
    if (elFn) elFn.innerText = Number(cm.false_negatives).toLocaleString();
    const elTn = document.getElementById('tech-tn');
    if (elTn) elTn.innerText = Number(cm.true_negatives).toLocaleString();
  }

  if (tAudit) {
    const elLeak = document.getElementById('tech-leakage-badge');
    if (elLeak) elLeak.innerText = tAudit.target_leakage_status === 'clean' ? '✓ Certifié (0 fuite)' : 'Alerte fuite';
    const elLat = document.getElementById('tech-latency');
    if (elLat) elLat.innerText = `${tAudit.inference_latency_ms} ms / requête`;
    const elStab = document.getElementById('tech-stability');
    if (elStab) elStab.innerText = tAudit.train_test_stability;
    const elCert = document.getElementById('tech-prod-cert');
    if (elCert) elCert.innerText = tAudit.production_certification.toUpperCase();
  }

  // Graphique SHAP
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

  // Constats d'audit pour le Responsable Technique
  const techFindingsContainer = document.getElementById('tech-findings-container');
  if (techFindingsContainer) {
    const techFindings = findings.filter(f => f.role === 'technical' || f.role === 'all' || (f.id && f.id.includes('TECH')));
    const displayList = techFindings.length > 0 ? techFindings : findings;
    techFindingsContainer.innerHTML = displayList.map(f => `
      <div class="p-3 bg-slate-50 border border-slate-200 rounded-md">
        <div class="flex items-center justify-between mb-1">
          <span class="font-mono text-xs text-blue-600 font-semibold">${f.id}</span>
          <span class="text-xs px-2 py-0.5 rounded bg-blue-100 text-blue-800 uppercase font-bold">${f.severity}</span>
        </div>
        <p class="text-sm font-semibold text-slate-800">${f.title}</p>
        <p class="text-xs text-slate-600 mt-1">${f.detail}</p>
        <p class="text-xs text-slate-500 mt-1"><span class="font-semibold text-slate-700">Directive MLOps :</span> ${f.recommendation}</p>
      </div>
    `).join('');
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
