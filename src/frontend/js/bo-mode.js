/**
 * Personalised Bayesian Optimisation mode.
 *
 * Single class wiring the new `/api/bo/*` namespace to the unified BO
 * dashboard in index.html (#bo-screen). Replaces the previous per-objective
 * OptimizationManager. Renders Plotly charts for the best-y curve and the
 * cross-participant insight figures.
 */

import { SurveyManager } from './systematic-mode/survey-manager.js';

const SURVEY_CONTAINER_ID = 'bo-survey-sections';

class BOMode {
    constructor(api, ui) {
        this.api = api;
        this.ui = ui;
        this.surveyManager = new SurveyManager(api, ui);

        this.session = null;          // last start-session response
        this.lastFit = null;          // last /fit-gp response
        this.lastSuggest = null;      // last /suggest-next response (with full candidates list)
        this.history = [];            // mirrored from server submit responses
        this.bestYCurve = [];         // running min(y_train) per submit
        this.participantsCache = null; // /bo/participants response

        this._wireEvents();
    }

    // ------------------------------------------------------------------
    // Public lifecycle
    // ------------------------------------------------------------------

    /**
     * Populate the participant select with the BO-eligible participants
     * (intersection of JSON pool + DB), called when the user enters BO mode.
     */
    async populateParticipantSelect() {
        try {
            const data = await this.api.boGetParticipants();
            this.participantsCache = data.participants;
            const select = document.getElementById('bo-participant-select');
            if (!select) return;
            select.innerHTML = '<option value="">Select a participant…</option>';
            for (const p of data.participants) {
                const opt = document.createElement('option');
                opt.value = p.mih;
                opt.textContent = p.db_participant_id == null
                    ? `${p.mih} (no DB record)`
                    : p.mih;
                select.appendChild(opt);
            }
        } catch (err) {
            this._notify(`Failed to load BO participants: ${err.message}`, 'danger');
        }
    }

    /**
     * Start a new BO session for the chosen participant + the current config.
     * Called when the user clicks "Start Session".
     */
    async start(participantMih) {
        if (!participantMih) {
            this._notify('Please select a participant first.', 'warning');
            return;
        }

        const cfg = this._readConfig();
        try {
            const resp = await this.api.boStartSession({
                participantMih,
                ...cfg,
            });
            this.session = resp;
            this.history = [];
            this.bestYCurve = [];
            this.lastFit = null;
            this.lastSuggest = null;

            const dash = document.getElementById('bo-dashboard');
            dash?.classList.remove('d-none');
            this._updateUserInfo();
            // Hide downstream cards until we re-fit
            document.getElementById('bo-gp-info-card')?.classList.add('d-none');
            document.getElementById('bo-candidates-card')?.classList.add('d-none');
            document.getElementById('bo-trial-card')?.classList.add('d-none');
            this._renderHistory();
            this._renderBestYCurve();
            this._notify(`BO session started for ${participantMih}`, 'success');
        } catch (err) {
            this._notify(`Failed to start session: ${err.message}`, 'danger');
        }
    }

    /**
     * Fit GP hyperparameters using the current pool. Reveals the GP info card
     * and unlocks suggesting next geometries.
     */
    async fitGP() {
        if (!this.session) {
            this._notify('Start a session first.', 'warning');
            return;
        }
        try {
            // Always send the *current* UI config so changes to mode / kernel /
            // objective / acquisition take effect without needing to restart.
            const cfg = this._readConfig();
            const resp = await this.api.boFitGP(this.session.session_id, cfg);
            this.lastFit = resp;
            // Mirror the server-side updates back onto the cached session so
            // the dashboard header + breakdown stay in sync.
            if (resp.config) this.session.config = resp.config;
            if (resp.pool_breakdown) this.session.pool_breakdown = resp.pool_breakdown;
            if (resp.n_pool_rows != null) this.session.n_pool_rows = resp.n_pool_rows;
            this._updateUserInfo();
            this._renderGPInfo(resp);
            // Auto-call suggest so the candidates table is populated.
            await this.suggestNext();
            this._notify('GP fitted', 'success');
        } catch (err) {
            this._notify(`Fit GP failed: ${err.message}`, 'danger');
        }
    }

    /**
     * Ask the server for the next-best candidate; populate candidates table
     * and trial-entry card.
     */
    async suggestNext() {
        if (!this.session) {
            this._notify('Start a session first.', 'warning');
            return;
        }
        if (!this.lastFit) {
            this._notify('Fit the GP first.', 'warning');
            return;
        }
        try {
            const resp = await this.api.boSuggestNext(this.session.session_id, { topK: 5 });
            this.lastSuggest = resp;
            this._renderCandidates(resp);
            this._populateTrialCard(resp.suggested);
        } catch (err) {
            this._notify(`Suggest-next failed: ${err.message}`, 'danger');
        }
    }

    /**
     * Submit the live-acquired observation (metabolic + survey) to the server.
     * On success the GP is automatically refit and a new candidate suggested.
     */
    async submitObservation() {
        if (!this.session) {
            this._notify('Start a session first.', 'warning');
            return;
        }
        const alpha = parseFloat(document.getElementById('bo-trial-alpha').value);
        const beta = parseFloat(document.getElementById('bo-trial-beta').value);
        const gamma = parseFloat(document.getElementById('bo-trial-gamma').value);
        const power = parseFloat(document.getElementById('bo-metabolic-power').value);
        const time = parseFloat(document.getElementById('bo-walking-time').value);
        const dist = parseFloat(document.getElementById('bo-distance').value);

        if ([alpha, beta, gamma, power, time, dist].some(v => Number.isNaN(v))) {
            this._notify('Fill in α, β, γ, metabolic power, time, distance.', 'warning');
            return;
        }
        if (time <= 0 || dist <= 0) {
            this._notify('Time and distance must be positive.', 'warning');
            return;
        }
        const validation = this.surveyManager.validateSurveyResponses(SURVEY_CONTAINER_ID);
        if (!validation.valid) {
            this._notify(`Survey invalid: ${validation.errors.join(', ')}`, 'warning');
            return;
        }
        const responses = this.surveyManager.collectSurveyResponses(SURVEY_CONTAINER_ID);
        const survey = {
            sus_q1: responses.sus_q1, sus_q2: responses.sus_q2, sus_q3: responses.sus_q3,
            sus_q4: responses.sus_q4, sus_q5: responses.sus_q5, sus_q6: responses.sus_q6,
            nrs_score: responses.nrs_score,
            tlx_mental_demand: responses.tlx_mental_demand,
            tlx_physical_demand: responses.tlx_physical_demand,
            tlx_performance: responses.tlx_performance,
            tlx_effort: responses.tlx_effort,
            tlx_frustration: responses.tlx_frustration,
        };

        try {
            const resp = await this.api.boSubmitObservation(this.session.session_id, {
                alpha, beta, gamma,
                metabolicPower: power,
                walkingTime: time,
                distance: dist,
                survey,
            });
            this.history = resp.history;
            this._renderHistory();
            this._renderBestYCurve();
            this.surveyManager.resetSurveyForms(SURVEY_CONTAINER_ID);
            // Reset trial inputs (geometry stays as-is until next suggest).
            document.getElementById('bo-metabolic-power').value = '';
            document.getElementById('bo-walking-time').value = '';
            document.getElementById('bo-distance').value = '';
            this._updateCotReadout();
            this._notify(`Observation saved (CoT=${resp.cot.toFixed(3)}). Refitting GP…`, 'success');
            // Refit + suggest next
            await this.fitGP();
            await this.refreshInsights();
        } catch (err) {
            this._notify(`Submit failed: ${err.message}`, 'danger');
        }
    }

    /**
     * Re-fetch the cross-participant insight figures and render them.
     */
    async refreshInsights() {
        try {
            const cfg = this._readConfig();
            const resp = await this.api.boGetInsights({
                objective: cfg.objective,
                wCot: cfg.wCot,
                wSurvey: cfg.wSurvey,
            });
            this._renderPlotly('bo-insights-outcome-per-geometry', resp.outcome_per_geometry);
            this._renderPlotly('bo-insights-correlation-matrix', resp.correlation_matrix);
        } catch (err) {
            this._notify(`Insights refresh failed: ${err.message}`, 'danger');
        }
    }

    // ------------------------------------------------------------------
    // Rendering
    // ------------------------------------------------------------------
    _renderGPInfo(resp) {
        const card = document.getElementById('bo-gp-info-card');
        card?.classList.remove('d-none');
        const set = (id, val) => {
            const el = document.getElementById(id);
            if (el) el.textContent = val;
        };
        const setHTML = (id, html) => {
            const el = document.getElementById(id);
            if (el) el.innerHTML = html;
        };
        set('bo-gp-kernel', resp.kernel);
        set('bo-gp-sigma-f', resp.theta.signal_variance.toFixed(4));

        // Lengthscale: scalar (isotropic) or per-feature dict (ARD).
        const ellsPerDim = resp.theta.lengthscales_per_dim;
        if (resp.theta.ard && ellsPerDim) {
            // Sort ascending so the most-relevant features appear first.
            const sorted = Object.entries(ellsPerDim)
                .sort((a, b) => a[1] - b[1]);
            const minL = sorted[0][1];
            const maxL = sorted[sorted.length - 1][1];
            const span = Math.max(maxL - minL, 1e-9);
            const rows = sorted.map(([feat, val]) => {
                const frac = (val - minL) / span;            // 0 = most relevant
                const widthPct = Math.max(4, (1 - frac) * 100);
                return `
                    <div class="d-flex align-items-center mb-1">
                        <code class="me-2" style="min-width: 8em">${feat}</code>
                        <div class="flex-grow-1 me-2"
                             style="background:#e9ecef; height:6px; border-radius:3px; overflow:hidden">
                            <div style="width:${widthPct}%; height:100%; background:#0d6efd"></div>
                        </div>
                        <span class="font-monospace" style="min-width: 5em; text-align:right">
                            ${val.toFixed(3)}
                        </span>
                    </div>`;
            }).join('');
            setHTML('bo-gp-lengthscale',
                `<div class="text-muted mb-1">ARD: smaller ℓ ⇒ feature is more relevant</div>${rows}`);
        } else if (resp.theta.lengthscale != null) {
            set('bo-gp-lengthscale', resp.theta.lengthscale.toFixed(4) + ' (isotropic)');
        } else {
            set('bo-gp-lengthscale', '—');
        }

        set('bo-gp-sigma-n', resp.theta.noise_variance.toFixed(4));
        set('bo-gp-nll', resp.train_nll.toFixed(4));
        set('bo-gp-n-train', String(resp.n_train));
        set('bo-gp-n-cand', String(resp.n_candidates));
        const bd = resp.pool_breakdown ?? this.session?.pool_breakdown;
        if (bd) {
            const txt = Object.entries(bd).map(([k, v]) => `${k}: ${v}`).join(', ');
            set('bo-gp-pool-breakdown', txt || 'empty');
        }
    }

    _updateUserInfo() {
        const info = document.getElementById('bo-user-info');
        if (!info || !this.session) return;
        const sid = this.session.session_id ?? '';
        const poolSize = this.session.n_pool_rows ?? '?';
        const nCand = this.session.n_candidates ?? '?';
        info.textContent =
            `Session ${String(sid).slice(0, 8)} · participant ${this.session.participant_mih}` +
            ` · pool size ${poolSize} · candidates ${nCand}`;
    }

    _renderCandidates(resp) {
        const card = document.getElementById('bo-candidates-card');
        card?.classList.remove('d-none');
        this._renderRationale(resp);
        const tbody = document.querySelector('#bo-candidates-table tbody');
        if (!tbody) return;

        const suggestedIdx = resp.suggested.idx;
        const rows = resp.candidates.slice().sort((a, b) => b.acquisition_value - a.acquisition_value);

        tbody.innerHTML = rows.map(c => {
            const isBest = c.idx === suggestedIdx;
            const cls = isBest ? 'table-warning fw-bold' : (c.excluded ? 'text-muted' : '');
            // Distinguish auto-excluded (already measured by participant) from
            // session-excluded (just submitted in this BO loop).
            let visitedMark = '';
            if (c.already_measured) {
                visitedMark = '<span title="Already measured by this participant — auto-excluded">📊</span>';
            } else if (c.excluded) {
                visitedMark = '<span title="Excluded by this BO session">✓</span>';
            }
            return `
                <tr class="${cls}" data-idx="${c.idx}" style="cursor:pointer">
                    <td>${c.idx}</td>
                    <td>${c.alpha}</td>
                    <td>${c.beta}</td>
                    <td>${c.gamma >= 0 ? '+' : ''}${c.gamma}</td>
                    <td>${c.posterior_mean.toFixed(3)}</td>
                    <td>${c.posterior_std.toFixed(3)}</td>
                    <td>${c.acquisition_value.toFixed(4)}</td>
                    <td>${visitedMark}</td>
                </tr>
            `;
        }).join('');

        // Click a row to load that candidate into the trial card.
        tbody.querySelectorAll('tr').forEach(tr => {
            tr.addEventListener('click', () => {
                const idx = parseInt(tr.dataset.idx, 10);
                const c = resp.candidates.find(x => x.idx === idx);
                if (c) this._populateTrialCard(c);
            });
        });
    }

    _populateTrialCard(suggested) {
        const card = document.getElementById('bo-trial-card');
        card?.classList.remove('d-none');
        document.getElementById('bo-trial-alpha').value = suggested.alpha;
        document.getElementById('bo-trial-beta').value = suggested.beta;
        document.getElementById('bo-trial-gamma').value = suggested.gamma;
        this._updateCotReadout();
    }

    _renderRationale(resp) {
        const box = document.getElementById('bo-suggest-rationale');
        if (!box) return;
        const r = resp.rationale;
        if (!r) { box.classList.add('d-none'); return; }
        box.classList.remove('d-none');

        const set = (id, val) => {
            const el = document.getElementById(id);
            if (el) el.textContent = val;
        };
        const fmt = (v, d = 4) =>
            v == null || !Number.isFinite(v) ? '—' : Number(v).toFixed(d);

        const acqLabel = r.acquisition_used === r.acquisition_requested
            ? r.acquisition_used
            : `${r.acquisition_used} (fallback from ${r.acquisition_requested})`;

        set('bo-rationale-summary', r.summary || '');
        set('bo-rationale-acq', acqLabel);
        set('bo-rationale-mu', fmt(r.posterior_mean, 3));
        set('bo-rationale-sigma', fmt(r.posterior_std, 3));
        set('bo-rationale-fbest',
            r.f_best == null ? 'no observations yet for this participant'
                              : fmt(r.f_best, 3));
        set('bo-rationale-rank',
            `${r.rank} of ${r.n_eligible} eligible candidates`);

        if (r.acquisition_used === 'TS') {
            set('bo-rationale-decomp',
                `f̃ ~ N(μ, Σ); chosen f̃ = ${fmt(r.sampled_value, 3)}`);
            set('bo-rationale-exploit', fmt(r.posterior_mean, 3) + ' (μ)');
            set('bo-rationale-explore', fmt(r.posterior_std, 3) + ' (σ)');
        } else if (r.acquisition_used === 'EI') {
            const decomp = r.f_best == null
                ? 'EI undefined → max σ'
                : `EI = (f*-μ)Φ(z) + σφ(z), z=${fmt(r.z, 3)}`;
            set('bo-rationale-decomp', decomp);
            set('bo-rationale-exploit', fmt(r.exploit_term));
            set('bo-rationale-explore', fmt(r.explore_term));
        } else if (r.acquisition_used === 'UCB') {
            set('bo-rationale-decomp',
                `LCB = μ - β·σ, β=${r.beta}, LCB=${fmt(r.lcb, 3)}`);
            set('bo-rationale-exploit', fmt(r.posterior_mean, 3) + ' (μ)');
            set('bo-rationale-explore',
                fmt(r.beta * r.posterior_std, 3) + ' (β·σ)');
        } else {
            set('bo-rationale-decomp', '—');
            set('bo-rationale-exploit', '—');
            set('bo-rationale-explore', '—');
        }
    }

    _renderHistory() {
        const tbody = document.querySelector('#bo-history-table tbody');
        if (!tbody) return;
        tbody.innerHTML = this.history.map(h => `
            <tr>
                <td>${h.iteration}</td>
                <td>${h.alpha}</td>
                <td>${h.beta}</td>
                <td>${h.gamma >= 0 ? '+' : ''}${h.gamma}</td>
                <td>${h.cot != null ? h.cot.toFixed(3) : '—'}</td>
                <td>${h.sus_score != null ? h.sus_score.toFixed(1) : '—'}</td>
                <td>${h.nrs_score ?? '—'}</td>
                <td>${h.tlx_score != null ? h.tlx_score.toFixed(1) : '—'}</td>
            </tr>
        `).join('');
    }

    _renderBestYCurve() {
        const div = document.getElementById('bo-best-y-curve');
        if (!div || typeof Plotly === 'undefined') return;
        if (this.history.length === 0) {
            Plotly.purge(div);
            return;
        }
        // Best-CoT curve as a simple proxy; full y_objective best lives server-side.
        const cots = this.history.map(h => h.cot).filter(v => v != null);
        const bestSoFar = [];
        let cur = Infinity;
        for (const c of cots) {
            cur = Math.min(cur, c);
            bestSoFar.push(cur);
        }
        Plotly.newPlot(div, [{
            x: bestSoFar.map((_, i) => i + 1),
            y: bestSoFar,
            mode: 'lines+markers',
            name: 'best CoT so far',
        }], {
            title: 'Best-so-far CoT',
            margin: { l: 50, r: 20, t: 40, b: 40 },
            xaxis: { title: 'iteration' },
            yaxis: { title: 'CoT (J/kg/m)' },
        }, { displayModeBar: false, responsive: true });
    }

    _renderPlotly(divId, figJsonObj) {
        const div = document.getElementById(divId);
        if (!div || typeof Plotly === 'undefined' || !figJsonObj) return;
        Plotly.react(div, figJsonObj.data || [], figJsonObj.layout || {}, { responsive: true });
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------
    _readConfig() {
        const mode = document.querySelector('input[name="bo-mode"]:checked')?.value || 'pool_with_self';
        const objective = document.querySelector('input[name="bo-objective"]:checked')?.value || 'combined';
        const wCot = parseFloat(document.getElementById('bo-w-cot')?.value) || 1.0;
        const wSurvey = parseFloat(document.getElementById('bo-w-survey')?.value) || 1.0;
        const kernel = document.getElementById('bo-kernel-select')?.value || 'matern52';
        const acquisition = document.getElementById('bo-acquisition-select')?.value || 'TS';
        return { mode, objective, wCot, wSurvey, kernel, acquisition };
    }

    _updateCotReadout() {
        const power = parseFloat(document.getElementById('bo-metabolic-power')?.value);
        const time = parseFloat(document.getElementById('bo-walking-time')?.value);
        const dist = parseFloat(document.getElementById('bo-distance')?.value);
        const velEl = document.getElementById('bo-velocity-readout');
        const cotEl = document.getElementById('bo-cot-readout');
        if (!velEl || !cotEl) return;
        if ([power, time, dist].some(v => Number.isNaN(v) || v <= 0)) {
            velEl.textContent = '—';
            cotEl.textContent = '—';
            return;
        }
        const v = dist / time;
        const cot = power / v;
        velEl.textContent = v.toFixed(3);
        cotEl.textContent = cot.toFixed(3);
    }

    _wireEvents() {
        const bind = (id, ev, fn) => {
            const el = document.getElementById(id);
            if (el) el.addEventListener(ev, fn);
        };
        bind('bo-fit-gp-btn', 'click', () => this.fitGP());
        bind('bo-suggest-next-btn', 'click', () => this.suggestNext());
        bind('bo-submit-observation-btn', 'click', () => this.submitObservation());
        bind('bo-refresh-insights-btn', 'click', () => this.refreshInsights());

        // Live update of CoT readout
        ['bo-metabolic-power', 'bo-walking-time', 'bo-distance'].forEach(id =>
            bind(id, 'input', () => this._updateCotReadout())
        );

        // Show / hide the weight sliders depending on the chosen objective.
        document.querySelectorAll('input[name="bo-objective"]').forEach(r => {
            r.addEventListener('change', () => {
                const sliders = document.getElementById('bo-weight-sliders');
                const isCombined = document.querySelector('input[name="bo-objective"]:checked')?.value === 'combined';
                if (sliders) sliders.classList.toggle('d-none', !isCombined);
            });
        });
    }

    _notify(msg, level = 'info') {
        if (this.ui?.showNotification) {
            this.ui.showNotification(msg, level);
        } else {
            console.log(`[BOMode/${level}] ${msg}`);
        }
    }
}

export { BOMode };
window.BOMode = BOMode;
