/**
 * UI Renderer - Handles all rendering logic for systematic mode
 */

export class UIRenderer {
    constructor(uiComponents) {
        this.ui = uiComponents;
    }

    /**
     * Render trials table for a participant
     * @param {Object} participant - Current participant
     * @param {Array} trials - All trials
     * @param {HTMLElement} tableBody - Table body element
     */
    renderTrialsTable(participant, trials, tableBody) {
        if (!tableBody || !participant) {
            return;
        }

        const participantTrials = trials?.filter(t =>
            t.participant_id === participant.id &&
            this._isGridSearchTrial(t)
        ) || [];

        tableBody.innerHTML = '';

        if (participantTrials.length === 0) {
            tableBody.innerHTML = `
                <tr><td colspan="20" class="text-center text-muted">
                    No trials recorded yet for this participant.
                </td></tr>
            `;
            return;
        }

        // Sort trials: Control first, then by G-number
        participantTrials.sort((a, b) => {
            const aIsControl = a.geometry_name === 'Control';
            const bIsControl = b.geometry_name === 'Control';
            if (aIsControl && !bIsControl) return -1;
            if (!aIsControl && bIsControl) return 1;

            const aGeomName = a.geometry_name || '';
            const bGeomName = b.geometry_name || '';

            if (aGeomName.startsWith('G') && bGeomName.startsWith('G')) {
                const aNum = parseInt(aGeomName.substring(1)) || 0;
                const bNum = parseInt(bGeomName.substring(1)) || 0;
                return aNum - bNum;
            }

            return aGeomName.localeCompare(bGeomName);
        });

        participantTrials.forEach((trial, index) => {
            const row = this._createTrialRow(trial, index, participant);
            tableBody.appendChild(row);
        });
    }

    /**
     * Create a trial table row
     * @private
     */
    _createTrialRow(trial, index, participant) {
        const row = document.createElement('tr');
        row.dataset.trialId = trial.id;
        row.classList.add('clickable-trial-row');

        const trialDate = new Date(trial.timestamp);
        const today = new Date();
        const isToday = trialDate.toDateString() === today.toDateString();

        // Calculate trial number
        let trialNumber;
        if (trial.geometry_name === 'Control') {
            trialNumber = 'Control';
        } else {
            trialNumber = index + 1;
        }

        // Get participant characteristics
        const characteristics = participant.characteristics || {};

        row.innerHTML = `
            <td>${trialNumber}</td>
            <td>${trial.geometry_name || 'Unknown'}</td>
            <td>${trial.alpha ?? '-'}°</td>
            <td>${trial.beta ?? '-'}°</td>
            <td>${trial.gamma ?? '-'}°</td>
            <td>${isToday ? 'Today' : trialDate.toLocaleDateString()}</td>
            <td>${trialDate.toLocaleTimeString()}</td>
            <td>${characteristics.age || '-'}</td>
            <td>${characteristics.sex || '-'}</td>
            <td>${characteristics.height || '-'}</td>
            <td>${characteristics.weight || '-'}</td>
            <td>${characteristics.forearm_length || '-'}</td>
            <td>${characteristics.activity_level || '-'}</td>
            <td>${characteristics.previous_crutch_experience !== undefined ?
                (characteristics.previous_crutch_experience ? 'Yes' : 'No') : '-'}</td>
            <td>${Number(trial.processed_features?.step_count ?? NaN) || '-'}</td>
            <td>${trial.instability_loss !== undefined ?
                this.ui.formatNumber(trial.instability_loss, 4) : '-'}</td>
            <td>${trial.survey_responses?.sus_score !== undefined ?
                this.ui.formatNumber(trial.survey_responses.sus_score) : '-'}</td>
            <td>${trial.survey_responses?.nrs_score !== undefined ?
                trial.survey_responses.nrs_score : '-'}</td>
            <td>${trial.survey_responses?.tlx_score !== undefined ?
                this.ui.formatNumber(trial.survey_responses.tlx_score) : '-'}</td>
            <td>${trial.metabolic_cost !== undefined ?
                this.ui.formatNumber(trial.metabolic_cost) : '-'}</td>
            <td>
                <button class="btn btn-sm btn-outline-primary" 
                        onclick="window.app.modules.systematic.viewTrialDetails(${trial.id})">
                    <i class="fas fa-eye"></i>
                </button>
                <button class="btn btn-sm btn-outline-danger" 
                        onclick="window.app.modules.systematic.deleteTrial(${trial.id})">
                    <i class="fas fa-trash"></i>
                </button>
            </td>
        `;

        return row;
    }

    /**
     * Render geometry grid
     * @param {Object} participant - Current participant
     * @param {Array} allGeometries - All available geometries
     * @param {Array} trials - All trials
     * @param {HTMLElement} gridElement - Grid container element
     */
    renderGeometryGrid(participant, allGeometries, trials, gridElement) {
        if (!gridElement) return;

        gridElement.innerHTML = '';

        if (!participant) {
            gridElement.innerHTML = `<div class="text-center text-muted">No participant selected.</div>`;
            return;
        }

        if (!allGeometries || allGeometries.length === 0) {
            gridElement.innerHTML = `<div class="text-center text-muted">No geometries available!</div>`;
            return;
        }

        // Get completed trials for this participant
        const completedTrials = trials?.filter(t => t.participant_id === participant.id) || [];
        const completedGeometryIds = new Set(completedTrials.map(t => t.geometry_id));

        // Define the specific values we want to show
        const allowedGammas = [-9, 0, 9];
        const allowedAlphas = [85, 105];
        const allowedBetas = [95, 125];

        // Group geometries
        const gammaGroups = {};
        const controlGroups = {};

        allGeometries.forEach(g => {
            if (g.name.startsWith('Control')) {
                if (!controlGroups[g.gamma]) {
                    controlGroups[g.gamma] = [];
                }
                controlGroups[g.gamma].push(g);
            } else {
                if (allowedGammas.includes(g.gamma) &&
                    allowedAlphas.includes(g.alpha) &&
                    allowedBetas.includes(g.beta)) {
                    if (!gammaGroups[g.gamma]) {
                        gammaGroups[g.gamma] = [];
                    }
                    gammaGroups[g.gamma].push(g);
                }
            }
        });

        // Count how many control trials have been completed
        let controlGeometry = null;
        if (Object.keys(controlGroups).length > 0) {
            controlGeometry = Object.values(controlGroups)[0][0];
        }
        const controlTrialsCompleted = controlGeometry ?
            completedTrials.filter(t => t.geometry_id === controlGeometry.id).length : 0;

        // Add control trial at the top (Baseline 1)
        // Show count of completed control trials (all control trials count for both baselines)
        if (controlGeometry) {
            this._renderBaselineControl(gridElement, controlGeometry, controlTrialsCompleted, 'Baseline 1');
        }

        // Render gamma grids
        const gammaOrder = [-9, 0, 9];
        gammaOrder.forEach(gamma => {
            if (gammaGroups[gamma]) {
                this._renderGammaGrid(gridElement, gamma, gammaGroups[gamma], completedTrials, allowedAlphas, allowedBetas);
            }
        });

        // Add control trial at the bottom (Baseline 2)
        // Show the same count since both baselines use the same geometry
        if (controlGeometry) {
            this._renderBaselineControl(gridElement, controlGeometry, controlTrialsCompleted, 'Baseline 2', true);
        }
    }

    /**
     * Render baseline control button
     * @private
     * @param {HTMLElement} container - Container element
     * @param {Object} geometry - Geometry object
     * @param {number} completionCount - Number of times this baseline has been completed
     * @param {string} label - Label for the baseline
     * @param {boolean} isBottom - Whether this is the bottom baseline
     */
    _renderBaselineControl(container, geometry, completionCount, label, isBottom = false) {
        const controlContainer = document.createElement('div');
        controlContainer.className = 'mb-4';

        // Generate completion indicators (green checkmarks)
        const completionIndicators = completionCount > 0 ?
            `<span class="ms-2">${'<i class="fas fa-check-circle text-success"></i> '.repeat(completionCount)}</span>` : '';

        const controlButtonHtml = `
            <button class="btn btn-sm ${completionCount > 0 ? 'btn-outline-success' : 'btn-warning'}" data-geom-id="${geometry.id}">
                ${label.toUpperCase()}<br><small>Data Collection</small>${completionIndicators}
            </button>
        `;

        if (isBottom) {
            controlContainer.innerHTML = `
                <hr class="my-4 border-2 border-secondary">
                <h5 class="mb-3 text-warning">${label} (α:95°, β:95°, γ:0°)</h5>
                <div class="mb-3">${controlButtonHtml}</div>
            `;
        } else {
            controlContainer.innerHTML = `
                <h5 class="mb-3 text-warning">${label} (α:95°, β:95°, γ:0°)</h5>
                <div class="mb-3">${controlButtonHtml}</div>
                <hr class="my-4 border-2 border-secondary">
            `;
        }

        container.appendChild(controlContainer);
    }

    /**
     * Render gamma grid table
     * @private
     */
    _renderGammaGrid(container, gamma, geometries, completedTrials, alphaValues, betaValues) {
        const tableContainer = document.createElement('div');
        tableContainer.className = 'mb-4';
        tableContainer.innerHTML = `<h5 class="mb-3">Gamma (γ): ${gamma}°</h5>`;

        const tableDiv = document.createElement('div');
        tableDiv.className = 'table-responsive';

        tableDiv.innerHTML = `
            <table class="table table-sm table-bordered grid-table" data-gamma="${gamma}">
                <thead class="table-light">
                    <tr>
                        <th></th>
                        <th class="text-center" colspan="${betaValues.length}">Beta (β)</th>
                    </tr>
                    <tr>
                        <th>Alpha (α)</th>
                        ${betaValues.map(beta => `<th class="text-center">${beta}°</th>`).join('')}
                    </tr>
                </thead>
                <tbody>
                    ${alphaValues.map(alpha => `
                        <tr>
                            <th class="table-light text-center">${alpha}°</th>
                            ${betaValues.map(beta => {
            const geom = geometries.find(g => g.alpha === alpha && g.beta === beta);
            if (geom) {
                // Count how many times this geometry has been completed
                const completionCount = completedTrials.filter(t => t.geometry_id === geom.id).length;
                const completionIndicators = completionCount > 0 ?
                    `<br><span>${'<i class="fas fa-check-circle text-success"></i> '.repeat(completionCount)}</span>` : '';

                return `<td class="text-center">
                                        <button class="btn btn-sm ${completionCount > 0 ? 'btn-outline-success' : 'btn-outline-primary'}" data-geom-id="${geom.id}">
                                            Start Trial<br><small>${geom.name}</small>${completionIndicators}
                                        </button>
                                    </td>`;
            } else {
                return `<td class="text-center text-muted">-</td>`;
            }
        }).join('')}
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;
        tableContainer.appendChild(tableDiv);
        container.appendChild(tableContainer);
    }

    /**
     * Render instability plot
     * @param {Array} plotData - Plot data points
     * @param {HTMLElement} plotDiv - Plot container element
     */
    renderInstabilityPlot(plotData, plotDiv) {
        if (!plotDiv) return;

        if (!plotData || plotData.length < 1) {
            plotDiv.innerHTML = `<div class="text-center text-muted pt-5">No completed trials with cumulative score data for this participant.</div>`;
            return;
        }

        try {
            const trace = {
                x: plotData.map(d => d.alpha),
                y: plotData.map(d => d.beta),
                z: plotData.map(d => d.gamma),
                mode: 'markers',
                type: 'scatter3d',
                marker: {
                    color: plotData.map(d => d.cumulative_score),
                    colorscale: 'Viridis',
                    colorbar: {
                        title: 'Cumulative Score'
                    },
                    size: 8,
                    symbol: plotData.map(d => d.geometry_name.includes('Control') ? 'cross' : 'diamond')
                },
                text: plotData.map(d => {
                    const breakdown = `Instability: ${(d.instability_loss || 0).toFixed(4)}<br>` +
                        `SUS: ${d.sus_score || 0}/100<br>` +
                        `NRS: ${d.nrs_score || 0}/10<br>` +
                        `TLX: ${d.tlx_score || 0}/100<br>` +
                        `Cumulative: ${(d.cumulative_score || 0).toFixed(4)}`;
                    return `Trial: ${d.geometry_name}<br>${breakdown}`;
                }),
                hoverinfo: 'text'
            };

            const layout = {
                title: 'Cumulative Score (Instability + Normalized Surveys) vs. Crutch Geometry',
                scene: {
                    xaxis: { title: 'Alpha (α)', range: [70, 120] },
                    yaxis: { title: 'Beta (β)', range: [90, 140] },
                    zaxis: { title: 'Gamma (γ)', range: [-12, 12] }
                },
                margin: { l: 0, r: 0, b: 0, t: 40 }
            };

            if (window.Plotly) {
                window.Plotly.newPlot(plotDiv, [trace], layout, { responsive: true });
            } else {
                plotDiv.innerHTML = `<div class="text-center text-muted pt-5">3D plot data available - Plotly.js loading...</div>`;
            }
        } catch (error) {
            console.error('Error rendering instability plot:', error);
            plotDiv.innerHTML = `<div class="text-center text-muted pt-5">Error rendering 3D plot.</div>`;
        }
    }

    /**
     * Render force plot with step detection
     * @param {Object} plots - Plot data
     * @param {HTMLElement} plotDiv - Plot container element
     * @param {Function} onPlotClick - Callback for plot clicks
     */
    renderForcePlot(plots, plotDiv, onPlotClick) {
        if (!plotDiv) {
            console.log('force-plot-div element not found');
            return;
        }

        if (!plots || !plots.time || !plots.force) {
            console.log('Missing plot data. Available keys:', Object.keys(plots || {}));
            return;
        }

        try {
            // Create force trace
            const forceTrace = {
                x: plots.time,
                y: plots.force,
                type: 'scatter',
                mode: 'lines',
                name: 'Force',
                yaxis: 'y',
                line: { color: 'rgb(31, 119, 180)', width: 2 }
            };

            // Create force derivative trace (if available)
            const traces = [forceTrace];
            if (plots.force_derivative && plots.force_derivative.length > 0) {
                const forceDerivTrace = {
                    x: plots.time,
                    y: plots.force_derivative,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Force Derivative',
                    yaxis: 'y2',
                    line: { color: 'rgb(255, 127, 14)', width: 2, dash: 'dash' }
                };
                traces.push(forceDerivTrace);
            }

            // Create step markers
            const stepForces = plots.step_times.map(stepTime => {
                let closestIdx = 0;
                let minDiff = Math.abs(plots.time[0] - stepTime);
                for (let i = 1; i < plots.time.length; i++) {
                    const diff = Math.abs(plots.time[i] - stepTime);
                    if (diff < minDiff) {
                        minDiff = diff;
                        closestIdx = i;
                    }
                }
                return plots.force[closestIdx];
            });

            const stepTrace = {
                x: plots.step_times,
                y: stepForces,
                type: 'scatter',
                mode: 'markers',
                name: 'Detected Steps',
                yaxis: 'y',
                marker: {
                    color: 'red',
                    size: 12,
                    symbol: 'circle',
                    line: { color: 'darkred', width: 2 }
                }
            };
            traces.push(stepTrace);

            const layout = {
                title: { text: 'Force & Derivative with Step Detection', font: { size: 16 } },
                xaxis: { title: 'Time (s)', showgrid: true, zeroline: false },
                yaxis: {
                    title: 'Force (N)',
                    showgrid: true,
                    zeroline: true,
                    side: 'left'
                },
                yaxis2: {
                    title: 'Force Derivative (N/s)',
                    overlaying: 'y',
                    side: 'right',
                    showgrid: false,
                    zeroline: true
                },
                showlegend: true,
                hovermode: 'closest',
                margin: { t: 50, r: 80, b: 50, l: 50 }
            };

            const config = {
                responsive: true,
                displayModeBar: true,
                displaylogo: false
            };

            Plotly.newPlot(plotDiv, traces, layout, config);
            console.log('Plot rendered successfully');

            // Attach click listener
            plotDiv.removeAllListeners?.('plotly_click');
            if (onPlotClick) {
                plotDiv.on('plotly_click', (data) => onPlotClick(data, plots));
            }

        } catch (error) {
            console.error('Error rendering plot:', error);
            plotDiv.innerHTML = '<div class="alert alert-danger">Failed to render plot</div>';
        }
    }

    /**
     * Render steps list
     * @param {Array} steps - Step times
     * @param {HTMLElement} listElement - List container element
     * @param {Function} onRemoveStep - Callback for removing steps
     */
    renderStepsList(steps, listElement) {
        if (!listElement) return;

        listElement.innerHTML = '';
        steps.forEach((stepTime, index) => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${index + 1}</td>
                <td>${this.ui.formatNumber(stepTime, 2)}s</td>
                <td>
                    <button class="btn btn-sm btn-outline-danger" 
                            onclick="window.app.modules.systematic.removeStep(${index})">
                        <i class="fas fa-trash"></i>
                    </button>
                </td>
            `;
            listElement.appendChild(row);
        });

        // Update step count displays
        const stepCountElements = [
            document.getElementById('step-count'),
            document.getElementById('step-count-display')
        ];
        stepCountElements.forEach(el => {
            if (el) el.textContent = steps.length;
        });
    }

    /**
     * Display participant details
     * @param {Object} participant - Participant object
     * @param {HTMLElement} tableBody - Table body element
     * @param {HTMLElement} footer - Footer element (optional)
     */
    displayParticipantDetails(participant, tableBody, footer) {
        if (!tableBody) return;

        tableBody.innerHTML = '';

        if (!participant || !participant.characteristics) {
            if (footer) footer.classList.add('d-none');
            return;
        }

        const characteristics = participant.characteristics;

        const detailsMap = {
            'Age': characteristics.age,
            'Sex': characteristics.sex,
            'Height (cm)': characteristics.height,
            'Weight (kg)': characteristics.weight,
            'Forearm Length (cm)': characteristics.forearm_length,
            'Activity Level': characteristics.activity_level,
            'Previous Crutch Experience': characteristics.previous_crutch_experience !== undefined ?
                (characteristics.previous_crutch_experience ? 'Yes' : 'No') : undefined
        };

        for (const [key, value] of Object.entries(detailsMap)) {
            const row = tableBody.insertRow();
            row.innerHTML = `<th class="text-muted">${key}</th><td>${value ?? 'N/A'}</td>`;
        }

        if (footer) footer.classList.remove('d-none');
    }

    /**
     * Check if trial is a grid search trial
     * @private
     */
    _isGridSearchTrial(trial) {
        return trial.source === 'grid_search' ||
            trial.source === 'GRID_SEARCH' ||
            trial.source === 'TrialSource.GRID_SEARCH';
    }
}

