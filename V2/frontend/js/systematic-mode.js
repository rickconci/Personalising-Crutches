/**
 * Systematic Mode Manager - Handles grid search experiment workflows
 * Manages systematic data collection, trial running, and visualization
 */

class SystematicMode {
    constructor(apiClient, uiComponents, deviceManager) {
        this.api = apiClient;
        this.ui = uiComponents;
        this.device = deviceManager;

        // Trial state
        this.trialState = {
            timer: null,
            startTime: null,
            elapsed: 0,
            running: false,
            metrics: null,
            steps: [],
            rawData: null,
            currentParticipant: null,
            currentGeometry: null
        };

        this.elements = this._getElements();
        this._setupEventListeners();
        this._setupDeviceCallbacks();
    }

    /**
     * Initialize systematic mode for a participant
     * @param {Object} participant - Participant data
     */
    async initialize(participant) {
        this.trialState.currentParticipant = participant;
        await this._loadParticipantData(participant.id);
        this._updateUI();
    }

    /**
     * Start a new trial
     * @param {Object} geometry - Geometry configuration
     */
    async startTrial(geometry) {
        if (this.trialState.running) {
            this.ui.showNotification('A trial is already running', 'warning');
            return;
        }

        if (!this.device.getConnectionStatus()) {
            this.ui.showNotification('Please connect to device first', 'warning');
            return;
        }

        try {
            this.trialState.currentGeometry = geometry;
            this.trialState.running = true;
            this.trialState.startTime = Date.now();

            // Start device data collection
            await this.device.startDataCollection();

            // Start UI timer
            this._startTimer();

            // Update UI
            this._updateTrialUI(true);

            this.ui.showNotification('Trial started', 'success');

        } catch (error) {
            this.ui.showNotification(`Failed to start trial: ${error.message}`, 'danger');
            this._resetTrialState();
        }
    }

    /**
     * Stop the current trial
     */
    async stopTrial() {
        if (!this.trialState.running) {
            return;
        }

        try {
            // Stop device data collection
            await this.device.stopDataCollection();

            // Stop timer
            this._stopTimer();

            // Get collected data
            const rawData = this.device.getDataBuffer();
            this.trialState.rawData = rawData;

            // Update UI
            this.trialState.running = false;
            this._updateTrialUI(false);

            // Process trial data
            await this._processTrialData(rawData);

            this.ui.showNotification('Trial stopped and data processed', 'success');

        } catch (error) {
            this.ui.showNotification(`Failed to stop trial: ${error.message}`, 'danger');
            this._resetTrialState();
        }
    }

    /**
     * Upload and process a data file
     * @param {File} file - Data file to upload
     */
    async uploadAndProcessFile(file) {
        if (!this.trialState.currentParticipant) {
            this.ui.showNotification('Please select a participant first', 'warning');
            return;
        }

        try {
            this.ui.toggleLoading(this.elements.uploadBtn, true, 'Processing...');

            // Create trial record first
            const trialData = {
                participant_id: this.trialState.currentParticipant.id,
                alpha: this.trialState.currentGeometry?.alpha || 95,
                beta: this.trialState.currentGeometry?.beta || 125,
                gamma: this.trialState.currentGeometry?.gamma || 0,
                delta: this.trialState.currentGeometry?.delta || 0,
                source: 'grid_search',
                survey_responses: {
                    effort_survey_answer: 3,
                    pain_survey_answer: 3,
                    stability_survey_answer: 3
                }
            };

            const newTrial = await this.api.createTrial(trialData);

            // Upload and process file
            const uploadResult = await this.api.uploadFile(file, newTrial.id);
            const processResult = await this.api.processFile(uploadResult.id);

            // Update trial with results
            const updateData = {
                processed_features: processResult.processing_results.data_info,
                steps: processResult.processing_results.step_detection.step_times,
                step_variance: processResult.processing_results.gait_metrics.step_variance,
                y_change: processResult.processing_results.gait_metrics.y_change,
                y_total: processResult.processing_results.gait_metrics.y_total,
                total_combined_loss: processResult.processing_results.gait_metrics.step_variance || 0
            };

            await this.api.updateTrial(newTrial.id, updateData);

            // Display results
            this._displayTrialResults(processResult.processing_results);

            // Refresh participant data
            await this._loadParticipantData(this.trialState.currentParticipant.id);

            this.ui.showNotification('File processed successfully', 'success');

        } catch (error) {
            this.ui.showNotification(`Failed to process file: ${error.message}`, 'danger');
        } finally {
            this.ui.toggleLoading(this.elements.uploadBtn, false);
        }
    }

    /**
     * Delete a trial
     * @param {number} trialId - Trial ID to delete
     */
    async deleteTrial(trialId) {
        const confirmed = await this.ui.showConfirmDialog(
            'Are you sure you want to delete this trial?',
            'Delete Trial'
        );

        if (!confirmed) return;

        try {
            await this.api.deleteTrial(trialId);
            await this._loadParticipantData(this.trialState.currentParticipant.id);
            this.ui.showNotification('Trial deleted successfully', 'success');
        } catch (error) {
            this.ui.showNotification(`Failed to delete trial: ${error.message}`, 'danger');
        }
    }

    /**
     * View trial details
     * @param {number} trialId - Trial ID to view
     */
    viewTrialDetails(trialId) {
        const trial = this.trials?.find(t => t.id === trialId);
        if (trial) {
            // Show trial details in a modal or dedicated view
            this.ui.showNotification(`Viewing trial ${trialId}`, 'info');
            console.log('Trial details:', trial);
        }
    }

    /**
     * Handle geometry button click
     * @param {string} geomId - Geometry ID
     * @private
     */
    _handleGeometryButtonClick(geomId) {
        const geometry = this.allGeometries?.find(g => g.id == geomId);
        if (!geometry) {
            this.ui.showNotification('Geometry not found', 'danger');
            return;
        }

        // Show the trial runner
        const trialRunnerCol = document.getElementById('trial-runner-col');
        const trialRunnerTitle = document.getElementById('trial-runner-title');

        if (trialRunnerCol) {
            trialRunnerCol.classList.remove('d-none');
        }

        if (trialRunnerTitle) {
            trialRunnerTitle.textContent = `Run Trial: ${geometry.name}`;
        }

        // Store the selected geometry for the trial
        this.trialState.currentGeometry = geometry;

        this.ui.showNotification(`Selected geometry: ${geometry.name}`, 'info');
    }

    /**
     * Remove a step from the current trial
     * @param {number} stepIndex - Index of step to remove
     */
    removeStep(stepIndex) {
        if (stepIndex >= 0 && stepIndex < this.trialState.steps.length) {
            this.trialState.steps.splice(stepIndex, 1);
            this._updateStepsList(this.trialState.steps);
            this.ui.showNotification(`Step ${stepIndex + 1} removed`, 'info');
        }
    }

    /**
     * Get DOM elements
     * @private
     */
    _getElements() {
        return {
            startStopBtn: document.getElementById('start-stop-btn'),
            connectDeviceBtn: document.getElementById('connect-device-btn'),
            uploadBtn: document.getElementById('upload-data-btn'),
            stopwatch: document.getElementById('stopwatch'),
            stepCount: document.getElementById('step-count'),
            instabilityLoss: document.getElementById('instability-loss-value'),
            stepList: document.getElementById('step-list'),
            trialsTableBody: document.querySelector('#participant-trials-table tbody'),
            geometryGrid: document.getElementById('grid-search-tables'),
            participantInfo: document.getElementById('participant-info'),
            fileUploadInput: document.getElementById('file-upload-input')
        };
    }

    /**
     * Setup event listeners
     * @private
     */
    _setupEventListeners() {
        // File upload
        this.elements.fileUploadInput?.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                this.uploadAndProcessFile(file);
            }
        });

        // Event delegation for dynamically generated geometry grid buttons
        document.addEventListener('click', (e) => {
            // Handle grid search table buttons
            if (e.target.closest('#grid-search-tables') && e.target.tagName === 'BUTTON' && e.target.dataset.geomId) {
                this._handleGeometryButtonClick(e.target.dataset.geomId);
            }
        });
    }

    /**
     * Setup device callbacks
     * @private
     */
    _setupDeviceCallbacks() {
        this.device.setConnectionChangeCallback((status, message) => {
            // Update device status in UI
            const statusElement = document.getElementById('device-status');
            if (statusElement) {
                statusElement.textContent = `Status: ${message}`;
                statusElement.className = `alert ${status === 'connected' ? 'alert-success' :
                    status === 'connecting' || status === 'searching' ? 'alert-info' :
                        'alert-secondary'
                    }`;
            }

            // Enable/disable start button based on connection
            if (this.elements.startStopBtn) {
                this.elements.startStopBtn.disabled = status !== 'connected';
            }
        });

        this.device.setErrorCallback((error) => {
            this.ui.showNotification(error, 'danger');
        });
    }

    /**
     * Load participant data
     * @private
     */
    async _loadParticipantData(participantId) {
        try {
            console.log('Loading participant data for ID:', participantId);
            console.log('Current participant before API call:', this.trialState.currentParticipant);

            // Make separate API calls to get all the data we need
            // (The single endpoint doesn't return the expected comprehensive structure)
            const [participantData, allTrials, allGeometries] = await Promise.all([
                this.api.request(`/experiments/participants/${participantId}`),
                this.api.getTrials(),
                this.api.getGeometries()
            ]);

            console.log('Participant API response:', participantData);
            console.log('Trials loaded:', allTrials.length);
            console.log('Geometries loaded:', allGeometries.length);

            // Update participant info - the API returns the participant directly, not wrapped
            this.trialState.currentParticipant = participantData;
            console.log('Current participant after API call:', this.trialState.currentParticipant);

            // Store trials and geometries
            this.trials = allTrials;
            this.allGeometries = allGeometries;

            // For now, create empty instability plot data (we'll need to check if there's a separate endpoint for this)
            this.instabilityPlotData = [];

            // Filter trials with instability data for the plot
            const participantTrialsWithInstability = allTrials.filter(t =>
                t.participant_id === participantId &&
                t.processed_features?.instability_loss !== undefined
            );

            if (participantTrialsWithInstability.length > 0) {
                // Convert trial data to plot format
                this.instabilityPlotData = participantTrialsWithInstability.map(trial => ({
                    alpha: trial.alpha,
                    beta: trial.beta,
                    gamma: trial.gamma,
                    instability_loss: trial.processed_features.instability_loss,
                    geometry_name: trial.geometry_name || `Trial ${trial.id}`
                }));
            }

            console.log('Instability plot data:', this.instabilityPlotData);
            console.log('About to render UI with participant:', this.trialState.currentParticipant);

            // Update UI
            this._renderTrialsTable();
            this._renderGeometryGrid();
            this._renderInstabilityPlot();
            this._displayParticipantDetails();

        } catch (error) {
            console.error('Error loading participant data:', error);
            this.ui.showNotification(`Failed to load participant data: ${error.message}`, 'danger');
        }
    }

    /**
     * Render trials table
     * @private
     */
    _renderTrialsTable() {
        if (!this.elements.trialsTableBody || !this.trialState.currentParticipant) {
            return;
        }

        const participantTrials = this.trials?.filter(t =>
            t.participant_id === this.trialState.currentParticipant.id &&
            this._isGridSearchTrial(t)
        ) || [];

        this.elements.trialsTableBody.innerHTML = '';

        if (participantTrials.length === 0) {
            this.elements.trialsTableBody.innerHTML = `
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
            const row = this._createTrialRow(trial, index);
            this.elements.trialsTableBody.appendChild(row);
        });
    }

    /**
     * Create a trial table row
     * @private
     */
    _createTrialRow(trial, index) {
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
        const characteristics = this.trialState.currentParticipant.characteristics || {};

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
            <td>${trial.processed_features?.instability_loss !== undefined ?
                this.ui.formatNumber(trial.processed_features.instability_loss, 4) : '-'}</td>
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
     * @private
     */
    _renderGeometryGrid() {
        console.log('_renderGeometryGrid called');
        console.log('geometryGrid element:', this.elements.geometryGrid);
        console.log('currentParticipant:', this.trialState.currentParticipant);

        if (!this.elements.geometryGrid) return;

        this.elements.geometryGrid.innerHTML = '';

        if (!this.trialState.currentParticipant) {
            console.log('No current participant, showing "No participant selected" message');
            this.elements.geometryGrid.innerHTML = `<div class="text-center text-muted">No participant selected.</div>`;
            return;
        }

        if (!this.allGeometries || this.allGeometries.length === 0) {
            this.elements.geometryGrid.innerHTML = `<div class="text-center text-muted">No geometries available!</div>`;
            return;
        }

        // Get completed trials for this participant (using geometry_id, not geometry_name)
        const completedTrials = this.trials?.filter(t => t.participant_id === this.trialState.currentParticipant.id) || [];
        const completedGeometryIds = new Set(completedTrials.map(t => t.geometry_id));

        // Group geometries by gamma value, separating control from regular geometries
        const gammaGroups = {};
        const controlGroups = {};
        this.allGeometries.forEach(g => {
            if (g.name.startsWith('Control')) {
                if (!controlGroups[g.gamma]) {
                    controlGroups[g.gamma] = [];
                }
                controlGroups[g.gamma].push(g);
            } else {
                if (!gammaGroups[g.gamma]) {
                    gammaGroups[g.gamma] = [];
                }
                gammaGroups[g.gamma].push(g);
            }
        });

        // Add control trial at the very top
        if (Object.keys(controlGroups).length > 0) {
            const controlGeometry = Object.values(controlGroups)[0][0];
            const isControlCompleted = completedGeometryIds.has(controlGeometry.id);

            const controlContainer = document.createElement('div');
            controlContainer.className = 'mb-4';

            let controlButtonHtml;
            if (isControlCompleted) {
                controlButtonHtml = `
                    <button class="btn btn-sm btn-success" disabled>
                        Completed<br><small>CONTROL</small>
                    </button>
                `;
            } else {
                controlButtonHtml = `
                    <button class="btn btn-sm btn-warning" data-geom-id="${controlGeometry.id}">
                        CONTROL<br><small>Data Collection</small>
                    </button>
                `;
            }

            controlContainer.innerHTML = `
                <h5 class="mb-3 text-warning">Control Trial (α:95°, β:125°, γ:0°)</h5>
                <div class="mb-3">
                    ${controlButtonHtml}
                </div>
                <hr class="my-4 border-2 border-secondary">
            `;
            this.elements.geometryGrid.appendChild(controlContainer);
        }

        // Dynamically create grid tables for each gamma value
        Object.keys(gammaGroups).sort((a, b) => parseInt(a) - parseInt(b)).forEach(gamma => {
            const groupGeometries = gammaGroups[gamma];

            // Get unique, sorted alpha and beta values for this gamma group
            const alphaValues = [...new Set(groupGeometries.map(g => g.alpha))].sort((a, b) => a - b);
            const betaValues = [...new Set(groupGeometries.map(g => g.beta))].sort((a, b) => a - b);

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
                const geom = groupGeometries.find(g => g.alpha === alpha && g.beta === beta);
                if (geom) {
                    const isCompleted = completedGeometryIds.has(geom.id);
                    if (isCompleted) {
                        return `<td class="text-center">
                                                <button class="btn btn-sm btn-success" disabled>
                                                    Completed<br><small>${geom.name}</small>
                                                </button>
                                            </td>`;
                    } else {
                        return `<td class="text-center">
                                                <button class="btn btn-sm btn-outline-primary" data-geom-id="${geom.id}">
                                                    Start Trial<br><small>${geom.name}</small>
                                                </button>
                                            </td>`;
                    }
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
            this.elements.geometryGrid.appendChild(tableContainer);
        });
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

    /**
     * Start timer
     * @private
     */
    _startTimer() {
        this.trialState.timer = setInterval(() => {
            this.trialState.elapsed = Date.now() - this.trialState.startTime;
            this._updateStopwatch();
        }, 100);
    }

    /**
     * Stop timer
     * @private
     */
    _stopTimer() {
        if (this.trialState.timer) {
            clearInterval(this.trialState.timer);
            this.trialState.timer = null;
        }
    }

    /**
     * Update stopwatch display
     * @private
     */
    _updateStopwatch() {
        if (this.elements.stopwatch) {
            this.elements.stopwatch.textContent = this.ui.formatDuration(this.trialState.elapsed);
        }
    }

    /**
     * Update trial UI
     * @private
     */
    _updateTrialUI(running) {
        if (this.elements.startStopBtn) {
            if (running) {
                this.elements.startStopBtn.innerHTML = '<i class="fas fa-stop"></i> Stop Trial';
                this.elements.startStopBtn.classList.remove('btn-success');
                this.elements.startStopBtn.classList.add('btn-danger');
            } else {
                this.elements.startStopBtn.innerHTML = '<i class="fas fa-play"></i> Start Trial';
                this.elements.startStopBtn.classList.remove('btn-danger');
                this.elements.startStopBtn.classList.add('btn-success');
            }
        }
    }

    /**
     * Reset trial state
     * @private
     */
    _resetTrialState() {
        this.trialState.running = false;
        this.trialState.startTime = null;
        this.trialState.elapsed = 0;
        this._stopTimer();
        this._updateTrialUI(false);
        this.device.clearDataBuffer();
    }

    /**
     * Update UI
     * @private
     */
    _updateUI() {
        if (this.trialState.currentParticipant) {
            // Update participant info display
            if (this.elements.participantInfo) {
                this.elements.participantInfo.textContent =
                    `Current Participant: ${this.trialState.currentParticipant.name}`;
            }
        }
    }

    /**
     * Process trial data
     * @private
     */
    async _processTrialData(rawData) {
        if (!rawData || rawData.length === 0) {
            throw new Error('No data collected during trial');
        }

        // Convert raw data to CSV format
        const csvContent = 'timestamp,force\n' +
            rawData.map(point => `${point.timestamp},${point.force}`).join('\n');

        const blob = new Blob([csvContent], { type: 'text/csv' });
        const file = new File([blob], 'trial_data.csv', { type: 'text/csv' });

        // Process using existing upload workflow
        await this.uploadAndProcessFile(file);
    }

    /**
     * Display trial results
     * @private
     */
    _displayTrialResults(results) {
        // Show plots area
        this.ui.toggleElements({
            'plots-area': true,
            'step-interaction-area': true,
            'metrics-and-survey-area': true
        });

        // Update metrics
        if (this.elements.stepCount) {
            this.elements.stepCount.textContent = results.step_detection.step_count;
        }

        if (this.elements.instabilityLoss) {
            this.elements.instabilityLoss.textContent =
                this.ui.formatNumber(results.gait_metrics.step_variance);
        }

        // Update steps list
        this._updateStepsList(results.step_detection.step_times);

        // Render plots if available
        if (results.plots) {
            this._renderPlots(results.plots);
        }
    }

    /**
     * Update steps list
     * @private
     */
    _updateStepsList(stepTimes) {
        if (!this.elements.stepList) return;

        this.elements.stepList.innerHTML = '';
        stepTimes.forEach((stepTime, index) => {
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
            this.elements.stepList.appendChild(row);
        });

        this.trialState.steps = [...stepTimes];
    }

    /**
     * Render instability plot
     * @private
     */
    _renderInstabilityPlot() {
        const plotDiv = document.getElementById('instability-plot-3d');
        if (!plotDiv) return;

        if (!this.instabilityPlotData || this.instabilityPlotData.length < 1) {
            plotDiv.innerHTML = `<div class="text-center text-muted pt-5">No completed trials with instability data for this participant.</div>`;
            return;
        }

        try {
            // Create the 3D scatter plot using the same structure as the original
            const trace = {
                x: this.instabilityPlotData.map(d => d.alpha),
                y: this.instabilityPlotData.map(d => d.beta),
                z: this.instabilityPlotData.map(d => d.gamma),
                mode: 'markers',
                type: 'scatter3d',
                marker: {
                    color: this.instabilityPlotData.map(d => d.instability_loss),
                    colorscale: 'Viridis',
                    colorbar: {
                        title: 'Instability Loss'
                    },
                    size: 8,
                    // Use a different symbol for the 'Control' trial
                    symbol: this.instabilityPlotData.map(d => d.geometry_name.includes('Control') ? 'cross' : 'diamond')
                },
                text: this.instabilityPlotData.map(d => `Trial: ${d.geometry_name}<br>Loss: ${d.instability_loss.toFixed(4)}`),
                hoverinfo: 'text'
            };

            const layout = {
                title: 'Instability Loss vs. Crutch Geometry',
                scene: {
                    xaxis: { title: 'Alpha (α)', range: [70, 120] },
                    yaxis: { title: 'Beta (β)', range: [110, 150] },
                    zaxis: { title: 'Gamma (γ)', range: [-12, 12] }
                },
                margin: {
                    l: 0,
                    r: 0,
                    b: 0,
                    t: 40
                }
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
     * Display participant details
     * @private
     */
    _displayParticipantDetails() {
        const detailsTableBody = document.getElementById('participant-details-table');
        const detailsFooter = document.getElementById('participant-details-footer');

        if (!detailsTableBody) return;

        detailsTableBody.innerHTML = '';

        if (!this.trialState.currentParticipant || !this.trialState.currentParticipant.characteristics) {
            if (detailsFooter) detailsFooter.classList.add('d-none');
            return;
        }

        const characteristics = this.trialState.currentParticipant.characteristics;

        // Map characteristics to display names
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

        // Add rows to the table
        for (const [key, value] of Object.entries(detailsMap)) {
            const row = detailsTableBody.insertRow();
            row.innerHTML = `<th class="text-muted">${key}</th><td>${value ?? 'N/A'}</td>`;
        }

        if (detailsFooter) detailsFooter.classList.remove('d-none');
    }

    /**
     * Render plots
     * @private
     */
    _renderPlots(plots) {
        // Implementation for rendering force and histogram plots
        console.log('Rendering plots:', plots);
    }
}

// Export for use in other modules
window.SystematicMode = SystematicMode;
