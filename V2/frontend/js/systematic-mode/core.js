/**
 * Systematic Mode Core - Main orchestrator for systematic mode
 * Coordinates all systematic mode modules
 */

import { TrialRunner } from './trial-runner.js';
import { StepManager } from './step-manager.js';
import { DataProcessor } from './data-processor.js';
import { UIRenderer } from './ui-renderer.js';
import { GeometrySequencer } from './geometry-sequencer.js';
import { SurveyManager } from './survey-manager.js';

export class SystematicMode {
    constructor(apiClient, uiComponents, deviceManager) {
        this.api = apiClient;
        this.ui = uiComponents;
        this.device = deviceManager;

        // Initialize sub-modules
        this.trialRunner = new TrialRunner(deviceManager, uiComponents);
        this.stepManager = new StepManager(uiComponents);
        this.dataProcessor = new DataProcessor(apiClient, uiComponents);
        this.uiRenderer = new UIRenderer(uiComponents);
        this.geometrySequencer = new GeometrySequencer(apiClient, uiComponents);
        this.surveyManager = new SurveyManager(apiClient, uiComponents);

        // Participant and data state
        this.currentParticipant = null;
        this.trials = [];
        this.allGeometries = [];
        this.currentTrialId = null;  // Store the current trial being edited
        this.instabilityPlotData = [];

        // DOM elements
        this.elements = this._getElements();

        // Setup
        this._setupEventListeners();
        this._setupDeviceCallbacks();
        this._setupTrialRunnerCallbacks();
    }

    /**
     * Initialize systematic mode for a participant
     * @param {Object} participant - Participant data
     */
    async initialize(participant) {
        this.currentParticipant = participant;
        await this._loadParticipantData(participant.id);
        this._updateUI();
    }

    /**
     * Start a new trial
     * @param {Object} geometry - Geometry configuration
     */
    async startTrial(geometry) {
        const testMode = this.elements.testModeCheckbox?.checked || false;

        try {
            const result = await this.trialRunner.startTrial(geometry, testMode);
            this.geometrySequencer.setCurrentGeometry(geometry);
            this._updateTrialUI(true);

            const message = result.testMode ? 'Trial started (Test Mode)' : 'Trial started';
            this.ui.showNotification(message, 'success');

        } catch (error) {
            this.ui.showNotification(`Failed to start trial: ${error.message}`, 'danger');
            this.trialRunner.reset();
        }
    }

    /**
     * Stop the current trial
     */
    async stopTrial() {
        if (!this.trialRunner.isRunning()) {
            return;
        }

        try {
            const result = await this.trialRunner.stopTrial();
            this._updateTrialUI(false);

            // Process trial data
            await this._processTrialData(result.rawData);

            this.ui.showNotification('Trial stopped and data processed', 'success');

        } catch (error) {
            this.ui.showNotification(`Failed to stop trial: ${error.message}`, 'danger');
            this.trialRunner.reset();
        }
    }

    /**
     * Upload and process a data file
     * @param {File} file - Data file to upload
     */
    async uploadAndProcessFile(file) {
        if (!this.currentParticipant) {
            this.ui.showNotification('Please select a participant first', 'warning');
            return;
        }

        try {
            if (this.elements.uploadBtn) {
                this.ui.toggleLoading(this.elements.uploadBtn, true, 'Processing...');
            }

            const trialContext = {
                participantId: this.currentParticipant.id,
                geometry: this.geometrySequencer.getCurrentGeometry()
            };

            const result = await this.dataProcessor.uploadAndProcessFile(file, trialContext);

            // Store the trial ID for later update
            this.currentTrialId = result.trial?.id;
            console.log('Trial created with ID:', this.currentTrialId);

            // Display results
            this._displayTrialResults(result.results);

            // Refresh participant data
            await this._loadParticipantData(this.currentParticipant.id);

            this.ui.showNotification('File processed successfully', 'success');

        } catch (error) {
            console.error('Upload and process error:', error);
            this.ui.showNotification(`Failed to process file: ${error.message}`, 'danger');
        } finally {
            if (this.elements.uploadBtn) {
                this.ui.toggleLoading(this.elements.uploadBtn, false);
            }
        }
    }

    /**
     * Delete a trial
     * @param {number} trialId - Trial ID to delete
     */
    async deleteTrial(trialId) {
        // Get trial details for better confirmation message
        const trial = this.trials?.find(t => t.id === trialId);
        const trialDescription = trial ?
            `Trial #${trialId} (${trial.geometry_name || 'Unknown Geometry'})` :
            `Trial #${trialId}`;

        const confirmed = await this.ui.showConfirmDialog(
            `Are you sure you want to permanently delete ${trialDescription}?<br><br><strong class="text-danger">⚠️ This action cannot be undone!</strong>`,
            'Delete Trial'
        );

        if (!confirmed) return;

        try {
            await this.api.deleteTrial(trialId);
            await this._loadParticipantData(this.currentParticipant.id);
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
            this.ui.showNotification(`Viewing trial ${trialId}`, 'info');
            console.log('Trial details:', trial);
        }
    }

    /**
     * Remove a step from the current trial
     * @param {number} stepIndex - Index of step to remove
     */
    removeStep(stepIndex) {
        const removed = this.stepManager.removeStep(stepIndex);

        if (removed) {
            // Update steps list
            this._updateStepsList(this.stepManager.getSteps());

            // Update variance
            this._updateVariance(this.stepManager.getSteps());

            // Re-render plot
            if (this.stepManager.getPlotData()) {
                const updatedPlots = {
                    ...this.stepManager.getPlotData(),
                    step_times: this.stepManager.getSteps()
                };
                this.stepManager.setPlotData(updatedPlots);
                this._renderPlots(updatedPlots);
            }

            this.ui.showNotification(`Step ${stepIndex + 1} removed`, 'info');
        }
    }

    /**
     * Set the current sequence
     * @param {string} sequenceName - Name of the sequence
     */
    async setSequence(sequenceName) {
        this.geometrySequencer.setCurrentSequence(sequenceName);
        if (this.currentParticipant) {
            await this.loadNextGeometry();
        }
    }

    /**
     * Load the next geometry
     */
    async loadNextGeometry() {
        if (!this.currentParticipant) {
            return;
        }

        try {
            const nextGeometry = await this.geometrySequencer.loadNextGeometry(
                this.currentParticipant.id
            );

            if (nextGeometry) {
                this.geometrySequencer.renderNextGeometry(
                    this.elements.nextGeometryInfo,
                    this.elements.startTrialBtn
                );
                this.elements.nextGeometrySection?.classList.remove('d-none');
            } else {
                this.geometrySequencer.renderCompletionMessage(
                    this.elements.nextGeometryInfo,
                    this.elements.startTrialBtn
                );
                this.elements.nextGeometrySection?.classList.remove('d-none');
            }

            await this._updateProgress();

        } catch (error) {
            console.error('Error loading next geometry:', error);
            this.ui.showNotification('Error loading next geometry', 'error');
        }
    }

    /**
     * Start trial from geometry
     */
    async startTrialFromGeometry() {
        const geometry = this.geometrySequencer.getCurrentGeometry();
        if (!geometry) {
            this.ui.showNotification('No geometry selected', 'error');
            return;
        }

        try {
            const result = await this.geometrySequencer.createTrialFromGeometry(
                this.currentParticipant.id
            );

            this.ui.showNotification(`Trial ${result.trial_id} created successfully`, 'success');

            // Set geometry in inputs
            if (this.elements.alphaInput) {
                this.elements.alphaInput.value = geometry.alpha;
            }
            if (this.elements.betaInput) {
                this.elements.betaInput.value = geometry.beta;
            }
            if (this.elements.gammaInput) {
                this.elements.gammaInput.value = geometry.gamma;
            }

            // Load next geometry
            await this.loadNextGeometry();

        } catch (error) {
            console.error('Error starting trial:', error);
            this.ui.showNotification('Error starting trial', 'error');
        }
    }

    /**
     * Skip current geometry
     */
    async skipCurrentGeometry() {
        if (confirm('Are you sure you want to skip this geometry? This action cannot be undone.')) {
            await this.loadNextGeometry();
        }
    }

    // ===== PRIVATE METHODS =====

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
            fileUploadInput: document.getElementById('file-upload-input'),
            testModeCheckbox: document.getElementById('test-mode-checkbox'),
            sequenceSelect: document.getElementById('sequence-select'),
            nextGeometrySection: document.getElementById('next-geometry-section'),
            nextGeometryInfo: document.getElementById('next-geometry-info'),
            startTrialBtn: document.getElementById('start-trial-btn'),
            skipGeometryBtn: document.getElementById('skip-geometry-btn'),
            progressBar: document.getElementById('progress-bar'),
            progressText: document.getElementById('progress-text'),
            alphaInput: document.getElementById('systematic-alpha'),
            betaInput: document.getElementById('systematic-beta'),
            gammaInput: document.getElementById('systematic-gamma')
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

        // Manual trial creation
        const createManualTrialBtn = document.getElementById('create-manual-trial-btn');
        createManualTrialBtn?.addEventListener('click', () => {
            this._handleManualTrialCreation();
        });

        // Close trial runner
        const closeTrialRunnerBtn = document.getElementById('close-trial-runner-btn');
        closeTrialRunnerBtn?.addEventListener('click', () => {
            const trialRunnerSection = document.getElementById('trial-runner-section');
            trialRunnerSection?.classList.add('d-none');
            this.geometrySequencer.setCurrentGeometry(null);
        });

        // Test mode checkbox
        this.elements.testModeCheckbox?.addEventListener('change', (e) => {
            const testMode = e.target.checked;
            if (testMode) {
                if (this.elements.startStopBtn) {
                    this.elements.startStopBtn.disabled = false;
                }
                this.ui.showNotification('Test mode enabled - trials will use simulated data', 'info');
            } else {
                if (this.elements.startStopBtn && !this.device.getConnectionStatus()) {
                    this.elements.startStopBtn.disabled = true;
                }
            }
        });

        // Save trial
        const saveTrialBtn = document.getElementById('save-trial-btn');
        saveTrialBtn?.addEventListener('click', () => {
            this._saveTrial();
        });

        // Start/Stop trial
        this.elements.startStopBtn?.addEventListener('click', () => {
            if (this.trialRunner.isRunning()) {
                this.stopTrial();
            } else {
                this.startTrial(this.geometrySequencer.getCurrentGeometry());
            }
        });

        // Event delegation for geometry buttons
        document.addEventListener('click', (e) => {
            if (e.target.closest('#grid-search-tables') && e.target.tagName === 'BUTTON' && e.target.dataset.geomId) {
                this._handleGeometryButtonClick(e.target.dataset.geomId);
            }
        });

        // Dynamic geometry listeners
        this.elements.sequenceSelect?.addEventListener('change', async (e) => {
            const sequenceName = e.target.value;
            if (sequenceName) {
                await this.setSequence(sequenceName);
            }
        });

        this.elements.startTrialBtn?.addEventListener('click', () => {
            this.startTrialFromGeometry();
        });

        this.elements.skipGeometryBtn?.addEventListener('click', () => {
            this.skipCurrentGeometry();
        });
    }

    /**
     * Setup device callbacks
     * @private
     */
    _setupDeviceCallbacks() {
        this.device.setConnectionChangeCallback((status, message) => {
            const statusElement = document.getElementById('device-status');
            if (statusElement) {
                statusElement.textContent = `Status: ${message}`;
                statusElement.className = `alert ${status === 'connected' ? 'alert-success' :
                    status === 'connecting' || status === 'searching' ? 'alert-info' :
                        'alert-secondary'
                    }`;
            }

            if (this.elements.startStopBtn) {
                this.elements.startStopBtn.disabled = status !== 'connected';
            }
        });

        this.device.setErrorCallback((error) => {
            this.ui.showNotification(error, 'danger');
        });
    }

    /**
     * Setup trial runner callbacks
     * @private
     */
    _setupTrialRunnerCallbacks() {
        this.trialRunner.setTimerCallback((elapsed) => {
            if (this.elements.stopwatch) {
                this.elements.stopwatch.textContent = this.ui.formatDuration(elapsed);
            }
        });
    }

    /**
     * Load participant data
     * @private
     */
    async _loadParticipantData(participantId) {
        try {
            console.log('Loading participant data for ID:', participantId);

            const [participantData, allTrials, allGeometries] = await Promise.all([
                this.api.request(`/experiments/participants/${participantId}`),
                this.api.getTrials(),
                this.api.getGeometries()
            ]);

            console.log('Participant API response:', participantData);
            console.log('Trials loaded:', allTrials.length);
            console.log('Geometries loaded:', allGeometries.length);

            this.currentParticipant = participantData;
            this.trials = allTrials;
            this.allGeometries = allGeometries;

            // Build instability plot data
            const participantTrialsWithInstability = allTrials.filter(t =>
                t.participant_id === participantId &&
                t.instability_loss !== undefined
            );

            this.instabilityPlotData = participantTrialsWithInstability.map(trial => {
                // Calculate normalized survey scores with proper null checks
                const susScore = trial.survey_responses?.sus_score || 0;
                const nrsScore = trial.survey_responses?.nrs_score || 0;
                const tlxScore = trial.survey_responses?.tlx_score || 0;
                const instabilityLoss = trial.instability_loss || 0;

                // Normalize scores to 0-1 range (lower is better for all metrics)
                // SUS: 0-100, lower is worse, so invert: (100 - score) / 100
                // NRS: 0-10, lower is better, so: score / 10
                // TLX: 0-100, lower is better, so: score / 100
                const normalizedSus = (100 - susScore) / 100;
                const normalizedNrs = nrsScore / 10;
                const normalizedTlx = tlxScore / 100;

                // Calculate cumulative score (instability loss + normalized surveys)
                const cumulativeScore = instabilityLoss + normalizedSus + normalizedNrs + normalizedTlx;

                return {
                    alpha: trial.alpha,
                    beta: trial.beta,
                    gamma: trial.gamma,
                    instability_loss: instabilityLoss,
                    cumulative_score: cumulativeScore,
                    sus_score: susScore,
                    nrs_score: nrsScore,
                    tlx_score: tlxScore,
                    geometry_name: trial.geometry_name || `Trial ${trial.id}`
                };
            });

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
        this.uiRenderer.renderTrialsTable(
            this.currentParticipant,
            this.trials,
            this.elements.trialsTableBody
        );
    }

    /**
     * Render geometry grid
     * @private
     */
    _renderGeometryGrid() {
        this.uiRenderer.renderGeometryGrid(
            this.currentParticipant,
            this.allGeometries,
            this.trials,
            this.elements.geometryGrid
        );
    }

    /**
     * Render instability plot
     * @private
     */
    _renderInstabilityPlot() {
        const plotDiv = document.getElementById('instability-plot-3d');
        this.uiRenderer.renderInstabilityPlot(this.instabilityPlotData, plotDiv);
    }

    /**
     * Display participant details
     * @private
     */
    _displayParticipantDetails() {
        const tableBody = document.getElementById('participant-details-table');
        const footer = document.getElementById('participant-details-footer');
        this.uiRenderer.displayParticipantDetails(this.currentParticipant, tableBody, footer);
    }

    /**
     * Handle manual trial creation
     * @private
     */
    _handleManualTrialCreation() {
        if (!this.currentParticipant) {
            this.ui.showNotification('Please select a participant first', 'warning');
            return;
        }

        const alpha = parseFloat(this.elements.alphaInput?.value) || 95;
        const beta = parseFloat(this.elements.betaInput?.value) || 95;
        const gamma = parseFloat(this.elements.gammaInput?.value) || 0;

        const geometry = {
            id: null,
            name: `α${alpha}_β${beta}_γ${gamma}`,
            alpha: alpha,
            beta: beta,
            gamma: gamma,
            delta: 0
        };

        const trialRunnerSection = document.getElementById('trial-runner-section');
        const trialRunnerTitle = document.getElementById('trial-runner-title');

        if (trialRunnerSection) {
            trialRunnerSection.classList.remove('d-none');
        }

        if (trialRunnerTitle) {
            trialRunnerTitle.textContent = `Run Trial: ${geometry.name}`;
        }

        this.geometrySequencer.setCurrentGeometry(geometry);
        this.ui.showNotification(`Ready to run trial with ${geometry.name}`, 'info');
        trialRunnerSection?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    /**
     * Handle geometry button click
     * @private
     */
    _handleGeometryButtonClick(geomId) {
        const geometry = this.allGeometries?.find(g => g.id == geomId);
        if (!geometry) {
            this.ui.showNotification('Geometry not found', 'danger');
            return;
        }

        const trialRunnerSection = document.getElementById('trial-runner-section');
        const trialRunnerTitle = document.getElementById('trial-runner-title');

        if (trialRunnerSection) {
            trialRunnerSection.classList.remove('d-none');
        }

        if (trialRunnerTitle) {
            trialRunnerTitle.textContent = `Run Trial: ${geometry.name}`;
        }

        this.geometrySequencer.setCurrentGeometry(geometry);
        this.ui.showNotification(`Selected geometry: ${geometry.name}`, 'info');
        trialRunnerSection?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
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
     * Update UI
     * @private
     */
    _updateUI() {
        if (this.currentParticipant && this.elements.participantInfo) {
            this.elements.participantInfo.textContent =
                `Current Participant: ${this.currentParticipant.name}`;
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

        console.log('Processing trial data:', {
            dataPoints: rawData.length,
            duration: rawData[rawData.length - 1]?.acc_x_time || 0,
            geometry: this.geometrySequencer.getCurrentGeometry()
        });

        const trialContext = {
            participantId: this.currentParticipant.id,
            geometry: this.geometrySequencer.getCurrentGeometry()
        };

        const result = await this.dataProcessor.processTrialData(rawData, trialContext);

        // Store the trial ID for later update
        this.currentTrialId = result.trial?.id;
        console.log('Trial created with ID:', this.currentTrialId);

        this._displayTrialResults(result.results);
    }

    /**
     * Display trial results
     * @private
     */
    _displayTrialResults(results) {
        console.log('Displaying trial results:', results);

        // Show UI areas
        this.ui.toggleElements({
            'plots-area': true,
            'step-interaction-area': true,
            'metrics-and-survey-area': true
        });

        // Update step count
        if (this.elements.stepCount) {
            this.elements.stepCount.textContent = results.step_detection.step_count;
        }

        // Update variance
        this._updateVariance(results.step_detection.step_times);

        // Update step manager
        this.stepManager.updateFromResults(results);

        // Update steps list
        this._updateStepsList(this.stepManager.getSteps());

        // Render plots
        if (results.plots) {
            this._renderPlots(results.plots);
        }
    }

    /**
     * Update steps list
     * @private
     */
    _updateStepsList(stepTimes) {
        if (this.elements.stepCount) {
            this.elements.stepCount.textContent = stepTimes.length;
        }

        this.uiRenderer.renderStepsList(stepTimes, this.elements.stepList);
    }

    /**
     * Update variance
     * @private
     */
    _updateVariance(stepTimes) {
        if (!this.elements.instabilityLoss) {
            return;
        }

        const variance = this.stepManager.calculateVariance(stepTimes);

        if (variance !== null) {
            this.elements.instabilityLoss.textContent = this.ui.formatNumber(variance, 6);
        } else {
            this.elements.instabilityLoss.textContent = 'N/A';
        }
    }

    /**
     * Render plots
     * @private
     */
    _renderPlots(plots) {
        const forcePlotDiv = document.getElementById('force-plot-div');
        this.uiRenderer.renderForcePlot(plots, forcePlotDiv, (data, plotData) => {
            this._onPlotClick(data, plotData);
        });
    }

    /**
     * Handle plot click
     * @private
     */
    _onPlotClick(data, plots) {
        const result = this.stepManager.handlePlotClick(data, plots);

        if (!result.success) {
            if (result.reason === 'duplicate') {
                this.ui.showNotification(result.message, 'warning');
            }
            return;
        }

        // Update UI
        this._updateStepsList(result.steps);
        this._updateVariance(result.steps);
        this._renderPlots(this.stepManager.getPlotData());

        this.ui.showNotification(`Step added at ${result.stepTime.toFixed(2)}s`, 'success');
    }

    /**
     * Update progress
     * @private
     */
    async _updateProgress() {
        if (!this.currentParticipant) {
            return;
        }

        await this.geometrySequencer.updateProgressUI(
            this.currentParticipant.id,
            this.elements.progressBar,
            this.elements.progressText
        );
    }

    /**
     * Save trial
     * @private
     */
    async _saveTrial() {
        try {
            // Collect survey data
            const surveyResponses = this.surveyManager.collectSurveyResponses();
            const metabolicCost = this.surveyManager.getMetabolicCost();

            // Get current metrics
            const instabilityLoss = this.elements.instabilityLoss ?
                parseFloat(this.elements.instabilityLoss.textContent) : 0;

            // Prepare trial data
            const trialData = {
                trialId: this.currentTrialId,  // Pass existing trial ID if available
                participantId: this.currentParticipant.id,
                geometry: this.geometrySequencer.getCurrentGeometry(),
                steps: this.stepManager.getSteps(),
                instabilityLoss: instabilityLoss,
                surveyResponses: surveyResponses,
                metabolicCost: metabolicCost
            };

            console.log('_saveTrial called with currentTrialId:', this.currentTrialId);
            console.log('Trial data being passed:', trialData);

            // Save trial (create new or update existing)
            await this.surveyManager.saveTrial(trialData);
            this.ui.showNotification('Trial saved successfully!', 'success');

            // Clear the current trial ID
            this.currentTrialId = null;
            console.log('Trial ID cleared after save');

            // Reset UI
            this._resetTrialUI();

            // Reload participant data
            if (this.currentParticipant) {
                await this._loadParticipantData(this.currentParticipant.id);
            }

        } catch (error) {
            console.error('Error saving trial:', error);
            this.ui.showNotification(`Error saving trial: ${error.message}`, 'error');
        }
    }

    /**
     * Reset trial UI
     * @private
     */
    _resetTrialUI() {
        // Hide trial runner
        const trialRunnerSection = document.getElementById('trial-runner-section');
        trialRunnerSection?.classList.add('d-none');

        // Hide plots and surveys
        this.ui.toggleElements({
            'plots-area': false,
            'step-interaction-area': false,
            'metrics-and-survey-area': false
        });

        // Reset forms
        this.surveyManager.resetSurveyForms();

        // Reset state
        this.stepManager.clear();
        this.geometrySequencer.setCurrentGeometry(null);

        // Reset metrics display
        if (this.elements.instabilityLoss) {
            this.elements.instabilityLoss.textContent = '-';
        }
        if (this.elements.stepCount) {
            this.elements.stepCount.textContent = '0';
        }
    }
}

