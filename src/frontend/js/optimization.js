/**
 * Optimization Module - Handles Bayesian Optimization workflows
 * Manages Pain, Effort, and Instability optimization sessions
 */

class OptimizationManager {
    constructor(apiClient, uiComponents) {
        this.api = apiClient;
        this.ui = uiComponents;

        // Optimization states
        this.painState = {
            active: false,
            userId: null,
            currentGeometry: null,
            timer: null,
            startTime: null,
            elapsed: 0,
            dataParser: null,
            bleCharacteristic: null,
            bleServer: null,
            existingData: null,
            restartMode: false,
            steps: []
        };

        this.effortState = {
            active: false,
            userId: null,
            currentGeometry: null,
            timer: null,
            startTime: null,
            elapsed: 0,
            dataParser: null,
            bleCharacteristic: null,
            bleServer: null,
            existingData: null,
            restartMode: false,
            steps: []
        };

        this.instabilityState = {
            active: false,
            userId: null,
            currentGeometry: null,
            timer: null,
            startTime: null,
            elapsed: 0,
            analysisResults: null,
            dataParser: null,
            bleCharacteristic: null,
            bleServer: null,
            existingData: null,
            restartMode: false,
            steps: []
        };

        this.elements = this._getElements();
        this._setupEventListeners();
    }

    /**
     * Start Pain Optimization workflow
     * @param {number} userId - Participant ID
     */
    async startPainOptimization(userId) {
        try {
            this.painState.userId = userId;

            // Check for existing NRS data
            const existingDataResponse = await this.api.request(`/pain-bo/check-existing-data?userId=${userId}`);

            if (existingDataResponse.has_existing_data) {
                this._showPainOptimizationDecision(existingDataResponse);
            } else {
                await this._startPainOptimizationSession(userId, true);
            }

        } catch (error) {
            this.ui.showNotification(`Failed to start pain optimization: ${error.message}`, 'danger');
        }
    }

    /**
     * Start Effort Optimization workflow
     * @param {number} userId - Participant ID
     */
    async startEffortOptimization(userId) {
        try {
            this.effortState.userId = userId;

            // Check for existing TLX data
            const existingDataResponse = await this.api.request(`/effort-bo/check-existing-data?userId=${userId}`);

            if (existingDataResponse.has_existing_data) {
                this._showEffortOptimizationDecision(existingDataResponse);
            } else {
                await this._startEffortOptimizationSession(userId, true);
            }

        } catch (error) {
            this.ui.showNotification(`Failed to start effort optimization: ${error.message}`, 'danger');
        }
    }

    /**
     * Start Instability Optimization workflow
     * @param {number} userId - Participant ID
     */
    async startInstabilityOptimization(userId) {
        try {
            const response = await this.api.request(`/instability-bo/check-existing-data?userId=${userId}`);

            if (response.has_existing_data) {
                this._showInstabilityOptimizationDecision(response, userId);
            } else {
                await this._startInstabilityOptimizationSession(userId, true);
            }

        } catch (error) {
            this.ui.showNotification(`Failed to start instability optimization: ${error.message}`, 'danger');
        }
    }

    /**
     * Exit current optimization session
     * @param {string} type - Type of optimization ('pain', 'effort', 'instability')
     */
    exitOptimization(type) {
        const state = this[`${type}State`];
        if (state) {
            state.active = false;
            this._resetOptimizationState(state);
            this._hideOptimizationScreen(type);
            this.ui.showNotification(`${type.charAt(0).toUpperCase() + type.slice(1)} optimization session ended`, 'info');
        }
    }

    /**
     * Process trial data for optimization
     * @param {string} type - Optimization type
     * @param {Object} trialData - Trial data to process
     */
    async processOptimizationTrial(type, trialData) {
        try {
            const endpoint = `/${type}-bo/process-trial`;
            const result = await this.api.request(endpoint, 'POST', trialData);

            this.ui.showNotification(`${type.charAt(0).toUpperCase() + type.slice(1)} trial processed successfully`, 'success');
            return result;
        } catch (error) {
            this.ui.showNotification(`Failed to process ${type} trial: ${error.message}`, 'danger');
            throw error;
        }
    }

    /**
     * Get optimization history for a participant
     * @param {number} userId - Participant ID
     * @param {string} type - Optimization type
     * @returns {Promise<Array>} Optimization history
     */
    async getOptimizationHistory(userId, type) {
        try {
            return await this.api.request(`/${type}-bo/history/${userId}`);
        } catch (error) {
            console.error(`Failed to get ${type} optimization history:`, error);
            return [];
        }
    }

    /**
     * Get next geometry suggestion
     * @param {number} userId - Participant ID
     * @param {string} type - Optimization type
     * @returns {Promise<Object>} Suggested geometry
     */
    async getNextGeometry(userId, type) {
        try {
            return await this.api.request(`/${type}-bo/suggest-geometry/${userId}`, 'POST');
        } catch (error) {
            this.ui.showNotification(`Failed to get next geometry suggestion: ${error.message}`, 'danger');
            throw error;
        }
    }

    /**
     * Private method to show pain optimization decision screen
     * @private
     */
    _showPainOptimizationDecision(existingData) {
        this.ui.toggleElements({
            'bo-objective-select-card': false,
            'pain-bo-decision-card': true
        });

        const gridSearchTrials = existingData.grid_search_trials;
        const painBoTrials = existingData.pain_bo_trials;
        const totalTrials = existingData.total_nrs_trials;

        let summaryText = `Found ${totalTrials} trial(s) with NRS pain scores: `;
        if (gridSearchTrials > 0 && painBoTrials > 0) {
            summaryText += `${gridSearchTrials} from Grid Search mode and ${painBoTrials} from previous Pain Optimization sessions.`;
        } else if (gridSearchTrials > 0) {
            summaryText += `${gridSearchTrials} from Grid Search mode.`;
        } else if (painBoTrials > 0) {
            summaryText += `${painBoTrials} from previous Pain Optimization sessions.`;
        }

        this.elements.painExistingDataSummary.textContent = summaryText;
    }

    /**
     * Private method to show effort optimization decision screen
     * @private
     */
    _showEffortOptimizationDecision(existingData) {
        this.ui.toggleElements({
            'bo-objective-select-card': false,
            'effort-bo-decision-card': true
        });

        const gridSearchTrials = existingData.grid_search_trials;
        const effortBoTrials = existingData.effort_bo_trials;
        const totalTrials = existingData.total_tlx_trials;

        let summaryText = `Found ${totalTrials} trial(s) with TLX effort scores: `;
        if (gridSearchTrials > 0 && effortBoTrials > 0) {
            summaryText += `${gridSearchTrials} from Grid Search mode and ${effortBoTrials} from previous Effort Optimization sessions.`;
        } else if (gridSearchTrials > 0) {
            summaryText += `${gridSearchTrials} from Grid Search mode.`;
        } else if (effortBoTrials > 0) {
            summaryText += `${effortBoTrials} from previous Effort Optimization sessions.`;
        }

        this.elements.effortExistingDataSummary.textContent = summaryText;
    }

    /**
     * Private method to show instability optimization decision screen
     * @private
     */
    _showInstabilityOptimizationDecision(existingData, userId) {
        this.ui.toggleElements({
            'bo-objective-select-card': false,
            'instability-bo-decision-card': true
        });

        const gridSearchTrials = existingData.grid_search_trials;
        const instabilityBoTrials = existingData.instability_bo_trials;
        const totalTrials = existingData.total_instability_trials;

        let summaryText = `Found ${totalTrials} trial(s) with instability data: `;
        if (gridSearchTrials > 0 && instabilityBoTrials > 0) {
            summaryText += `${gridSearchTrials} from Grid Search mode and ${instabilityBoTrials} from previous Instability Optimization sessions.`;
        } else if (gridSearchTrials > 0) {
            summaryText += `${gridSearchTrials} from Grid Search mode.`;
        } else if (instabilityBoTrials > 0) {
            summaryText += `${instabilityBoTrials} from previous Instability Optimization sessions.`;
        }

        this.elements.instabilityExistingDataSummary.textContent = summaryText;
        this.instabilityState.userId = userId;
        this.instabilityState.existingData = existingData;
    }

    /**
     * Private method to start pain optimization session
     * @private
     */
    async _startPainOptimizationSession(userId, restartMode) {
        try {
            this.painState.restartMode = restartMode;
            this.painState.active = true;
            this._showOptimizationScreen('pain');

            // Initialize session
            const response = await this.api.request(`/pain-bo/start-session/${userId}`, 'POST', {
                restart: restartMode
            });

            if (response.geometry) {
                this._displayCurrentGeometry('pain', response.geometry);
            }

        } catch (error) {
            this.ui.showNotification(`Failed to start pain optimization session: ${error.message}`, 'danger');
        }
    }

    /**
     * Private method to start effort optimization session
     * @private
     */
    async _startEffortOptimizationSession(userId, restartMode) {
        try {
            this.effortState.restartMode = restartMode;
            this.effortState.active = true;
            this._showOptimizationScreen('effort');

            // Initialize session
            const response = await this.api.request(`/effort-bo/start-session/${userId}`, 'POST', {
                restart: restartMode
            });

            if (response.geometry) {
                this._displayCurrentGeometry('effort', response.geometry);
            }

        } catch (error) {
            this.ui.showNotification(`Failed to start effort optimization session: ${error.message}`, 'danger');
        }
    }

    /**
     * Private method to start instability optimization session
     * @private
     */
    async _startInstabilityOptimizationSession(userId, restartMode) {
        try {
            this.instabilityState.restartMode = restartMode;
            this.instabilityState.active = true;
            this._showOptimizationScreen('instability');

            // Initialize session
            const response = await this.api.request(`/instability-bo/start-session/${userId}`, 'POST', {
                restart: restartMode
            });

            if (response.geometry) {
                this._displayCurrentGeometry('instability', response.geometry);
            }

        } catch (error) {
            this.ui.showNotification(`Failed to start instability optimization session: ${error.message}`, 'danger');
        }
    }

    /**
     * Private method to show optimization screen
     * @private
     */
    _showOptimizationScreen(type) {
        // Hide all other screens
        this.ui.toggleElements({
            'bo-objective-select-card': false,
            'pain-bo-decision-card': false,
            'effort-bo-decision-card': false,
            'instability-bo-decision-card': false,
            [`${type}-optimization-screen`]: true
        });
    }

    /**
     * Private method to hide optimization screen
     * @private
     */
    _hideOptimizationScreen(type) {
        this.ui.toggleElements({
            [`${type}-optimization-screen`]: false,
            'bo-objective-select-card': true
        });
    }

    /**
     * Private method to display current geometry
     * @private
     */
    _displayCurrentGeometry(type, geometry) {
        const state = this[`${type}State`];
        state.currentGeometry = geometry;

        // Update UI elements for current geometry
        const elements = this.elements[type];
        if (elements && elements.currentGeometry) {
            elements.currentGeometry.innerHTML = `
                <h6>Current Geometry: ${geometry.name}</h6>
                <p>Alpha: ${geometry.alpha}°, Beta: ${geometry.beta}°, Gamma: ${geometry.gamma}°</p>
            `;
        }
    }

    /**
     * Private method to reset optimization state
     * @private
     */
    _resetOptimizationState(state) {
        Object.assign(state, {
            active: false,
            userId: null,
            currentGeometry: null,
            timer: null,
            startTime: null,
            elapsed: 0,
            dataParser: null,
            bleCharacteristic: null,
            bleServer: null,
            existingData: null,
            restartMode: false,
            steps: [],
            analysisResults: null
        });
    }

    /**
     * Private method to get DOM elements
     * @private
     */
    _getElements() {
        return {
            painExistingDataSummary: document.getElementById('pain-existing-data-summary'),
            effortExistingDataSummary: document.getElementById('effort-existing-data-summary'),
            instabilityExistingDataSummary: document.getElementById('instability-existing-data-summary'),

            pain: {
                screen: document.getElementById('pain-optimization-screen'),
                currentGeometry: document.getElementById('pain-current-geometry'),
                trialCard: document.getElementById('pain-trial-card'),
                surveyCard: document.getElementById('pain-survey-card'),
                resultsCard: document.getElementById('pain-results-card')
            },

            effort: {
                screen: document.getElementById('effort-optimization-screen'),
                currentGeometry: document.getElementById('effort-current-geometry'),
                trialCard: document.getElementById('effort-trial-card'),
                surveyCard: document.getElementById('effort-survey-card'),
                resultsCard: document.getElementById('effort-results-card')
            },

            instability: {
                screen: document.getElementById('instability-optimization-screen'),
                currentGeometry: document.getElementById('instability-current-geometry'),
                trialCard: document.getElementById('instability-trial-card'),
                surveyCard: document.getElementById('instability-survey-card'),
                resultsCard: document.getElementById('instability-results-card')
            }
        };
    }

    /**
     * Private method to setup event listeners
     * @private
     */
    _setupEventListeners() {
        // Pain optimization event listeners
        document.getElementById('pain-continue-btn')?.addEventListener('click', () => {
            this._startPainOptimizationSession(this.painState.userId, false);
        });

        document.getElementById('pain-restart-btn')?.addEventListener('click', () => {
            this._startPainOptimizationSession(this.painState.userId, true);
        });

        document.getElementById('pain-exit-btn')?.addEventListener('click', () => {
            this.exitOptimization('pain');
        });

        // Effort optimization event listeners
        document.getElementById('effort-continue-btn')?.addEventListener('click', () => {
            this._startEffortOptimizationSession(this.effortState.userId, false);
        });

        document.getElementById('effort-restart-btn')?.addEventListener('click', () => {
            this._startEffortOptimizationSession(this.effortState.userId, true);
        });

        document.getElementById('effort-exit-btn')?.addEventListener('click', () => {
            this.exitOptimization('effort');
        });

        // Instability optimization event listeners
        document.getElementById('instability-continue-btn')?.addEventListener('click', () => {
            this._startInstabilityOptimizationSession(this.instabilityState.userId, false);
        });

        document.getElementById('instability-restart-btn')?.addEventListener('click', () => {
            this._startInstabilityOptimizationSession(this.instabilityState.userId, true);
        });

        document.getElementById('instability-exit-btn')?.addEventListener('click', () => {
            this.exitOptimization('instability');
        });
    }

    /**
     * Get current optimization state
     * @param {string} type - Optimization type
     * @returns {Object} Current state for the optimization type
     */
    getState(type) {
        return this[`${type}State`];
    }

    /**
     * Check if any optimization is currently active
     * @returns {boolean} True if any optimization is active
     */
    isAnyOptimizationActive() {
        return this.painState.active || this.effortState.active || this.instabilityState.active;
    }

    /**
     * Get the currently active optimization type
     * @returns {string|null} Active optimization type or null
     */
    getActiveOptimizationType() {
        if (this.painState.active) return 'pain';
        if (this.effortState.active) return 'effort';
        if (this.instabilityState.active) return 'instability';
        return null;
    }
}

// Export for use in other modules
window.OptimizationManager = OptimizationManager;
