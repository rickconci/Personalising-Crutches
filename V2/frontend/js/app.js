/**
 * Main application logic for Personalising Crutches
 */

class CrutchApp {
    constructor() {
        // Global application state
        this.state = {
            mode: null, // 'systematic' or 'bo'
            participants: [],
            currentParticipant: null,
            geometries: [],
            trials: [],
            initialized: false
        };

        // Module instances
        this.modules = {
            api: null,
            ui: null,
            device: null,
            systematic: null,
            optimization: null
        };

        this.elements = this._getElements();
        this._initializeModules();
    }

    /**
     * Initialize all modules and set up the application
     */
    async _initializeModules() {
        try {
            // Initialize core modules
            this.modules.api = window.api; // Already created by api.js
            this.modules.ui = window.ui;   // Already created by ui-components.js
            this.modules.device = new DeviceManager();

            // Initialize mode-specific modules
            this.modules.systematic = new SystematicMode(
                this.modules.api,
                this.modules.ui,
                this.modules.device
            );

            this.modules.optimization = new OptimizationManager(
                this.modules.api,
                this.modules.ui
            );

            // Load initial data and setup UI
            await this._loadInitialData();
            this._setupEventListeners();
            this._showScreen('modeSelection');

            this.state.initialized = true;
            console.log('App initialized successfully');

            // Expose modules globally for backward compatibility
            window.systematicMode = this.modules.systematic;
            window.optimizationManager = this.modules.optimization;

        } catch (error) {
            console.error('Failed to initialize app:', error);
            this.modules.ui?.showNotification('Failed to initialize app. Please refresh the page.', 'danger');
        }
    }

    /**
     * Load initial application data
     * @private
     */
    async _loadInitialData() {
        try {
            const [participants, geometries, trials] = await Promise.all([
                this.modules.api.getParticipants(),
                this.modules.api.getGeometries(),
                this.modules.api.getTrials()
            ]);

            this.state.participants = participants;
            this.state.geometries = geometries;
            this.state.trials = trials;

            console.log('Initial data loaded:', {
                participants: participants.length,
                geometries: geometries.length,
                trials: trials.length
            });

            // Update UI with loaded data
            this._updateParticipantSelects();

        } catch (error) {
            console.error('Failed to load initial data:', error);
            throw error;
        }
    }

    /**
     * Set up global event listeners
     * @private
     */
    _setupEventListeners() {
        // Mode selection
        this.elements.selectSystematicBtn?.addEventListener('click', async () => {
            await this._switchToMode('systematic');
        });

        this.elements.selectBOBtn?.addEventListener('click', async () => {
            await this._switchToMode('bo');
        });

        // Home button
        this.elements.homeBtn?.addEventListener('click', () => {
            this._returnToModeSelection();
        });

        // Participant management
        this.elements.participantSelect?.addEventListener('change', (e) => {
            this._onParticipantSelected(e.target.value);
        });

        this.elements.saveParticipantBtn?.addEventListener('click', (e) => {
            e.preventDefault();
            this._createParticipant();
        });

        this.elements.deleteParticipantBtn?.addEventListener('click', () => {
            this._deleteCurrentParticipant();
        });

        // BO objective selection
        document.querySelectorAll('.bo-objective-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const objective = e.target.dataset.objective;
                this._startBOObjective(objective);
            });
        });

        // Connect device button (systematic mode)
        this.elements.connectDeviceBtn?.addEventListener('click', () => {
            console.log('Connect device button clicked');
            this._connectDevice();
        });
    }

    /**
     * Switch between application modes
     * @param {string} mode - Mode to switch to ('systematic' or 'bo')
     */
    async _switchToMode(mode) {
        if (!this.state.initialized) {
            this.modules.ui.showNotification('App is still initializing, please wait...', 'warning');
            return;
        }

        this.state.mode = mode;
        this._showScreen(mode);
        this._updateParticipantSelects();

        // Initialize mode-specific functionality
        if (mode === 'systematic') {
            // If we have a current participant, initialize systematic mode with them
            if (this.state.currentParticipant) {
                await this.modules.systematic.initialize(this.state.currentParticipant);
            }
        } else if (mode === 'bo') {
            // BO mode setup is handled by the OptimizationManager module
        }

        this.modules.ui.showNotification(`Switched to ${mode} mode`, 'success');
    }

    /**
     * Show a specific screen and hide others
     * @param {string} screenName - Screen to show
     * @private
     */
    _showScreen(screenName) {
        const screens = {
            modeSelection: 'mode-selection-screen',
            systematic: 'systematic-screen',
            bo: 'bo-screen'
        };

        // Hide all screens
        Object.values(screens).forEach(screenId => {
            const screen = document.getElementById(screenId);
            if (screen) screen.classList.add('d-none');
        });

        // Show target screen
        const targetScreen = document.getElementById(screens[screenName]);
        if (targetScreen) {
            targetScreen.classList.remove('d-none');
        }
    }

    /**
     * Return to mode selection screen
     */
    _returnToModeSelection() {
        // Clean up any active sessions
        if (this.state.mode === 'bo' && this.modules.optimization.isAnyOptimizationActive()) {
            const activeType = this.modules.optimization.getActiveOptimizationType();
            this.modules.optimization.exitOptimization(activeType);
        }

        this.state.mode = null;
        this.state.currentParticipant = null;
        this._showScreen('modeSelection');
    }

    /**
     * Update all participant select elements
     * @private
     */
    _updateParticipantSelects() {
        const selects = [
            document.getElementById('participant-select'),
            document.getElementById('bo-participant-select')
        ].filter(select => select); // Remove null elements

        selects.forEach(select => {
            select.innerHTML = '<option value="">Select a participant...</option>';
            this.state.participants.forEach(participant => {
                const option = document.createElement('option');
                option.value = participant.id;
                option.textContent = participant.name;
                select.appendChild(option);
            });
        });
    }

    /**
     * Handle participant selection
     * @param {string} participantId - Selected participant ID
     * @private
     */
    async _onParticipantSelected(participantId) {
        console.log('_onParticipantSelected called with ID:', participantId);

        if (!participantId) {
            this.state.currentParticipant = null;
            return;
        }

        const participant = this.state.participants.find(p => p.id === parseInt(participantId));
        console.log('Found participant:', participant);

        if (!participant) {
            this.modules.ui.showNotification('Participant not found', 'danger');
            return;
        }

        this.state.currentParticipant = participant;
        console.log('Participant selected:', participant);
        console.log('Current mode:', this.state.mode);

        // Initialize mode-specific functionality for the selected participant
        if (this.state.mode === 'systematic') {
            console.log('Initializing systematic mode with participant');
            await this.modules.systematic.initialize(participant);
        } else if (this.state.mode === 'bo') {
            // BO mode initialization if needed
            console.log('BO mode - no initialization needed yet');
        } else {
            // If no mode is set yet, we might be in initial state
            // For now, assume systematic mode for participant selection
            console.log('No mode set, defaulting to systematic mode initialization');
            await this.modules.systematic.initialize(participant);
        }

        // Enable/disable relevant buttons
        this._updateUIForParticipant(participant);
    }

    /**
     * Update UI elements based on selected participant
     * @param {Object} participant - Selected participant
     * @private
     */
    _updateUIForParticipant(participant) {
        // Enable delete button
        if (this.elements.deleteParticipantBtn) {
            this.elements.deleteParticipantBtn.disabled = false;
        }

        // Update any participant-specific UI elements
        const participantInfo = document.getElementById('current-participant-info');
        if (participantInfo) {
            participantInfo.textContent = `Current: ${participant.name}`;
        }
    }

    /**
     * Create a new participant
     * @private
     */
    async _createParticipant() {
        const form = document.getElementById('new-participant-form');
        if (!form) return;

        const participantData = {
            name: document.getElementById('new-participant-name')?.value,
            characteristics: {
                height: parseFloat(document.getElementById('char-height')?.value) || null,
                weight: parseFloat(document.getElementById('char-weight')?.value) || null,
                forearm_length: parseFloat(document.getElementById('char-forearm')?.value) || null,
                fitness_level: document.getElementById('char-activity')?.value || null,
                age: parseInt(document.getElementById('char-age')?.value) || null,
                sex: document.getElementById('char-sex')?.value || null,
                activity_level: (() => {
                    const value = document.getElementById('char-activity')?.value;
                    if (!value || value === '-- Select --') return null;
                    const levelMap = { 'very-low': 1, 'low': 2, 'medium': 3, 'high': 4 };
                    return levelMap[value] || null;
                })(),
                previous_crutch_experience: document.querySelector('input[name="crutch-experience"]:checked')?.value === 'true'
            }
        };

        // Debug: Log the participant data being sent
        console.log('Participant data being sent:', participantData);

        // Validate required fields
        if (!participantData.name) {
            this.modules.ui.showNotification('Please enter a participant ID', 'warning');
            return;
        }

        try {
            const newParticipant = await this.modules.api.createParticipant(participantData);
            this.state.participants.push(newParticipant);
            this._updateParticipantSelects();

            // Close modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('create-participant-modal'));
            if (modal) modal.hide();

            this.modules.ui.showNotification('Participant created successfully', 'success');
            form.reset();

            // Automatically select the newly created participant
            if (this.elements.participantSelect) {
                this.elements.participantSelect.value = newParticipant.id;
                await this._onParticipantSelected(newParticipant.id);
            }

        } catch (error) {
            this.modules.ui.showNotification(`Failed to create participant: ${error.message}`, 'danger');
        }
    }

    /**
     * Delete the currently selected participant
     * @private
     */
    async _deleteCurrentParticipant() {
        if (!this.state.currentParticipant) {
            this.modules.ui.showNotification('No participant selected', 'warning');
            return;
        }

        const confirmed = await this.modules.ui.showConfirmDialog(
            `Are you sure you want to delete participant "${this.state.currentParticipant.name}"? This action cannot be undone.`,
            'Delete Participant'
        );

        if (!confirmed) return;

        try {
            await this.modules.api.deleteParticipant(this.state.currentParticipant.id);

            // Remove from state
            this.state.participants = this.state.participants.filter(
                p => p.id !== this.state.currentParticipant.id
            );

            // Update UI
            this._updateParticipantSelects();
            this.state.currentParticipant = null;

            // Reset participant selects
            const selects = [this.elements.participantSelect, document.getElementById('bo-participant-select')];
            selects.forEach(select => {
                if (select) select.value = '';
            });

            // Disable delete button
            if (this.elements.deleteParticipantBtn) {
                this.elements.deleteParticipantBtn.disabled = true;
            }

            this.modules.ui.showNotification('Participant deleted successfully', 'success');

        } catch (error) {
            this.modules.ui.showNotification(`Failed to delete participant: ${error.message}`, 'danger');
        }
    }

    /**
     * Start BO objective workflow
     * @param {string} objective - Objective type ('pain', 'effort', 'instability')
     * @private
     */
    async _startBOObjective(objective) {
        if (!this.state.currentParticipant) {
            this.modules.ui.showNotification('Please select a participant first', 'warning');
            return;
        }

        try {
            switch (objective) {
                case 'pain':
                    await this.modules.optimization.startPainOptimization(this.state.currentParticipant.id);
                    break;
                case 'effort':
                    await this.modules.optimization.startEffortOptimization(this.state.currentParticipant.id);
                    break;
                case 'instability':
                    await this.modules.optimization.startInstabilityOptimization(this.state.currentParticipant.id);
                    break;
                default:
                    this.modules.ui.showNotification('Unknown objective type', 'danger');
            }
        } catch (error) {
            this.modules.ui.showNotification(`Failed to start ${objective} optimization: ${error.message}`, 'danger');
        }
    }

    /**
     * Connect to device
     * @private
     */
    async _connectDevice() {
        console.log('_connectDevice called');
        try {
            console.log('Attempting to connect to device...');
            const success = await this.modules.device.connect();
            console.log('Connection result:', success);
            if (success) {
                this.modules.ui.showNotification('Device connected successfully', 'success');
            }
        } catch (error) {
            console.error('Connection error:', error);
            this.modules.ui.showNotification(`Failed to connect to device: ${error.message}`, 'danger');
        }
    }

    /**
     * Refresh application data
     */
    async refreshData() {
        try {
            await this._loadInitialData();
            this.modules.ui.showNotification('Data refreshed successfully', 'success');
        } catch (error) {
            this.modules.ui.showNotification(`Failed to refresh data: ${error.message}`, 'danger');
        }
    }

    /**
     * Get current application state
     * @returns {Object} Current application state
     */
    getState() {
        return { ...this.state };
    }

    /**
     * Get a specific module instance
     * @param {string} moduleName - Name of the module
     * @returns {Object|null} Module instance or null if not found
     */
    getModule(moduleName) {
        return this.modules[moduleName] || null;
    }

    /**
     * Check if the app is fully initialized
     * @returns {boolean} True if initialized
     */
    isInitialized() {
        return this.state.initialized;
    }

    /**
     * Get the current participant
     * @returns {Object|null} Current participant or null
     */
    getCurrentParticipant() {
        return this.state.currentParticipant;
    }

    /**
     * Get all participants
     * @returns {Array} Array of participants
     */
    getParticipants() {
        return [...this.state.participants];
    }

    /**
     * Get all geometries
     * @returns {Array} Array of geometries
     */
    getGeometries() {
        return [...this.state.geometries];
    }

    /**
     * Get all trials
     * @returns {Array} Array of trials
     */
    getTrials() {
        return [...this.state.trials];
    }

    /**
     * Get DOM elements
     * @private
     */
    _getElements() {
        return {
            selectSystematicBtn: document.getElementById('select-systematic-mode'),
            selectBOBtn: document.getElementById('select-bo-mode'),
            homeBtn: document.getElementById('home-btn'),
            participantSelect: document.getElementById('participant-select'),
            saveParticipantBtn: document.getElementById('save-participant-btn'),
            deleteParticipantBtn: document.getElementById('delete-participant-btn'),
            connectDeviceBtn: document.getElementById('connect-device-btn')
        };
    }

    /**
     * Legacy method for backward compatibility
     * @deprecated Use modules.systematic.deleteTrial instead
     */
    async deleteTrial(trialId) {
        if (this.modules.systematic) {
            await this.modules.systematic.deleteTrial(trialId);
        }
    }

    /**
     * Legacy method for backward compatibility
     * @deprecated Use modules.systematic.viewTrialDetails instead
     */
    viewTrialDetails(trialId) {
        if (this.modules.systematic) {
            this.modules.systematic.viewTrialDetails(trialId);
        }
    }




}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new CrutchApp();
});

