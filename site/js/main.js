document.addEventListener('DOMContentLoaded', function () {
    // By leaving this empty, the browser will make API requests to the same
    // origin that served the page, which eliminates all CORS issues.
    const SERVER_URL = '';

    // --- App Configuration ---
    // Metabolic cost removed from grid search mode

    // --- State Management ---
    let appState = {
        mode: null, // 'systematic' or 'bo'
        participants: [],
        currentParticipant: null,
        geometries: [],
        trials: [],
        boSession: {
            active: false,
            userId: null,
            history: [],
            suggestion: null,
            selectedObjective: null, // Added for BO
        }
    };

    // --- Element Selectors ---
    const screens = {
        modeSelection: document.getElementById('mode-selection-screen'),
        systematic: document.getElementById('systematic-screen'),
        bo: document.getElementById('bo-screen'),
    };

    const modeButtons = {
        selectSystematic: document.getElementById('select-systematic-mode'),
        selectBO: document.getElementById('select-bo-mode'),
    };

    // Elements for Systematic Mode
    const systematic = {
        participantSelect: document.getElementById('participant-select'),
        newParticipantForm: document.getElementById('new-participant-form'),
        saveParticipantBtn: document.getElementById('save-participant-btn'),
        createParticipantModal: new bootstrap.Modal(document.getElementById('create-participant-modal')),
        deleteParticipantBtn: document.getElementById('delete-participant-btn'),
        downloadDataBtn: document.getElementById('download-data-btn'),
        gridSearchTables: document.getElementById('grid-search-tables'),
        instabilityPlot3D: document.getElementById('instability-plot-3d'),
        participantTrialsTableBody: document.querySelector('#participant-trials-table tbody'),
        participantTrialsTitle: document.getElementById('participant-trials-title'),
        trialRunnerCol: document.getElementById('trial-runner-col'),
        trialRunnerTitle: document.getElementById('trial-runner-title'),
        trialForm: document.getElementById('systematic-trial-form'),
        connectDeviceBtn: document.getElementById('connect-device-btn'),
        deviceStatus: document.getElementById('device-status'),
        stopwatch: document.getElementById('stopwatch'),
        startStopBtn: document.getElementById('start-stop-btn'),
        plotsArea: document.getElementById('plots-area'),
        forcePlotDiv: document.getElementById('force-plot-div'),
        histPlotDiv: document.getElementById('hist-plot-div'),
        stepInteractionArea: document.getElementById('step-interaction-area'),
        stepList: document.getElementById('step-list'),
        stepCount: document.getElementById('step-count'),
        metricsAndSurveyArea: document.getElementById('metrics-and-survey-area'),
        instabilityLossValue: document.getElementById('instability-loss-value'),
        surveyArea: document.getElementById('survey-area'), // This is now part of the above
        uploadDataBtn: document.getElementById('upload-data-btn'),

        fileUploadInput: document.getElementById('file-upload-input'),
        discardTrialBtn: document.getElementById('discard-trial-btn'),
        closeTrialRunnerBtn: document.getElementById('close-trial-runner-btn'),

        // Elements for Trial Editing Modal
        editTrialModal: new bootstrap.Modal(document.getElementById('edit-trial-modal')),
        editTrialForm: document.getElementById('edit-trial-form'),
        editTrialModalLabel: document.getElementById('editTrialModalLabel'),
        editForcePlotDiv: document.getElementById('edit-force-plot-div'),
        editHistPlotDiv: document.getElementById('edit-hist-plot-div'),
        editStepInteractionArea: document.getElementById('edit-step-interaction-area'),
        editStepList: document.getElementById('edit-step-list'),
        editStepCount: document.getElementById('edit-step-count'),
        editInstabilityLossValue: document.getElementById('edit-instability-loss-value'),
        resaveTrialBtn: document.getElementById('resave-trial-btn'),
        editPlotsArea: document.getElementById('edit-plots-area'),
    };

    // --- Live Trial State & BLE ---
    const CHARACTERISTIC_UUID = '0000ffe1-0000-1000-8000-00805f9b34fb';
    let bleServer = null;
    let bleCharacteristic = null;
    let trialDataBuffer = [];
    let uploadedFile = null;

    let trialState = {
        timer: null,
        startTime: null,
        elapsed: 0,
        running: false,
        metrics: null, // To store calculated metrics
        steps: [], // To store editable step times
        rawData: null, // To store { force, timestamp } for recalculations
    };

    // Elements for BO Mode
    const bo = {
        participantSelect: document.getElementById('bo-participant-select'),
        startBtn: document.getElementById('start-bo-btn'),
        dashboard: document.getElementById('bo-dashboard'),
        userInfo: document.getElementById('bo-user-info'),
        suggestionBox: document.getElementById('bo-suggestion-box'),
        acceptBtn: document.getElementById('bo-accept-geometry'),
        rejectBtn: document.getElementById('bo-reject-geometry'),
        trialCard: document.getElementById('bo-trial-card'),
        trialForm: document.getElementById('bo-trial-form'),
        historyTable: document.querySelector('#bo-history-table tbody'),
        objectiveSelectCard: document.getElementById('bo-objective-select-card'),
        objectiveButtons: document.querySelectorAll('.bo-objective-btn'),
        createParticipantBtn: document.getElementById('bo-create-participant-btn'),
        deleteParticipantBtn: document.getElementById('bo-delete-participant-btn'),
        objectiveBackBtn: document.getElementById('bo-objective-back-btn'),
        
        // Pain BO Decision Card
        painDecisionCard: document.getElementById('pain-bo-decision-card'),
        painExistingDataSummary: document.getElementById('pain-existing-data-summary'),
        instabilityExistingDataSummary: document.getElementById('instability-existing-data-summary'),
        painContinueBtn: document.getElementById('pain-continue-btn'),
        painRestartBtn: document.getElementById('pain-restart-btn'),
        painDecisionBackBtn: document.getElementById('pain-decision-back-btn'),
    };

    // Elements for Pain Optimization
    const painOptimization = {
        screen: document.getElementById('pain-optimization-screen'),
        trialNumber: document.getElementById('pain-trial-number'),
        alphaDisplay: document.getElementById('pain-geometry-alpha'),
        betaDisplay: document.getElementById('pain-geometry-beta'),
        gammaDisplay: document.getElementById('pain-geometry-gamma'),
        testGeometryBtn: document.getElementById('pain-test-geometry-btn'),
        suggestAlternativeBtn: document.getElementById('pain-suggest-alternative-btn'),
        manualLossBtn: document.getElementById('pain-manual-loss-btn'),
        highLossCheckbox: document.getElementById('pain-high-loss-checkbox'),
        surveyCard: document.getElementById('pain-survey-card'),
        nrsScore: document.getElementById('pain-nrs-score'),
        nrsDisplay: document.getElementById('pain-nrs-display'),
        submitScoreBtn: document.getElementById('pain-submit-score-btn'),
        cancelTestBtn: document.getElementById('pain-cancel-test-btn'),
        firstGeometryCard: document.getElementById('pain-first-geometry-card'),
        firstAlpha: document.getElementById('pain-first-alpha'),
        firstBeta: document.getElementById('pain-first-beta'),
        firstGamma: document.getElementById('pain-first-gamma'),
        setFirstGeometryBtn: document.getElementById('pain-set-first-geometry-btn'),
        cancelFirstGeometryBtn: document.getElementById('pain-cancel-first-geometry-btn'),
        alternativeGeometryCard: document.getElementById('pain-alternative-geometry-card'),
        altAlpha: document.getElementById('pain-alt-alpha'),
        altBeta: document.getElementById('pain-alt-beta'),
        altGamma: document.getElementById('pain-alt-gamma'),
        setAlternativeGeometryBtn: document.getElementById('pain-set-alternative-geometry-btn'),
        cancelAlternativeGeometryBtn: document.getElementById('pain-cancel-alternative-geometry-btn'),
        manualLossSection: document.getElementById('pain-manual-loss-section'),
        manualLossInput: document.getElementById('pain-manual-loss-input'),
        historyTable: document.getElementById('pain-history-table'),
        historyTbody: document.getElementById('pain-history-tbody'),
        noHistory: document.getElementById('pain-no-history'),
        progressCard: document.getElementById('pain-progress-card'),
        bestScore: document.getElementById('pain-best-score'),
        totalTrials: document.getElementById('pain-total-trials'),

        exitBtn: document.getElementById('pain-exit-btn')
    };

    // Elements for Instability Optimization
    const instabilityOptimization = {
        screen: document.getElementById('instability-optimization-screen'),
        trialNumber: document.getElementById('instability-trial-number'),
        alphaDisplay: document.getElementById('instability-geometry-alpha'),
        betaDisplay: document.getElementById('instability-geometry-beta'),
        gammaDisplay: document.getElementById('instability-geometry-gamma'),
        connectionStatus: document.getElementById('instability-connection-status'),
        connectionText: document.getElementById('instability-connection-text'),
        connectBtn: document.getElementById('instability-connect-btn'),
        startStopBtn: document.getElementById('instability-start-stop-btn'),
        stopwatch: document.getElementById('instability-stopwatch'),
        plotsArea: document.getElementById('instability-plots-area'),
        forcePlotDiv: document.getElementById('instability-force-plot-div'),
        histPlotDiv: document.getElementById('instability-hist-plot-div'),
        resultsCard: document.getElementById('instability-results-card'),
        lossDisplay: document.getElementById('instability-loss-display'),
        stepsCount: document.getElementById('instability-steps-count'),
        stepsList: document.getElementById('instability-steps-list'),
        testGeometryBtn: document.getElementById('instability-test-geometry-btn'),
        suggestAlternativeBtn: document.getElementById('instability-suggest-alternative-btn'),
        manualLossBtn: document.getElementById('instability-manual-loss-btn'),
        manualLossSection: document.getElementById('instability-manual-loss-section'),
        manualLossInput: document.getElementById('instability-manual-loss-input'),
        alternativeGeometryCard: document.getElementById('instability-alternative-geometry-card'),
        altAlphaInput: document.getElementById('instability-alt-alpha'),
        altBetaInput: document.getElementById('instability-alt-beta'),
        altGammaInput: document.getElementById('instability-alt-gamma'),
        setAlternativeGeometryBtn: document.getElementById('instability-set-alternative-geometry-btn'),
        cancelAlternativeGeometryBtn: document.getElementById('instability-cancel-alternative-geometry-btn'),
        showSurveyBtn: document.getElementById('instability-show-survey-btn'),
        surveyCard: document.getElementById('instability-survey-card'),
        surveyForm: document.getElementById('instability-survey-form'),
        surveyCancelBtn: document.getElementById('instability-survey-cancel-btn'),
        historyTbody: document.getElementById('instability-history-tbody'),
        bestLoss: document.getElementById('instability-best-loss'),
        totalTrials: document.getElementById('instability-total-trials'),
        suggestNextBtn: document.getElementById('instability-suggest-next-btn'),
        exitBtn: document.getElementById('instability-exit-btn')
    };

    // Elements for Effort Optimization (identical to Pain)
    const effortOptimization = {
        screen: document.getElementById('effort-optimization-screen'),
        alphaDisplay: document.getElementById('effort-geometry-alpha'),
        betaDisplay: document.getElementById('effort-geometry-beta'),
        gammaDisplay: document.getElementById('effort-geometry-gamma'),
        testGeometryBtn: document.getElementById('effort-test-geometry-btn'),
        suggestAlternativeBtn: document.getElementById('effort-suggest-alternative-btn'),
        manualLossBtn: document.getElementById('effort-manual-loss-btn'),
        manualLossSection: document.getElementById('effort-manual-loss-section'),
        manualLossInput: document.getElementById('effort-manual-loss-input'),
        alternativeGeometryCard: document.getElementById('effort-alternative-geometry-card'),
        altAlphaInput: document.getElementById('effort-alt-alpha'),
        altBetaInput: document.getElementById('effort-alt-beta'),
        altGammaInput: document.getElementById('effort-alt-gamma'),
        setAlternativeGeometryBtn: document.getElementById('effort-set-alternative-geometry-btn'),
        cancelAlternativeGeometryBtn: document.getElementById('effort-cancel-alternative-geometry-btn'),
        firstGeometryCard: document.getElementById('effort-first-geometry-card'),
        firstAlphaInput: document.getElementById('effort-first-alpha'),
        firstBetaInput: document.getElementById('effort-first-beta'),
        firstGammaInput: document.getElementById('effort-first-gamma'),
        setFirstGeometryBtn: document.getElementById('effort-set-first-geometry-btn'),
        cancelFirstGeometryBtn: document.getElementById('effort-cancel-first-geometry-btn'),
        surveyCard: document.getElementById('effort-survey-card'),
        surveyForm: document.getElementById('effort-survey-form'),
        surveyCancelBtn: document.getElementById('effort-survey-cancel-btn'),
        historyTbody: document.getElementById('effort-history-tbody'),
        bestScore: document.getElementById('effort-best-score'),
        totalTrials: document.getElementById('effort-total-trials'),

        exitBtn: document.getElementById('effort-exit-btn')
    };

    // Pain optimization state
    let painState = {
        active: false,
        userId: null,
        currentGeometry: null,
        trials: [],
        currentTrialNumber: 1
    };

    // Instability optimization state (same structure as trialState)
    let instabilityState = {
        active: false,
        userId: null,
        currentGeometry: null,
        trials: [],
        currentTrialNumber: 1,
        isConnected: false,
        running: false,  // Changed from isCollecting to match Grid Search
        collectedData: [],
        analysisResults: null,
        // Timer variables (same as Grid Search)
        timer: null,
        startTime: null,
        elapsed: 0,
        dataParser: null,
        bleCharacteristic: null,
        bleServer: null,
        existingData: null,
        restartMode: false,
        steps: [], // For step editing like Grid Search
        rawData: null
    };

    // Effort optimization state (identical to Pain)
    let effortState = {
        active: false,
        userId: null,
        currentGeometry: null,
        trials: [],
        currentTrialNumber: 1
    };


    // --- Generic Helper Functions ---
    async function apiRequest(endpoint, method = 'GET', body = null) {
        const options = {
            method,
            headers: { 'Content-Type': 'application/json' },
        };
        if (body) {
            options.body = JSON.stringify(body);
        }
        const response = await fetch(`${SERVER_URL}${endpoint}`, options);
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || `HTTP error! Status: ${response.status}`);
        }
        return response.json();
    }

    function showScreen(screenName) {
        console.log(`showScreen called with: ${screenName}`);
        
        // Hide all screens first
        const screens = document.querySelectorAll('.screen');
        console.log(`Found ${screens.length} screens to hide`);
        screens.forEach(screen => {
            screen.classList.add('d-none');
        });
        
        // End any active sessions when going to home
        if (screenName === 'home') {
            endAllActiveSessions();
        }
        
        // Show the selected screen based on screenName
        let targetScreen;
        switch (screenName) {
            case 'modeSelection':
                targetScreen = document.getElementById('mode-selection-screen');
                break;
            case 'systematic':
                targetScreen = document.getElementById('systematic-screen');
                break;
            case 'bo':
                targetScreen = document.getElementById('bo-screen');
                break;
            case 'effortOptimization':
                targetScreen = document.getElementById('effort-optimization-screen');
                // Also show the parent bo-screen
                const boScreen = document.getElementById('bo-screen');
                if (boScreen) {
                    boScreen.classList.remove('d-none');
                }
                break;
            default:
                targetScreen = document.getElementById(`${screenName}-screen`);
        }
        
        console.log(`Target screen element:`, targetScreen);
        console.log(`Target screen classes:`, targetScreen ? targetScreen.className : 'null');
        console.log(`Target screen style:`, targetScreen ? targetScreen.style.display : 'null');
        
        if (targetScreen) {
            targetScreen.classList.remove('d-none');
            console.log(`Removed d-none class. New classes:`, targetScreen.className);
            console.log(`New style display:`, targetScreen.style.display);
            
            // Additional debugging for visibility
            console.log(`Element offsetHeight:`, targetScreen.offsetHeight);
            console.log(`Element offsetWidth:`, targetScreen.offsetWidth);
            console.log(`Element getBoundingClientRect:`, targetScreen.getBoundingClientRect());
            console.log(`Element computed style display:`, window.getComputedStyle(targetScreen).display);
            console.log(`Element computed style visibility:`, window.getComputedStyle(targetScreen).visibility);
            console.log(`Element computed style opacity:`, window.getComputedStyle(targetScreen).opacity);
            
            // Check parent elements
            let parent = targetScreen.parentElement;
            let level = 0;
            while (parent && level < 5) {
                console.log(`Parent level ${level}:`, parent.tagName, parent.className, parent.style.display);
                parent = parent.parentElement;
                level++;
            }
        } else {
            console.error(`Screen not found: ${screenName}`);
        }
    }
    
    function updateHomeButtonState() {
        const homeButton = document.getElementById('home-button');
        const isOptimizationActive = painState.active || effortState.active || instabilityState.active;
        
        if (isOptimizationActive) {
            homeButton.style.cursor = 'not-allowed';
            homeButton.style.opacity = '0.6';
            homeButton.title = 'Please exit optimization session first';
        } else {
            homeButton.style.cursor = 'pointer';
            homeButton.style.opacity = '1';
            homeButton.title = 'Go to home';
        }
    }

    function endAllActiveSessions() {
        // End BO session
        if (appState.boSession.active) {
            appState.boSession = { active: false, userId: null, history: [], suggestion: null, selectedObjective: null };
        }
        
        // End pain optimization session
        if (painState.active) {
            painState = {
                active: false,
                userId: null,
                currentGeometry: null,
                trials: [],
                currentTrialNumber: 1,
                restartMode: false
            };
        }
        
        // End instability optimization session
        if (instabilityState.active) {
            instabilityState = {
                active: false,
                userId: null,
                currentGeometry: null,
                trials: [],
                currentTrialNumber: 1,
                isConnected: false,
                isCollecting: false,
                collectedData: [],
                analysisResults: null
            };
        }
        
        // End effort optimization session
        if (effortState.active) {
            effortState = {
                active: false,
                userId: null,
                currentGeometry: null,
                trials: [],
                currentTrialNumber: 1
            };
        }
        
        // Reset UI to initial state
        resetBOUI();
        resetPainOptimizationUI();
        resetInstabilityOptimizationUI();
        resetEffortOptimizationUI();
        
        console.log("All active sessions ended");
    }
    
    function resetBOUI() {
        // Hide BO dashboard and show setup
        bo.dashboard.classList.add('d-none');
        document.getElementById('bo-setup-card').classList.remove('d-none');
        bo.objectiveSelectCard.classList.add('d-none');
        bo.painDecisionCard.classList.add('d-none');
        bo.userInfo.classList.add('d-none');
        document.getElementById('bo-main-flow').classList.add('d-none');
        
        // Reset participant selection
        bo.participantSelect.value = '';
        bo.deleteParticipantBtn.disabled = true;
        bo.participantSelect.parentElement.classList.remove('d-none');
    }
    
    function resetPainOptimizationUI() {
        // Hide pain optimization screen
        painOptimization.screen.classList.add('d-none');
        
        // Reset pain optimization state
        painOptimization.surveyCard.style.display = 'none';
        painOptimization.highLossCheckbox.checked = false;
        painOptimization.nrsScore.value = 0;
        painOptimization.nrsDisplay.textContent = '0';
        hideManualLossInput();
        hideFirstGeometryInput();
        hideAlternativeGeometryInput();
    }

    function resetInstabilityOptimizationUI() {
        // Hide instability optimization screen
        instabilityOptimization.screen.classList.add('d-none');
        
        // Reset instability optimization state
        instabilityOptimization.surveyCard.classList.add('d-none');
        instabilityOptimization.resultsCard.classList.add('d-none');
        instabilityOptimization.suggestNextBtn.classList.add('d-none');
        instabilityOptimization.collectionStatus.classList.add('d-none');
        
        // Hide decision card and geometry actions
        document.getElementById('instability-bo-decision-card').classList.add('d-none');
        document.getElementById('instability-geometry-actions').style.display = 'none';
        
        // Clear any active intervals
        if (instabilityState.collectionInterval) {
            clearInterval(instabilityState.collectionInterval);
        }
        if (instabilityState.timer) {
            clearInterval(instabilityState.timer);
            instabilityState.timer = null;
        }
    }

    function resetEffortOptimizationUI() {
        // Hide effort optimization screen
        effortOptimization.screen.classList.add('d-none');
        
        // Reset effort optimization state
        effortOptimization.surveyCard.style.display = 'none';
        hideEffortManualLossInput();
        hideEffortAlternativeGeometryInput();
        
        // Hide decision card
        document.getElementById('effort-bo-decision-card').classList.add('d-none');
    }

    function showNotification(message, type = 'success') {
        const toastContainer = document.querySelector('.toast-container');
        const toastEl = document.createElement('div');
        toastEl.className = `toast align-items-center text-white bg-${type} border-0`;
        toastEl.innerHTML = `<div class="d-flex"><div class="toast-body">${message}</div><button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button></div>`;
        toastContainer.appendChild(toastEl);
        const toast = new bootstrap.Toast(toastEl, { delay: 3000 });
        toast.show();
        toastEl.addEventListener('hidden.bs.toast', () => toastEl.remove());
    }

    function showTrialManagementScreen(participant) {
        showScreen('systematic');
        
        // Update header
        systematic.participantTrialsTitle.textContent = `${participant.name}'s Trial Management`;
        const characteristics = participant.characteristics || {};
        const details = `Age: ${characteristics.age || 'N/A'}, Height: ${characteristics.height || 'N/A'} cm, Weight: ${characteristics.weight || 'N/A'} kg`;
        systematic.trialRunnerTitle.textContent = details;
        
        // Load participant data
        renderParticipantTrialsTable(participant.id);
        
        // Load geometries
        renderRemainingGeometries(appState.geometries);
    }

    // --- Initial Load ---
    async function loadInitialData() {
        try {
            appState.participants = await apiRequest('/api/participants');
            appState.geometries = await apiRequest('/api/geometries');
            appState.trials = await apiRequest('/api/trials');
            populateParticipantSelects();
            renderParticipantTrialsTable(null);
        } catch (error) {
            showNotification(`Failed to load initial data: ${error.message}`, 'danger');
        }
    }

    // --- Initial UI Setup based on Config ---
    // Metabolic cost removed from grid search mode


    function populateParticipantSelects() {
        // Populate for participant selection screen and BO mode
        [systematic.participantSelect, bo.participantSelect].forEach(select => {
            select.innerHTML = '<option selected disabled>Choose...</option>';
            appState.participants.forEach(p => {
                const option = document.createElement('option');
                option.value = p.id;
                option.textContent = p.name;
                select.appendChild(option);
            });
        });
    }

    // --- Mode Selection Logic ---
    modeButtons.selectSystematic.addEventListener('click', () => {
        appState.mode = 'systematic';
        showScreen('systematic');
    });

    modeButtons.selectBO.addEventListener('click', () => {
        appState.mode = 'bo';
        showScreen('bo');
    });

    // --- Home Button Logic ---
    document.getElementById('home-button').addEventListener('click', () => {
        // Check if any optimization session is active
        if (painState.active || effortState.active || instabilityState.active) {
            showNotification('Please exit the current optimization session before going home.', 'warning');
            return;
        }
        
        appState.mode = null;
        showScreen('modeSelection');
    });

    // --- Participant Selection Logic ---
    systematic.participantSelect.addEventListener('change', async (e) => {
        const participantId = e.target.value;
        systematic.deleteParticipantBtn.disabled = !participantId;
        systematic.downloadDataBtn.disabled = !participantId;
        
        if (!participantId) {
            systematic.gridSearchTables.innerHTML = `<div class="text-center text-muted">Select a participant to see grid search trials.</div>`;
            systematic.instabilityPlot3D.innerHTML = `<div class="text-center text-muted pt-5">Select a participant to view the 3D plot.</div>`;
            renderParticipantTrialsTable(null);
            return;
        }
        
        await refreshParticipantView(participantId);
    });

    systematic.downloadDataBtn.addEventListener('click', async () => {
        const participantId = appState.currentParticipant?.id;
        if (!participantId) {
            showNotification('No participant selected.', 'warning');
            return;
        }

        showNotification('Preparing your download... This may take a moment.', 'info');

        try {
            const response = await apiRequest(`/api/participants/${participantId}/download`);
            if (response.download_url) {
                // Create a temporary link to trigger the download
                const link = document.createElement('a');
                link.href = `${SERVER_URL}${response.download_url}`;
                link.download = response.download_url.split('/').pop(); // Suggest a filename
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                showNotification('Download started!', 'success');
            } else {
                throw new Error('Download URL not found in response.');
            }
        } catch (error) {
            showNotification(`Failed to prepare download: ${error.message}`, 'danger');
        }
    });

    async function refreshParticipantView(participantId) {
        try {
            const data = await apiRequest(`/api/participants/${participantId}`);
            appState.currentParticipant = data.participant;
            displayParticipantDetails(data.participant);
            renderParticipantTrialsTable(participantId);
            renderRemainingGeometries(data.all_geometries);
            renderInstabilityPlot(data.instability_plot_data);
        } catch (error) {
            showNotification(`Error fetching participant details: ${error.message}`, 'danger');
        }
    }

    systematic.saveParticipantBtn.addEventListener('click', async () => {
        const name = systematic.newParticipantForm.querySelector('#new-participant-name').value.trim();
        if (!name) {
            showNotification('Name is required.', 'danger');
            return;
        }

        const payload = {
            name: name,
            userCharacteristics: {
                age: parseInt(document.getElementById('char-age').value) || null,
                sex: document.getElementById('char-sex').value,
                height: parseFloat(document.getElementById('char-height').value) || null,
                weight: parseFloat(document.getElementById('char-weight').value) || null,
                forearm_length: parseFloat(document.getElementById('char-forearm').value) || null,
                activity_level: document.getElementById('char-activity').value,
                previous_crutch_experience: document.querySelector('input[name="crutch-experience"]:checked').value === 'true',
            }
        };

        try {
            const newParticipant = await apiRequest('/api/participants', 'POST', payload);
            showNotification(`Participant "${newParticipant.name}" created successfully!`, 'success');

            systematic.createParticipantModal.hide();
            systematic.newParticipantForm.reset();

            // Refresh participant list, select the new one, and trigger the change event
            await loadInitialData();
            systematic.participantSelect.value = newParticipant.id;
            systematic.participantSelect.dispatchEvent(new Event('change'));

        } catch (error) {
            showNotification(`Error creating participant: ${error.message}`, 'danger');
        }
    });

    systematic.deleteParticipantBtn.addEventListener('click', async () => {
        const participantId = systematic.participantSelect.value;
        if (!participantId) return;
        
        if (confirm('Are you sure you want to delete this participant? This action cannot be undone.')) {
            try {
                await apiRequest(`/api/participants/${participantId}`, 'DELETE');
                showNotification('Participant deleted successfully', 'success');
                
                // Refresh the page to update all data
                location.reload();
            } catch (error) {
                showNotification(`Error deleting participant: ${error.message}`, 'danger');
            }
        }
    });

    // --- Systematic Mode Logic ---
    function displayParticipantDetails(participant) {
        const detailsFooter = document.getElementById('participant-details-footer');
        const detailsTableBody = document.getElementById('participant-details-table');
        detailsTableBody.innerHTML = ''; // Clear previous details

        if (!participant || !participant.characteristics) {
            detailsFooter.classList.add('d-none');
            return;
        }

        const chars = participant.characteristics;
        const detailsMap = {
            "Age": chars.age,
            "Sex": chars.sex,
            "Height": chars.height ? `${chars.height} cm` : 'N/A',
            "Weight": chars.weight ? `${chars.weight} kg` : 'N/A',
            "Forearm Length": chars.forearm_length ? `${chars.forearm_length} cm` : 'N/A',
            "Activity Level": chars.activity_level,
            "Crutch Experience": chars.previous_crutch_experience ? 'Yes' : 'No'
        };

        for (const [key, value] of Object.entries(detailsMap)) {
            const row = detailsTableBody.insertRow();
            row.innerHTML = `<th class="text-muted">${key}</th><td>${value ?? 'N/A'}</td>`;
        }

        detailsFooter.classList.remove('d-none');
    }

    function renderParticipantTrialsTable(participantId) {
        systematic.participantTrialsTableBody.innerHTML = '';
        
        if (!participantId) {
            systematic.participantTrialsTitle.textContent = 'Participant Trials';
            systematic.participantTrialsTableBody.innerHTML = `<tr><td colspan="13" class="text-center text-muted">Select a participant to see their trials.</td></tr>`;
            return;
        }
        
        // Ensure participantId is a number for proper comparison
        const numericParticipantId = parseInt(participantId);
        const participant = appState.participants.find(p => p.id === numericParticipantId);
        // Filter to only show grid_search trials (exclude Pain BO and other BO trials)
        const participantTrials = appState.trials.filter(t => 
            t.participant_id === numericParticipantId && 
            t.source === 'grid_search'
        );
        
        if (!participant) {
            systematic.participantTrialsTitle.textContent = 'Participant Trials';
            systematic.participantTrialsTableBody.innerHTML = `<tr><td colspan="13" class="text-center text-muted">Participant not found.</td></tr>`;
            return;
        }
        
        systematic.participantTrialsTitle.textContent = `${participant.name}'s Trials`;
        
        if (participantTrials.length === 0) {
            systematic.participantTrialsTableBody.innerHTML = `<tr><td colspan="13" class="text-center text-muted">No Grid Search trials recorded yet for this participant.</td></tr>`;
            return;
        }

        // Sort trials: Control first, then by G-number
        participantTrials.sort((a, b) => {
            const aIsControl = a.geometry_name === 'Control';
            const bIsControl = b.geometry_name === 'Control';
            if (aIsControl && !bIsControl) return -1;
            if (!aIsControl && bIsControl) return 1;

            // Handle cases where geometry_name might be undefined
            const aGeomName = a.geometry_name || '';
            const bGeomName = b.geometry_name || '';
            
            // If both have G-numbers, sort by number
            if (aGeomName.startsWith('G') && bGeomName.startsWith('G')) {
                const aNum = parseInt(aGeomName.substring(1)) || 0;
                const bNum = parseInt(bGeomName.substring(1)) || 0;
                return aNum - bNum;
            }
            
            // Fallback to alphabetical sorting
            return aGeomName.localeCompare(bGeomName);
        });
        
        participantTrials.forEach((t, index) => {
            const row = document.createElement('tr');
            row.dataset.trialId = t.id;
            row.classList.add('clickable-trial-row'); // Add class to make it clickable
            const trialDate = new Date(t.timestamp);
            const today = new Date();
            const isToday = trialDate.toDateString() === today.toDateString();
            
            // Calculate trial number: Control shows as "Control", others start from 1
            let trialNumber;
            if (t.geometry_name === 'Control') {
                trialNumber = 'Control';
            } else {
                // Count non-control trials before this one to get the correct number
                const nonControlTrialsBefore = participantTrials.slice(0, index).filter(trial => trial.geometry_name !== 'Control').length;
                trialNumber = nonControlTrialsBefore + 1;
            }
            
            row.innerHTML = `
                <td>${trialNumber}</td>
                <td>${t.geometry_name || 'Unknown'}</td>
                <td>${t.alpha ?? '-'}°</td>
                <td>${t.beta ?? '-'}°</td>
                <td>${t.gamma ?? '-'}°</td>
                <td>${isToday ? 'Today' : trialDate.toLocaleDateString()}</td>
                <td>${trialDate.toLocaleTimeString()}</td>
                <td>${Number(t.processed_features?.step_count ?? NaN) || '-'}</td>
                <td>${t.processed_features?.instability_loss !== undefined ? Number(t.processed_features.instability_loss).toFixed(4) : '-'}</td>
                <td>${t.survey_responses?.sus_score !== undefined ? Number(t.survey_responses.sus_score).toFixed(2) : '-'}</td>
                <td>${t.survey_responses?.nrs_score !== undefined ? t.survey_responses.nrs_score : '-'}</td>
                <td>${t.survey_responses?.tlx_score !== undefined ? Number(t.survey_responses.tlx_score).toFixed(2) : '-'}</td>
                <td>${t.survey_responses?.metabolic_cost !== undefined ? Number(t.survey_responses.metabolic_cost).toFixed(2) : '-'}</td>
                <td class="text-center">
                    <button class="btn btn-sm btn-outline-danger delete-trial-btn" data-trial-id="${t.id}" data-geometry-id="${t.geometry_id}">
                        🗑️
                    </button>
                </td>
            `;
            systematic.participantTrialsTableBody.appendChild(row);
        });
    }

    async function openTrialForEditing(trialId) {
        try {
            const results = await apiRequest(`/api/trials/${trialId}/details`);
            const trialInfo = results.trial_info;
            
            // Populate the modal's hidden fields
            systematic.editTrialForm.querySelector('#edit-trial-id').value = trialId;
            systematic.editTrialForm.querySelector('#edit-participant-id').value = trialInfo.participant_id;
            systematic.editTrialForm.querySelector('#edit-geometry-id').value = trialInfo.geometry_id;

            // Update modal title
            systematic.editTrialModalLabel.textContent = `Editing Trial: ${trialInfo.geometry_name} for ${trialInfo.participant_name}`;

            // Store data in the global trialState for consistency with existing functions
            trialState.metrics = results.metrics;
            trialState.steps = results.steps.sort((a, b) => a - b);
            trialState.rawData = results.processed_data;

            // Use a new function to render plots and steps inside the modal
            await renderEditView(results);
            
            systematic.editTrialModal.show();

        } catch (error) {
            showNotification(`Error loading trial for editing: ${error.message}`, 'danger');
        }
    }
    
    // This is similar to handleAnalysisUpdate but targets the modal elements
    async function renderEditView(results) {
        // --- Render Plots ---
        async function renderPlot(plotDiv, plotPath) {
            const plotResponse = await fetch(`${SERVER_URL}${plotPath}?t=${new Date().getTime()}`);
            if (!plotResponse.ok) throw new Error(`Failed to fetch plot: ${plotPath}`);
            const plotHtml = await plotResponse.text();
            const tempDiv = document.createElement('div');
            tempDiv.innerHTML = plotHtml;
            const plotlyGraphDiv = tempDiv.querySelector('.plotly-graph-div');
            if (plotlyGraphDiv) {
                const plotData = JSON.parse(plotlyGraphDiv.dataset.raw);
                Plotly.newPlot(plotDiv, plotData.data, plotData.layout, { responsive: true });
            } else {
                plotDiv.innerHTML = plotHtml;
            }
        }

        await renderPlot(systematic.editForcePlotDiv, results.plots.timeseries);
        await renderPlot(systematic.editHistPlotDiv, results.plots.histogram);
        
        // --- Attach Listeners ---
        // Detach old listeners first
        if (systematic.editForcePlotDiv.removeListener) {
            systematic.editForcePlotDiv.removeListener('plotly_click', onEditPlotClick);
        }
        systematic.editForcePlotDiv.on('plotly_click', onEditPlotClick);

        // --- Display Metrics and Step List ---
        systematic.editInstabilityLossValue.textContent = results.metrics.instability_loss?.toFixed(4) ?? 'N/A';
        renderEditStepList();
    }

    function renderEditStepList() {
        systematic.editStepList.innerHTML = '';
        systematic.editStepCount.textContent = trialState.steps.length;
        
        trialState.steps.forEach((stepTime, index) => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${stepTime.toFixed(3)}s</td>
                <td class="text-end">
                    <button type="button" class="btn btn-sm btn-outline-danger py-0 px-1 delete-edit-step-btn" 
                            data-index="${index}" title="Delete">
                        <i class="fas fa-trash-alt"></i>
                    </button>
                </td>
            `;
            systematic.editStepList.appendChild(row);
        });
    }

    // A separate click handler for the edit plot
    function onEditPlotClick(data) {
        const point = data.points[0];
        if (point.curveNumber !== 0) return;
    
        const clickedTime = point.x;
        if (trialState.steps.some(step => Math.abs(step - clickedTime) < 0.1)) {
            showNotification("A step already exists near this time.", "warning");
            return;
        }
    
        trialState.steps.push(clickedTime);
        trialState.steps.sort((a, b) => a - b);
    
        requestEditRecalculation();
    }

    async function requestEditRecalculation() {
        const participantId = parseInt(systematic.editTrialForm.querySelector('#edit-participant-id').value);
        const geometryId = parseInt(systematic.editTrialForm.querySelector('#edit-geometry-id').value);
    
        systematic.editPlotsArea.style.opacity = '0.5';
    
        try {
            const payload = { participantId, geometryId, steps: trialState.steps };
            const results = await apiRequest('/api/trials/recalculate', 'POST', payload);
            
            // Update the global state with new metrics
            trialState.metrics = results.metrics;
            trialState.steps = results.steps.sort((a, b) => a-b);
            
            // Re-render the edit view with the new data
            await renderEditView(results);

        } catch (error) {
            showNotification(`Recalculation failed: ${error.message}`, 'danger');
        } finally {
            systematic.editPlotsArea.style.opacity = '1';
        }
    }


    async function deleteTrial(trialId, geometryId) {
        try {
            await apiRequest(`/api/trials/${trialId}`, 'DELETE');
            showNotification('Trial deleted successfully', 'success');
            
            // Force a reload of all global data to ensure UI is in sync
            await loadInitialData();
            
            // Refresh the entire view for the currently selected participant
            if (appState.currentParticipant) {
                await refreshParticipantView(appState.currentParticipant.id);
            }

        } catch (error) {
            showNotification(`Error deleting trial: ${error.message}`, 'danger');
        }
    }

    function renderRemainingGeometries(allGeometries) {
        const gridSearchTables = document.getElementById('grid-search-tables');
        gridSearchTables.innerHTML = '';
        
        if (!appState.currentParticipant) {
            gridSearchTables.innerHTML = `<div class="text-center text-muted">No participant selected.</div>`;
            return;
        }
        
        if (allGeometries.length === 0) {
            gridSearchTables.innerHTML = `<div class="text-center text-muted">No geometries available!</div>`;
            return;
        }

        // Get completed trials for this participant
        const completedTrials = appState.trials.filter(t => t.participant_id === appState.currentParticipant?.id);
        const completedGeometryIds = new Set(completedTrials.map(t => t.geometry_id));

        // Group geometries by gamma value
        const gammaGroups = {};
        const controlGroups = {};
        allGeometries.forEach(g => {
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
            gridSearchTables.appendChild(controlContainer);
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
                                                <button class="btn btn-sm btn-primary" data-geom-id="${geom.id}">
                                                    ${geom.name}<br><small>Data Collection</small>
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
            gridSearchTables.appendChild(tableContainer);
        });
    }

    // Event delegation for grid search tables and delete trial buttons
    document.addEventListener('click', (e) => {
        // Handle grid search table buttons
        if (e.target.closest('#grid-search-tables') && e.target.tagName === 'BUTTON' && e.target.dataset.geomId) {
            const geomId = e.target.dataset.geomId;
            const geometry = appState.geometries.find(g => g.id == geomId);

            // Show the trial runner
            systematic.trialRunnerCol.classList.remove('d-none');
            systematic.trialRunnerTitle.textContent = `Run Trial: ${geometry.name}`;

            // Set hidden inputs in the form
            systematic.trialForm.querySelector('#systematic-participant-id').value = appState.currentParticipant.id;
            systematic.trialForm.querySelector('#systematic-geometry-id').value = geomId;

            resetTrialState();
        }
        
        // Handle delete trial buttons
        if (e.target.closest('.delete-trial-btn')) {
            const trialId = e.target.closest('.delete-trial-btn').dataset.trialId;
            const geometryId = e.target.closest('.delete-trial-btn').dataset.geometryId;
            
            if (confirm('Are you sure you want to delete this trial? This action cannot be undone.')) {
                deleteTrial(trialId, geometryId);
            }
        }
        // Handle clicking on a trial row to edit
        else if (e.target.closest('.clickable-trial-row')) {
            const trialId = e.target.closest('.clickable-trial-row').dataset.trialId;
            openTrialForEditing(trialId);
        }
    });

    systematic.editStepList.addEventListener('click', (e) => {
        const target = e.target.closest('button');
        if (!target) return;

        if (target.classList.contains('delete-edit-step-btn')) {
            const index = parseInt(target.dataset.index, 10);
            trialState.steps.splice(index, 1);
            requestEditRecalculation();
        }
    });

    systematic.resaveTrialBtn.addEventListener('click', async () => {
        const trialId = parseInt(systematic.editTrialForm.querySelector('#edit-trial-id').value);
        if (!trialId) {
            showNotification("No trial ID found to save.", 'danger');
            return;
        }

        const payload = {
            metrics: trialState.metrics,
            steps: trialState.steps,
            // For now, survey responses are not editable in this view, so we don't send them
        };

        try {
            await apiRequest(`/api/trials/${trialId}`, 'PUT', payload);
            showNotification('Trial updated successfully!', 'success');
            systematic.editTrialModal.hide();
            
            // Refresh the entire view for the currently selected participant
            await loadInitialData();
            if (appState.currentParticipant) {
                await refreshParticipantView(appState.currentParticipant.id);
            }
        } catch (error) {
            showNotification(`Failed to resave trial: ${error.message}`, 'danger');
        }
    });

    function resetTrialState() {
        // Reset buttons
        systematic.connectDeviceBtn.disabled = false;
        systematic.startStopBtn.disabled = true; // Disabled until device is connected
        systematic.startStopBtn.innerHTML = '<i class="fas fa-play"></i> Start Trial';
        systematic.startStopBtn.classList.replace('btn-danger', 'btn-success');
    
        // Reset status
        systematic.deviceStatus.textContent = 'Status: Disconnected';
        systematic.deviceStatus.classList.remove('alert-success');
        systematic.deviceStatus.classList.add('alert-secondary');
        // Reset stopwatch
        stopStopwatch();
        trialState.elapsed = 0;
        updateStopwatchDisplay();
        // Hide plots and survey
        systematic.plotsArea.classList.add('d-none');
        systematic.metricsAndSurveyArea.classList.add('d-none');
        Plotly.purge(systematic.forcePlotDiv); // Clear the plot
        Plotly.purge(systematic.histPlotDiv); // Clear the plot
        systematic.instabilityLossValue.textContent = '-';
        systematic.stepInteractionArea.classList.add('d-none');
        systematic.stepList.innerHTML = '';
        trialState.metrics = null;
        trialState.steps = [];
        trialState.rawData = null;
        trialDataBuffer = []; // Clear the data buffer
        // Hide discard buttons
        systematic.discardTrialBtn.style.display = 'none';

        // Detach any existing plot listeners to prevent memory leaks
        if (systematic.forcePlotDiv.removeListener) {
            systematic.forcePlotDiv.removeListener('plotly_click', onPlotClick);
        }
    }

    // --- Step Editing & Recalculation (Backend Driven) ---

    async function handleAnalysisUpdate(results) {
        try {
            showNotification(results.message, 'info');
    
            // --- Store server results in trialState ---
            trialState.metrics = results.metrics;
            trialState.steps = results.steps.sort((a, b) => a - b);
            // rawData is only sent on the first analysis, not on recalculation.
            // So we only update it if it's in the response.
            if (results.processed_data) {
                trialState.rawData = results.processed_data;
            }
    
            // --- Render Plots ---
            async function renderPlot(plotDiv, plotPath) {
                // Add a cache-busting query parameter
                const plotResponse = await fetch(`${SERVER_URL}${plotPath}?t=${new Date().getTime()}`);
                if (!plotResponse.ok) throw new Error(`Failed to fetch plot: ${plotPath}`);
                const plotHtml = await plotResponse.text();
                const tempDiv = document.createElement('div');
                tempDiv.innerHTML = plotHtml;
                const plotlyGraphDiv = tempDiv.querySelector('.plotly-graph-div');
                if (plotlyGraphDiv) {
                    const plotData = JSON.parse(plotlyGraphDiv.dataset.raw);
                    Plotly.newPlot(plotDiv, plotData.data, plotData.layout, { responsive: true });
                } else {
                    plotDiv.innerHTML = plotHtml; // Fallback for simple HTML
                }
            }
    
            await renderPlot(systematic.forcePlotDiv, results.plots.timeseries);
            await renderPlot(systematic.histPlotDiv, results.plots.histogram);
    
            // Re-attach the plotly click listener now that the plot exists
            // First, remove any old listener to avoid duplicates
            if (systematic.forcePlotDiv.removeListener) {
                systematic.forcePlotDiv.removeListener('plotly_click', onPlotClick);
            }
            systematic.forcePlotDiv.on('plotly_click', onPlotClick);
    
            // --- Display Metrics and Step List ---
            systematic.instabilityLossValue.textContent = results.metrics.instability_loss?.toFixed(4) ?? 'N/A';
            renderStepList();
    
            // --- Show UI Elements ---
            systematic.stepInteractionArea.classList.remove('d-none');
            systematic.plotsArea.classList.remove('d-none');
            systematic.metricsAndSurveyArea.classList.remove('d-none');
            systematic.discardTrialBtn.style.display = 'inline-block';
    
        } catch (error) {
            showNotification(`Failed to update analysis view: ${error.message}`, 'danger');
            console.error("Error in handleAnalysisUpdate:", error);
        }
    }
    
    function onPlotClick(data) {
        const point = data.points[0];
        // Check if the click is on the main force trace (usually trace 0)
        if (point.curveNumber !== 0) {
            return;
        }
    
        const clickedTime = point.x;
    
        // Avoid duplicates (within a small tolerance, e.g., 100ms)
        if (trialState.steps.some(step => Math.abs(step - clickedTime) < 0.1)) {
            showNotification("A step already exists near this time.", "warning");
            return;
        }
    
        // Add new step and sort
        trialState.steps.push(clickedTime);
        trialState.steps.sort((a, b) => a - b);
    
        // Trigger backend recalculation
        requestRecalculation();
    }
    
    async function requestRecalculation() {
        const participantId = parseInt(systematic.trialForm.querySelector('#systematic-participant-id').value);
        const geometryId = parseInt(systematic.trialForm.querySelector('#systematic-geometry-id').value);
    
        if (!participantId || !geometryId) {
            showNotification("Cannot recalculate without participant and geometry IDs.", 'danger');
            return;
        }
    
        // Show a loading overlay on plots
        systematic.plotsArea.style.opacity = '0.5';
    
        try {
            const payload = {
                participantId,
                geometryId,
                steps: trialState.steps,
            };
            const results = await apiRequest('/api/trials/recalculate', 'POST', payload);
            await handleAnalysisUpdate(results);
        } catch (error) {
            showNotification(`Recalculation failed: ${error.message}`, 'danger');
        } finally {
            systematic.plotsArea.style.opacity = '1';
        }
    }

    function renderStepList() {
        systematic.stepList.innerHTML = '';
        systematic.stepCount.textContent = trialState.steps.length;
        
        trialState.steps.forEach((stepTime, index) => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${stepTime.toFixed(3)}s</td>
                <td class="text-end">
                    <button type="button" class="btn btn-sm btn-outline-info py-0 px-1 inspect-step-btn" 
                            data-time="${stepTime}" title="Inspect">
                        <i class="fas fa-search-plus"></i>
                    </button>
                    <button type="button" class="btn btn-sm btn-outline-danger py-0 px-1 delete-step-btn" 
                            data-index="${index}" title="Delete">
                        <i class="fas fa-trash-alt"></i>
                    </button>
                </td>
            `;
            systematic.stepList.appendChild(row);
        });
    }

    systematic.stepList.addEventListener('click', (e) => {
        const target = e.target.closest('button');
        if (!target) return;

        if (target.classList.contains('delete-step-btn')) {
            const index = parseInt(target.dataset.index, 10);
            trialState.steps.splice(index, 1);
            requestRecalculation();
        }

        if (target.classList.contains('inspect-step-btn')) {
            const time = parseFloat(target.dataset.time);
            // Zoom to ±1 second around the step
            Plotly.relayout(systematic.forcePlotDiv, {
                'xaxis.range': [time - 1.0, time + 1.0]
            });
        }
    });


    // --- Live Trial Workflow & Bluetooth ---

    systematic.connectDeviceBtn.addEventListener('click', async () => {
        if (!navigator.bluetooth) {
            showNotification('Web Bluetooth is not available on this browser.', 'danger');
            return;
        }

        try {
            systematic.deviceStatus.textContent = 'Status: Searching...';
            const device = await navigator.bluetooth.requestDevice({
                filters: [{ namePrefix: 'HIP_EXO' }],
                optionalServices: ['0000ffe0-0000-1000-8000-00805f9b34fb'] // The service UUID
            });

            systematic.deviceStatus.textContent = 'Status: Connecting...';
            device.addEventListener('gattserverdisconnected', onDisconnected);
            bleServer = await device.gatt.connect();

            const service = await bleServer.getPrimaryService('0000ffe0-0000-1000-8000-00805f9b34fb');
            bleCharacteristic = await service.getCharacteristic(CHARACTERISTIC_UUID);

            systematic.deviceStatus.textContent = 'Status: Connected';
            systematic.deviceStatus.classList.replace('alert-secondary', 'alert-success');
            systematic.connectDeviceBtn.disabled = true;
            systematic.startStopBtn.disabled = false;

        } catch (error) {
            showNotification(`Bluetooth Error: ${error.message}`, 'danger');
            systematic.deviceStatus.textContent = 'Status: Connection Failed';
        }
    });



    function onDisconnected() {
        showNotification('Device disconnected.', 'warning');
        // Don't reset trial state - keep the data that was collected
        systematic.deviceStatus.textContent = 'Status: Disconnected';
        systematic.deviceStatus.classList.replace('alert-success', 'alert-warning');
        systematic.connectDeviceBtn.disabled = false;
        systematic.startStopBtn.disabled = true;
    }

    async function startDataCollection() {
        trialDataBuffer = []; // Clear previous data
        await bleCharacteristic.startNotifications();
        bleCharacteristic.addEventListener('characteristicvaluechanged', handleCharacteristicValueChanged);
    }

    async function stopDataCollection() {
        await bleCharacteristic.stopNotifications();
        bleCharacteristic.removeEventListener('characteristicvaluechanged', handleCharacteristicValueChanged);
    }

    const dataParser = {
        buffer: new Uint8Array(),
        HEADER_MARKER: 0xAA,
        FOOTER_MARKER: 0xBB,
        PACKET_SIZE: 14, // 1 byte header + 3*4=12 bytes floats + 1 byte footer

        append(data) {
            const newBuffer = new Uint8Array(this.buffer.length + data.byteLength);
            newBuffer.set(this.buffer);
            newBuffer.set(new Uint8Array(data), this.buffer.length);
            this.buffer = newBuffer;
        },

        parse() {
            let packets = [];
            let stillSearching = true;
            while (stillSearching) {
                if (this.buffer.length < this.PACKET_SIZE) {
                    stillSearching = false;
                    continue;
                }

                const headerIndex = this.buffer.indexOf(this.HEADER_MARKER);
                if (headerIndex === -1) {
                    // No header found, discard buffer
                    this.buffer = new Uint8Array();
                    stillSearching = false;
                    continue;
                }

                // If header isn't at the start, discard the bytes before it
                if (headerIndex > 0) {
                    this.buffer = this.buffer.slice(headerIndex);
                }

                // Now that the header is at index 0, check if we have a full packet
                if (this.buffer.length < this.PACKET_SIZE) {
                    stillSearching = false;
                    continue;
                }

                if (this.buffer[this.PACKET_SIZE - 1] === this.FOOTER_MARKER) {
                    // We have a valid packet
                    const packetData = this.buffer.slice(1, this.PACKET_SIZE - 1);
                    const view = new DataView(packetData.buffer);
                    const force = view.getFloat32(0, true); // true for little-endian
                    const accX = view.getFloat32(4, true);
                    const accY = view.getFloat32(8, true);
                    packets.push({ force, accX, accY });

                    // Remove the processed packet from the buffer
                    this.buffer = this.buffer.slice(this.PACKET_SIZE);
                } else {
                    // Corrupted packet, discard the header and search again
                    this.buffer = this.buffer.slice(1);
                }
            }
            return packets;
        }
    };

    function handleCharacteristicValueChanged(event) {
        // `event.target.value` is a DataView. We need its underlying ArrayBuffer.
        dataParser.append(event.target.value.buffer);
        const newPackets = dataParser.parse();
        if (newPackets.length > 0) {
            trialDataBuffer.push(...newPackets);
            // Optional: Log the latest data for debugging
            // console.log(trialDataBuffer[trialDataBuffer.length-1]);
        }
    }


    systematic.startStopBtn.addEventListener('click', async () => {
        if (trialState.running) {
            // Stopping the trial
            stopStopwatch();
            await stopDataCollection();
            systematic.startStopBtn.disabled = true;
            
            // Analyze raw data immediately
            if (trialDataBuffer.length > 0) {
                const participantId = parseInt(systematic.trialForm.querySelector('#systematic-participant-id').value);
                const geometryId = parseInt(systematic.trialForm.querySelector('#systematic-geometry-id').value);
                
                try {
                    // Analyze data on the backend
                    const payload = {
                        participantId,
                        geometryId,
                        trialData: trialDataBuffer,
                    };
                    const results = await apiRequest('/api/trials/analyze', 'POST', payload);
                    
                    // The new handler function takes care of all UI updates
                    await handleAnalysisUpdate(results);


                } catch (error) {
                    showNotification(`Analysis failed: ${error.message}`, 'danger');
                }
            } else {
                showNotification('No data collected from device. Please discard this trial.', 'danger');
                systematic.discardTrialBtn.style.display = 'inline-block';
            }
        } else {
            // Starting the trial
            if (bleCharacteristic) {
                await startDataCollection();
                startStopwatch(); // Start stopwatch when trial starts
                systematic.startStopBtn.innerHTML = '<i class="fas fa-stop"></i> Stop Trial';
                systematic.startStopBtn.classList.replace('btn-success', 'btn-danger');
            } else {
                showNotification('Please connect to device first', 'warning');
            }
        }
    });

    function startStopwatch() {
        if (trialState.running) return;
        trialState.running = true;
        trialState.startTime = Date.now() - trialState.elapsed;
        trialState.timer = setInterval(updateStopwatchDisplay, 100); // Update every 100ms
    }

    function stopStopwatch() {
        if (!trialState.running) return;
        trialState.running = false;
        clearInterval(trialState.timer);
        trialState.elapsed = Date.now() - trialState.startTime;
    }

    function updateStopwatchDisplay() {
        const now = Date.now();
        const diff = trialState.running ? now - trialState.startTime : trialState.elapsed;

        let minutes = Math.floor(diff / 60000);
        let seconds = Math.floor((diff % 60000) / 1000);
        let tenths = Math.floor((diff % 1000) / 100);

        systematic.stopwatch.textContent =
            `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}.${tenths}`;
    }

    // Instability timer functions (same as Grid Search)
    function startInstabilityStopwatch() {
        instabilityState.running = true;
        instabilityState.startTime = Date.now() - instabilityState.elapsed;
        instabilityState.timer = setInterval(updateInstabilityStopwatchDisplay, 100);
    }

    function stopInstabilityStopwatch() {
        instabilityState.running = false;
        if (instabilityState.timer) {
            clearInterval(instabilityState.timer);
            instabilityState.timer = null;
        }
        instabilityState.elapsed = Date.now() - instabilityState.startTime;
    }

    function updateInstabilityStopwatchDisplay() {
        const now = Date.now();
        const diff = instabilityState.running ? now - instabilityState.startTime : instabilityState.elapsed;

        let minutes = Math.floor(diff / 60000);
        let seconds = Math.floor((diff % 60000) / 1000);
        let tenths = Math.floor((diff % 1000) / 100);

        instabilityOptimization.stopwatch.textContent =
            `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}.${tenths}`;
    }

    systematic.trialForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const surveyResponses = {};
        
        // --- SUS Score Calculation ---
        const susQuestions = systematic.trialForm.querySelectorAll('.sus-question');
        let susScore = 0;
        susQuestions.forEach((q, index) => {
            const value = parseInt(q.value);
            const isPositive = q.dataset.positive === 'true';
            surveyResponses[`sus_q${index + 1}`] = value;
            if (isPositive) { susScore += (value - 1); } 
            else { susScore += (5 - value); }
        });
        surveyResponses['sus_score'] = (susScore / 24) * 100;

        // --- NRS Score Calculation ---
        const nrsScoreValue = parseInt(systematic.trialForm.querySelector('#nrs-score').value);
        surveyResponses['nrs_score'] = nrsScoreValue;

        // --- TLX Score Calculation ---
        const tlxQuestions = systematic.trialForm.querySelectorAll('.tlx-question');
        let tlxScore = 0;
        tlxQuestions.forEach((q, index) => {
            const value = parseInt(q.value);
            surveyResponses[`tlx_q${index + 1}`] = value;
            tlxScore += value;
        });
        surveyResponses['tlx_score'] = (tlxScore / 120) * 100; // Scale to 0-100

        // --- Metabolic Cost Input ---
        const metabolicCostInput = systematic.trialForm.querySelector('#metabolic-cost-input');
        const metabolicCost = metabolicCostInput.value ? parseFloat(metabolicCostInput.value) : null;
        surveyResponses['metabolic_cost'] = metabolicCost;
        
        // Grab metrics directly from the UI display
        const stepCountFromUI = parseInt(systematic.stepCount.textContent) || 0;
        const instabilityLossFromUI = parseFloat(systematic.instabilityLossValue.textContent) || 0;
        
        const payload = {
            participantId: parseInt(systematic.trialForm.querySelector('#systematic-participant-id').value),
            geometryId: parseInt(systematic.trialForm.querySelector('#systematic-geometry-id').value),
            surveyResponses: surveyResponses,
            metrics: {
                step_count: stepCountFromUI,
                instability_loss: instabilityLossFromUI
            },
            steps: trialState.steps // Add the final steps to the payload
        };



        try {
            // Validate required fields
            if (!payload.participantId || !payload.geometryId) {
                throw new Error('Missing participant or geometry ID');
            }
            if (!payload.surveyResponses || !trialState.metrics) {
                throw new Error('Missing survey responses or metrics');
            }
            
            console.log('Saving trial with payload:', payload); // Debug
            const newTrial = await apiRequest('/api/trials/save', 'POST', payload);
            console.log('Trial saved successfully:', newTrial); // Debug
            showNotification('Trial completed successfully! You can now run another trial.', 'success');

            systematic.trialRunnerCol.classList.add('d-none'); // Hide trial runner
            systematic.trialForm.reset();
            resetTrialState(); // Fully reset the panel

            // Force a reload of all data from the server to ensure the UI is in sync
            await loadInitialData();
            
            // Refresh the trial management screen for the current participant
            if (appState.currentParticipant) {
                await refreshParticipantView(appState.currentParticipant.id);
            }

        } catch (error) {
            console.error('Error saving trial:', error); // Debug
            showNotification(`Error submitting trial: ${error.message}`, 'danger');
        }
    });

    // --- Trial Runner Control Buttons ---
    systematic.closeTrialRunnerBtn.addEventListener('click', () => {
        if (confirm('Are you sure you want to close the trial runner? Any unsaved data will be lost.')) {
            systematic.trialRunnerCol.classList.add('d-none');
            resetTrialState();
        }
    });

    systematic.discardTrialBtn.addEventListener('click', () => {
        if (confirm('Are you sure you want to discard this trial? Any collected data will be lost.')) {
            // Simply hide the trial runner and reset state - no need to delete from database
            // since the trial was never saved in the first place
            systematic.trialRunnerCol.classList.add('d-none');
            resetTrialState();
            showNotification('Trial discarded successfully', 'success');
        }
    });

    function renderInstabilityPlot(plotData) {
        const plotDiv = systematic.instabilityPlot3D;
    
        if (!plotData || plotData.length < 1) {
            plotDiv.innerHTML = `<div class="text-center text-muted pt-5">No completed trials with instability data for this participant.</div>`;
            return;
        }
    
        const trace = {
            x: plotData.map(d => d.alpha),
            y: plotData.map(d => d.beta),
            z: plotData.map(d => d.gamma),
            mode: 'markers',
            type: 'scatter3d',
            marker: {
                color: plotData.map(d => d.instability_loss),
                colorscale: 'Viridis',
                colorbar: {
                    title: 'Instability Loss'
                },
                size: 8,
                // Use a different symbol for the 'Control' trial
                symbol: plotData.map(d => d.geometry_name.includes('Control') ? 'cross' : 'diamond')
            },
            text: plotData.map(d => `Trial: ${d.geometry_name}<br>Loss: ${d.instability_loss.toFixed(4)}`),
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
    
        Plotly.newPlot(plotDiv, [trace], layout, { responsive: true });
    }

    // --- BO Mode Logic ---

    bo.startBtn.addEventListener('click', async () => {
        const userId = bo.participantSelect.value;
        if (!userId || userId === "Choose...") {
            showNotification("Please select a participant to start a BO session.", 'warning');
            return;
        }

        try {
            const response = await apiRequest('/api/bo/start', 'POST', { userId });
            appState.boSession.active = true;
            appState.boSession.userId = userId;
            appState.boSession.history = response.history;

            bo.dashboard.classList.remove('d-none');
            bo.participantSelect.parentElement.classList.add('d-none');
            
            // Get participant name for display
            const participant = appState.participants.find(p => p.id == userId);
            const participantName = participant ? participant.name : `ID ${userId}`;
            bo.userInfo.textContent = `Active BO Session for: ${participantName}`;
            renderBOHistory();
            // Don't call getNextBOSuggestion() here - wait for objective selection
            bo.objectiveSelectCard.classList.remove('d-none'); // show objective choices only
            bo.userInfo.classList.add('d-none');
            document.getElementById('bo-main-flow').classList.remove('d-none');
            // hide suggestion/history rows
            bo.suggestionBox.parentElement.parentElement.parentElement.classList.add('d-none');
            document.getElementById('bo-setup-card').classList.add('d-none');
            // hide all columns in main-flow except objective card initially
            document.querySelectorAll('#bo-main-flow > div').forEach(col => {
                if(col.id !== 'bo-objective-select-card') col.classList.add('d-none');
            });

        } catch (error) {
            showNotification(`Could not start BO session: ${error.message}`, 'danger');
        }
    });

    function renderBOHistory() {
        // Very similar to original history table
        bo.historyTable.innerHTML = '';
        if (appState.boSession.history.length === 0) {
            bo.historyTable.innerHTML = `<tr><td colspan="5" class="text-center text-muted">No trials yet.</td></tr>`;
            return;
        }
        appState.boSession.history.forEach((entry, index) => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${index + 1}</td>
                <td>${entry.alpha?.toFixed(1) ?? '-'}</td>
                <td>${entry.beta?.toFixed(1) ?? '-'}</td>
                <td>${entry.gamma?.toFixed(1) ?? '-'}</td>
                <td>${entry.Total_Combined_Loss?.toFixed(3) ?? 'N/A'}</td>
            `;
            bo.historyTable.appendChild(row);
        });
    }

    async function getNextBOSuggestion() {
        bo.suggestionBox.innerHTML = `<div class="spinner-border" role="status"><span class="visually-hidden">Loading...</span></div>`;
        try {
            const geometry = await apiRequest(`/api/bo/next-geometry?userId=${appState.boSession.userId}`);
            appState.boSession.suggestion = geometry;
            bo.suggestionBox.innerHTML = `
                <div class="row g-2">
                    <div class="col-4 stat-box">
                        <div class="stat-value">${geometry.alpha.toFixed(1)}</div><div class="stat-label">Alpha</div>
                    </div>
                    <div class="col-4 stat-box">
                        <div class="stat-value">${geometry.beta.toFixed(1)}</div><div class="stat-label">Beta</div>
                    </div>
                    <div class="col-4 stat-box">
                         <div class="stat-value">${geometry.gamma.toFixed(1)}</div><div class="stat-label">Gamma</div>
                    </div>
                </div>`;
        } catch (error) {
            showNotification(`Error getting suggestion: ${error.message}`, 'danger');
            bo.suggestionBox.innerHTML = `<p class="text-danger">Error: ${error.message}</p>`;
        }
    }

    bo.acceptBtn.addEventListener('click', () => {
        const geom = appState.boSession.suggestion;
        bo.trialForm.querySelector('#bo-alpha').value = geom.alpha.toFixed(1);
        bo.trialForm.querySelector('#bo-beta').value = geom.beta.toFixed(1);
        bo.trialForm.querySelector('#bo-gamma').value = geom.gamma.toFixed(1);
        bo.trialCard.classList.remove('d-none');
    });

    bo.objectiveSelectCard.addEventListener('click', async (e) => {
        const btn = e.target.closest('.bo-objective-btn');
        if (!btn) return;

        const objective = btn.dataset.objective;
        
        bo.objectiveButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        if (objective === 'pain') {
            // Start pain optimization
            const userId = appState.boSession.userId;
            if (userId) {
                await startPainOptimization(userId);
            } else {
                showNotification('No user selected for pain optimization', 'danger');
            }
        } else if (objective === 'instability') {
            // Start instability optimization
            const userId = appState.boSession.userId;
            if (userId) {
                await startInstabilityOptimization(userId);
            } else {
                showNotification('No user selected for instability optimization', 'danger');
            }
        } else if (objective === 'effort') {
            // Start effort optimization
            const userId = appState.boSession.userId;
            if (userId) {
                await startEffortOptimization(userId);
            } else {
                showNotification('No user selected for effort optimization', 'danger');
            }
        } else {
            // For other objectives (not implemented yet)
            showNotification(`${objective} optimization is not implemented yet`, 'warning');
            // hide objective choices, show optimisation UI
            bo.objectiveSelectCard.classList.add('d-none');
            bo.userInfo.classList.remove('d-none');
            document.getElementById('bo-main-flow').classList.remove('d-none');
            // show other cols
            document.querySelectorAll('#bo-main-flow > div').forEach(col => {
                if(col.id !== 'bo-objective-select-card') col.classList.remove('d-none');
            });
            // Only call getNextBOSuggestion() for non-pain objectives
            getNextBOSuggestion();
        }
    });

    bo.trialForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData();
        const fileInput = bo.trialForm.querySelector('input[type="file"]');

        const trialData = {
            userId: appState.boSession.userId,
            crutchGeometry: {
                alpha: parseFloat(bo.trialForm.querySelector('#bo-alpha').value),
                beta: parseFloat(bo.trialForm.querySelector('#bo-beta').value),
                gamma: parseFloat(bo.trialForm.querySelector('#bo-gamma').value),
            },
            subjectiveMetrics: {
                effort: parseFloat(bo.trialForm.querySelector('#bo-effort').value),
                pain: parseFloat(bo.trialForm.querySelector('#bo-pain').value),
                stability: parseFloat(bo.trialForm.querySelector('#bo-instability').value),
            }
        };

        formData.append('data', JSON.stringify(trialData));
        formData.append('file', fileInput.files[0]);

        try {
            const response = await fetch(`${SERVER_URL}/api/bo/trial`, { method: 'POST', body: formData });
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error);
            }
            const data = await response.json();

            showNotification('BO trial processed successfully.');
            appState.boSession.history = data.history;
            renderBOHistory();
            // TODO: Update best trial, plot, etc.

            bo.trialCard.classList.add('d-none');
            bo.trialForm.reset();
            getNextBOSuggestion();

        } catch (error) {
            showNotification(error.message, 'danger');
        }
    });

    // Enable delete button when participant selected in BO setup
    bo.participantSelect.addEventListener('change', () => {
        const pid = bo.participantSelect.value;
        bo.deleteParticipantBtn.disabled = !pid;
    });

    // BO Create participant button reuses existing modal, nothing extra to do

    bo.deleteParticipantBtn.addEventListener('click', async () => {
        const participantId = bo.participantSelect.value;
        if (!participantId) return;
        if (confirm('Are you sure you want to delete this participant? This action cannot be undone.')) {
            try {
                await apiRequest(`/api/participants/${participantId}`, 'DELETE');
                showNotification('Participant deleted successfully', 'success');
                await loadInitialData();
            } catch (err) {
                showNotification(`Error deleting participant: ${err.message}`, 'danger');
            }
        }
    });

    // Back button handler
    document.getElementById('bo-objective-back-btn')?.addEventListener('click', () => {
        // Hide dashboard, show initial setup card
        bo.dashboard.classList.add('d-none');
        document.getElementById('bo-setup-card').classList.remove('d-none');
        // Hide objective & main flow
        bo.objectiveSelectCard.classList.add('d-none');
        bo.userInfo.classList.add('d-none');
        document.getElementById('bo-main-flow').classList.add('d-none');
        // reset dropdown & state
        bo.participantSelect.value = '';
        bo.deleteParticipantBtn.disabled = true;
        appState.boSession = { active:false, userId:null, history:[], suggestion:null, selectedObjective:null };
        // Show the dropdown by removing d-none from its parent
        bo.participantSelect.parentElement.classList.remove('d-none');
        // Repopulate the dropdown to show available participants
        populateParticipantSelects();
    });
    
    // Pain BO Decision buttons
    bo.painContinueBtn?.addEventListener('click', async () => {
        // Continue with existing data (restart mode = false)
        await startPainOptimizationSession(painState.userId, false);
    });
    
    bo.painRestartBtn?.addEventListener('click', async () => {
        // Start fresh (restart mode = true)
        await startPainOptimizationSession(painState.userId, true);
    });
    
    bo.painDecisionBackBtn?.addEventListener('click', () => {
        // Go back to objective selection
        bo.painDecisionCard.classList.add('d-none');
        bo.objectiveSelectCard.classList.remove('d-none');
    });

    // --- Pain Optimization Functions ---
    
    async function startPainOptimization(userId) {
        try {
            painState.userId = userId;
            
            // First, check if there's existing NRS data for this participant
            const existingDataResponse = await apiRequest(`/api/pain-bo/check-existing-data?userId=${userId}`);
            
            if (existingDataResponse.has_existing_data) {
                // Show restart/continue decision screen
                showPainOptimizationDecision(existingDataResponse);
            } else {
                // No existing data, start fresh automatically
                await startPainOptimizationSession(userId, true); // restart mode = true (fresh start)
            }
            
        } catch (error) {
            showNotification(`Failed to start pain optimization: ${error.message}`, 'danger');
        }
    }
    
    function showPainOptimizationDecision(existingData) {
        // Hide objective selection
        bo.objectiveSelectCard.classList.add('d-none');
        // Show decision card
        bo.painDecisionCard.classList.remove('d-none');
        
        // Update the summary text
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
        
        bo.painExistingDataSummary.textContent = summaryText;
    }
    
    async function startPainOptimizationSession(userId, restartMode) {
        try {
            // Start the pain optimization session with restart mode flag
            const response = await apiRequest('/api/pain-bo/start', 'POST', { 
                userId, 
                restartMode 
            });
            
            painState.active = true;  // Mark session as active
            painState.userId = userId;
            painState.trials = response.history || [];
            painState.currentTrialNumber = response.trial_count + 1;
            painState.restartMode = restartMode;
            
            // Show the pain optimization screen
            showPainOptimizationScreen();
            
            // Hide decision card if it was shown
            bo.painDecisionCard.classList.add('d-none');
            
            // Update home button state
            updateHomeButtonState();
            
            // If restart mode or no previous trials, show first geometry input
            if (restartMode || painState.trials.length === 0) {
                showFirstGeometryInput();
            } else {
                // Load next geometry from BO
                await loadNextPainGeometry();
            }
            
            const modeText = restartMode ? 'restarted' : 'continued';
            showNotification(`Pain optimization ${modeText} for ${userId}`, 'success');
            
        } catch (error) {
            showNotification(`Failed to start pain optimization: ${error.message}`, 'danger');
        }
    }
    
    async function loadNextPainGeometry() {
        try {
            let geometry;
            if (painState.currentTrialNumber === 1) {
                geometry = await apiRequest(`/api/pain-bo/first-geometry?userId=${painState.userId}`);
            } else {
                geometry = await apiRequest(`/api/pain-bo/next-geometry?userId=${painState.userId}`);
            }
            
            painState.currentGeometry = geometry;
            displayPainGeometry(geometry);
            
        } catch (error) {
            showNotification(`Failed to load geometry: ${error.message}`, 'danger');
        }
    }
    
    function displayPainGeometry(geometry) {
        painOptimization.trialNumber.textContent = geometry.trial_number;
        painOptimization.alphaDisplay.textContent = `${geometry.alpha}°`;
        painOptimization.betaDisplay.textContent = `${geometry.beta}°`;
        painOptimization.gammaDisplay.textContent = `${geometry.gamma}°`;
        
        // Reset the survey card and checkbox
        painOptimization.surveyCard.style.display = 'none';
        painOptimization.highLossCheckbox.checked = false;
        painOptimization.nrsScore.value = 0;
        painOptimization.nrsDisplay.textContent = '0';
        
        // Show geometry actions immediately
        document.getElementById('pain-geometry-actions').style.display = 'block';
    }
    
    function showPainOptimizationScreen() {
        // Hide objective selection
        bo.objectiveSelectCard.classList.add('d-none');
        // Show pain optimization screen
        painOptimization.screen.classList.remove('d-none');
        
        // Update trial history
        renderPainHistory();
        renderPainProgress();
    }
    
    function renderPainHistory() {
        if (painState.trials.length === 0) {
            painOptimization.historyTable.style.display = 'none';
            painOptimization.noHistory.style.display = 'block';
            return;
        }
        
        painOptimization.historyTable.style.display = 'table';
        painOptimization.noHistory.style.display = 'none';
        
        painOptimization.historyTbody.innerHTML = '';
        
        painState.trials.forEach((trial, index) => {
            const row = document.createElement('tr');
            const painScore = trial.survey_responses?.nrs_score ?? 0;
            const isHighLoss = trial.survey_responses?.is_high_loss_penalty || false;
            const status = isHighLoss ? 'High Loss' : 'Normally Tested';
            const statusClass = isHighLoss ? 'text-danger' : 'text-success';
            
            // Add source indicator for Grid Search trials
            const sourceIndicator = trial.source === 'grid_search' ? ' (Grid Search)' : '';
            
            row.innerHTML = `
                <td>${index + 1}${sourceIndicator}</td>
                <td>${trial.alpha}°</td>
                <td>${trial.beta}°</td>
                <td>${trial.gamma}°</td>
                <td><strong>${painScore}</strong></td>
                <td><span class="${statusClass}">${status}</span></td>
            `;
            painOptimization.historyTbody.appendChild(row);
        });
    }
    
    function renderPainProgress() {
        if (painState.trials.length === 0) {
            painOptimization.progressCard.style.display = 'none';
            return;
        }
        
        painOptimization.progressCard.style.display = 'block';
        
        // Find best (lowest) pain score
        let bestScore = Infinity;
        painState.trials.forEach(trial => {
            if (trial.survey_responses?.nrs_score !== undefined) {
                const score = trial.survey_responses.nrs_score;
                if (!trial.survey_responses.is_high_loss_penalty && score < bestScore) {
                    bestScore = score;
                }
            }
        });
        
        painOptimization.bestScore.textContent = bestScore === Infinity ? '-' : bestScore;
        painOptimization.totalTrials.textContent = painState.trials.length;
    }
    
    async function submitPainScore() {
        try {
            const painScore = parseInt(painOptimization.nrsScore.value) || 0;
            
            const response = await apiRequest('/api/pain-bo/record-trial', 'POST', {
                userId: painState.userId,
                geometry: painState.currentGeometry,
                painScore: painScore,
                isHighLoss: false
            });
            
            // Add trial to local state
            painState.trials.push(response.trial);
            painState.currentTrialNumber++;
            
            // Update displays
            renderPainHistory();
            renderPainProgress();
            
            // Hide survey card and reset
            painOptimization.surveyCard.style.display = 'none';
            painOptimization.nrsScore.value = 0;
            painOptimization.nrsDisplay.textContent = '0';
            hideManualLossInput();
            
            showNotification('Pain score recorded successfully!', 'success');
            
            // Load next geometry
            await loadNextPainGeometry();
            
        } catch (error) {
            showNotification(`Failed to record pain score: ${error.message}`, 'danger');
        }
    }
    
    async function suggestAlternativePainGeometry() {
        try {
            const response = await apiRequest('/api/pain-bo/suggest-alternative', 'POST', {
                userId: painState.userId
            });
            
            painState.currentGeometry = response;
            displayPainGeometry(response);
            
            showNotification('Alternative geometry suggested', 'info');
            
        } catch (error) {
            showNotification(`Failed to suggest alternative: ${error.message}`, 'danger');
        }
    }
    
    function exitPainOptimization() {
        // Reset pain state
        painState = {
            active: false,
            userId: null,
            currentGeometry: null,
            trials: [],
            currentTrialNumber: 1,
            restartMode: false
        };
        
        // Hide pain optimization screen
        painOptimization.screen.classList.add('d-none');
        
        // Show objective selection
        bo.objectiveSelectCard.classList.remove('d-none');
        
        // Update home button state
        updateHomeButtonState();
    }

    function showFirstGeometryInput() {
        // Hide the current geometry display and show input form
        painOptimization.screen.querySelector('.card').style.display = 'none';
        painOptimization.firstGeometryCard.style.display = 'block';
        
        // Set default values
        painOptimization.firstAlpha.value = 90;
        painOptimization.firstBeta.value = 110;
        painOptimization.firstGamma.value = 0;
    }
    
    function hideFirstGeometryInput() {
        // Hide input form and show geometry display
        painOptimization.firstGeometryCard.style.display = 'none';
        painOptimization.screen.querySelector('.card').style.display = 'block';
    }
    
    function setFirstGeometry() {
        const alpha = parseInt(painOptimization.firstAlpha.value);
        const beta = parseInt(painOptimization.firstBeta.value);
        const gamma = parseInt(painOptimization.firstGamma.value);
        
        // Validate inputs
        if (alpha < 70 || alpha > 120 || alpha % 5 !== 0) {
            showNotification('Alpha must be between 70-120° in steps of 5°', 'danger');
            return;
        }
        if (beta < 90 || beta > 140 || beta % 5 !== 0) {
            showNotification('Beta must be between 90-140° in steps of 5°', 'danger');
            return;
        }
        if (gamma < -12 || gamma > 12 || gamma % 3 !== 0) {
            showNotification('Gamma must be between -12 to +12° in steps of 3°', 'danger');
            return;
        }
        
        // Set the geometry
        painState.currentGeometry = {
            alpha: alpha,
            beta: beta,
            gamma: gamma,
            trial_number: 1,
            is_first_trial: true
        };
        
        displayPainGeometry(painState.currentGeometry);
        hideFirstGeometryInput();
        
        showNotification('First geometry set successfully!', 'success');
    }
    
    function showAlternativeGeometryInput() {
        // Hide the current geometry display and show alternative input form
        painOptimization.screen.querySelector('.card').style.display = 'none';
        painOptimization.alternativeGeometryCard.style.display = 'block';
        
        // Set default values based on current geometry, ensuring they match discrete values
        if (painState.currentGeometry) {
            // Use current geometry values but ensure they match discrete steps
            const currentAlpha = painState.currentGeometry.alpha;
            const currentBeta = painState.currentGeometry.beta;
            const currentGamma = painState.currentGeometry.gamma;
            
            // Round to nearest discrete values
            painOptimization.altAlpha.value = Math.round(currentAlpha / 5) * 5; // Steps of 5°
            painOptimization.altBeta.value = Math.round(currentBeta / 5) * 5;   // Steps of 5°
            painOptimization.altGamma.value = Math.round(currentGamma / 3) * 3;  // Steps of 3°
        } else {
            // Default to valid discrete values
            painOptimization.altAlpha.value = 95;  // Valid discrete value
            painOptimization.altBeta.value = 125;  // Valid discrete value
            painOptimization.altGamma.value = 0;   // Valid discrete value
        }
    }
    
    function hideAlternativeGeometryInput() {
        // Hide input form and show geometry display
        painOptimization.alternativeGeometryCard.style.display = 'none';
        painOptimization.screen.querySelector('.card').style.display = 'block';
    }
    
    function setAlternativeGeometry() {
        const alpha = parseInt(painOptimization.altAlpha.value);
        const beta = parseInt(painOptimization.altBeta.value);
        const gamma = parseInt(painOptimization.altGamma.value);
        
        // Validate inputs
        if (alpha < 70 || alpha > 120 || alpha % 5 !== 0) {
            showNotification('Alpha must be between 70-120° in steps of 5°', 'danger');
            return;
        }
        if (beta < 90 || beta > 140 || beta % 5 !== 0) {
            showNotification('Beta must be between 90-140° in steps of 5°', 'danger');
            return;
        }
        if (gamma < -12 || gamma > 12 || gamma % 3 !== 0) {
            showNotification('Gamma must be between -12 to +12° in steps of 3°', 'danger');
            return;
        }
        
        // Set the geometry
        painState.currentGeometry = {
            alpha: alpha,
            beta: beta,
            gamma: gamma,
            trial_number: painState.currentTrialNumber,
            is_alternative: true
        };
        
        displayPainGeometry(painState.currentGeometry);
        hideAlternativeGeometryInput();
        
        showNotification('Alternative geometry set successfully!', 'success');
    }
    
    function showManualLossInput() {
        painOptimization.manualLossSection.style.display = 'block';
        painOptimization.manualLossInput.focus();
    }
    
    function hideManualLossInput() {
        painOptimization.manualLossSection.style.display = 'none';
    }
    
    async function submitManualLoss() {
        const manualLoss = parseInt(painOptimization.manualLossInput.value);
        
        if (manualLoss < 11 || manualLoss > 20) {
            showNotification('Manual loss must be between 11-20', 'danger');
            return;
        }
        
        try {
            const response = await apiRequest('/api/pain-bo/record-trial', 'POST', {
                userId: painState.userId,
                geometry: painState.currentGeometry,
                painScore: manualLoss,
                isHighLoss: true
            });
            
            if (response.trial) {
                painState.trials.push(response.trial);
                painState.currentTrialNumber++;
                
                // Reset UI
                painOptimization.highLossCheckbox.checked = false;
                hideManualLossInput();
                painOptimization.surveyCard.style.display = 'none';
                
                // Update displays
                renderPainHistory();
                renderPainProgress();
                
                // Get next geometry
                await loadNextPainGeometry();
                
                showNotification(`Manual loss ${manualLoss} recorded successfully!`, 'success');
            }
        } catch (error) {
            showNotification(`Failed to record manual loss: ${error.message}`, 'danger');
        }
    }

    // --- Pain Optimization Event Listeners ---
    
    // First geometry input event listeners
    painOptimization.setFirstGeometryBtn.addEventListener('click', () => {
        setFirstGeometry();
    });
    
    painOptimization.cancelFirstGeometryBtn.addEventListener('click', () => {
        hideFirstGeometryInput();
        // Go back to objective selection
        exitPainOptimization();
    });
    
    // Alternative geometry input event listeners
    painOptimization.setAlternativeGeometryBtn.addEventListener('click', () => {
        setAlternativeGeometry();
    });
    
    painOptimization.cancelAlternativeGeometryBtn.addEventListener('click', () => {
        hideAlternativeGeometryInput();
    });
    
    // NRS score display update
    painOptimization.nrsScore.addEventListener('input', () => {
        painOptimization.nrsDisplay.textContent = painOptimization.nrsScore.value;
    });
    
    // Test geometry button
    painOptimization.testGeometryBtn.addEventListener('click', () => {
        // Show NRS survey
        painOptimization.surveyCard.style.display = 'block';
    });
    
    // Suggest alternative button
    painOptimization.suggestAlternativeBtn.addEventListener('click', () => {
        showAlternativeGeometryInput();
    });
    
    // Manual loss button
    painOptimization.manualLossBtn.addEventListener('click', () => {
        showManualLossInput();
    });
    
    // Submit pain score button
    painOptimization.submitScoreBtn.addEventListener('click', () => {
        submitPainScore();
    });
    
    // Cancel test button
    painOptimization.cancelTestBtn.addEventListener('click', () => {
        painOptimization.surveyCard.style.display = 'none';
    });
    
    // Manual loss input event listener
    painOptimization.manualLossInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            submitManualLoss();
        }
    });
    
    // Exit optimization button
    painOptimization.exitBtn.addEventListener('click', () => {
        exitPainOptimization();
    });

    // ========================================================================================
    // === INSTABILITY OPTIMIZATION FUNCTIONS ================================================
    // ========================================================================================

    async function startInstabilityOptimization(userId) {
        try {
            // Check for existing instability data
            const response = await apiRequest(`/api/instability-bo/check-existing-data?userId=${userId}`, 'GET');
            
            if (response.has_existing_data) {
                // Show decision screen
                showInstabilityOptimizationDecision(response, userId);
            } else {
                // No existing data, start fresh
                instabilityState.existingData = { has_existing_data: false };
                await startInstabilityOptimizationSession(userId, true); // restart mode
            }
            
        } catch (error) {
            showNotification(`Failed to check existing data: ${error.message}`, 'danger');
        }
    }

    function showInstabilityOptimizationDecision(existingData, userId) {
        // Hide objective selection and show decision card
        bo.objectiveSelectCard.classList.add('d-none');
        document.getElementById('instability-bo-decision-card').classList.remove('d-none');
        
        // Update the summary text
        const gridSearchTrials = existingData.grid_search_trials;
        const instabilityBoTrials = existingData.instability_bo_trials;
        const totalTrials = existingData.total_instability_trials;
        
        let summaryText = `Found ${totalTrials} trial(s) with instability loss data: `;
        if (gridSearchTrials > 0 && instabilityBoTrials > 0) {
            summaryText += `${gridSearchTrials} from Grid Search mode and ${instabilityBoTrials} from previous Instability Optimization sessions.`;
        } else if (gridSearchTrials > 0) {
            summaryText += `${gridSearchTrials} from Grid Search mode.`;
        } else if (instabilityBoTrials > 0) {
            summaryText += `${instabilityBoTrials} from previous Instability Optimization sessions.`;
        }
        
        bo.instabilityExistingDataSummary.textContent = summaryText;
        
        // Store existing data and userId for later use
        instabilityState.existingData = existingData;
        instabilityState.userId = userId;
    }

    async function startInstabilityOptimizationSession(userId, restartMode = false) {
        try {
            // Hide decision card and show instability screen
            document.getElementById('instability-bo-decision-card').classList.add('d-none');
            instabilityOptimization.screen.classList.remove('d-none');
            
            // Update home button state
            updateHomeButtonState();
            
            // Start session with backend
            const response = await apiRequest('/api/instability-bo/start', 'POST', {
                userId: userId,
                restartMode: restartMode
            });
            
            // Initialize instability state
            instabilityState.active = true;
            instabilityState.userId = userId;
            instabilityState.trials = response.history || [];
            instabilityState.currentTrialNumber = response.trial_count + 1;
            instabilityState.restartMode = restartMode;
            
            // Set currentParticipant for data analysis (need participant ID, not name)
            const participantId = bo.participantSelect.value;
            appState.currentParticipant = parseInt(participantId);
            
            // Load first geometry (now that session is initialized)
            await loadNextInstabilityGeometry();
            
            // Update history and progress
            renderInstabilityHistory();
            renderInstabilityProgress();
            
            showNotification(`Instability optimization started in ${restartMode ? 'RESTART' : 'CONTINUE'} mode`, 'success');
            
        } catch (error) {
            showNotification(`Failed to start instability optimization session: ${error.message}`, 'danger');
        }
    }

    async function loadNextInstabilityGeometry() {
        try {
            let geometry;
            
            // If restart mode OR no existing data at all, get first geometry for manual input
            // Otherwise, always run BO using all available historical data
            if (instabilityState.restartMode || !instabilityState.existingData?.has_existing_data) {
                // Truly starting fresh - get first geometry for manual input
                const response = await apiRequest(`/api/instability-bo/first-geometry?userId=${instabilityState.userId}`, 'GET');
                geometry = response;
            } else {
                // We have existing data - always run BO optimization using all historical data
                const response = await apiRequest(`/api/instability-bo/next-geometry?userId=${instabilityState.userId}`, 'GET');
                geometry = response;
            }
            
            instabilityState.currentGeometry = geometry;
            displayInstabilityGeometry(geometry);
            
        } catch (error) {
            showNotification(`Failed to load next geometry: ${error.message}`, 'danger');
        }
    }

    function displayInstabilityGeometry(geometry) {
        instabilityOptimization.trialNumber.textContent = geometry.trial_number || instabilityState.currentTrialNumber;
        instabilityOptimization.alphaDisplay.textContent = `${geometry.alpha}°`;
        instabilityOptimization.betaDisplay.textContent = `${geometry.beta}°`;
        instabilityOptimization.gammaDisplay.textContent = `${geometry.gamma}°`;
        
        // Reset UI state
        instabilityOptimization.resultsCard.classList.add('d-none');
        instabilityOptimization.surveyCard.classList.add('d-none');
        instabilityState.isConnected = false;
        instabilityState.isCollecting = false;
        
        // Show geometry actions immediately (like Pain and Effort optimization)
        document.getElementById('instability-geometry-actions').style.display = 'block';
        
        updateInstabilityConnectionStatus();
    }

    function updateInstabilityConnectionStatus() {
        if (instabilityState.running) {
            instabilityOptimization.connectionStatus.className = 'alert alert-warning mb-3';
            instabilityOptimization.connectionText.textContent = 'Collecting data...';
            instabilityOptimization.connectBtn.classList.add('d-none');
            instabilityOptimization.startStopBtn.innerHTML = '<i class="fas fa-stop"></i> Stop';
            instabilityOptimization.startStopBtn.className = 'btn btn-danger btn-lg';
            instabilityOptimization.startStopBtn.disabled = false;
        } else if (instabilityState.isConnected) {
            instabilityOptimization.connectionStatus.className = 'alert alert-success mb-3';
            instabilityOptimization.connectionText.textContent = 'IMU Connected - Ready to start trial';
            instabilityOptimization.connectBtn.classList.add('d-none');
            instabilityOptimization.startStopBtn.innerHTML = '<i class="fas fa-play"></i> Start';
            instabilityOptimization.startStopBtn.className = 'btn btn-success btn-lg';
            instabilityOptimization.startStopBtn.disabled = false;
        } else {
            instabilityOptimization.connectionStatus.className = 'alert alert-info mb-3';
            instabilityOptimization.connectionText.textContent = 'Click "Connect IMU" to begin';
            instabilityOptimization.connectBtn.classList.remove('d-none');
            instabilityOptimization.startStopBtn.innerHTML = '<i class="fas fa-play"></i> Start';
            instabilityOptimization.startStopBtn.className = 'btn btn-success btn-lg';
            instabilityOptimization.startStopBtn.disabled = true;
        }
    }

    async function connectInstabilityIMU() {
        if (!navigator.bluetooth) {
            showNotification('Web Bluetooth is not available on this browser.', 'danger');
            return;
        }

        try {
            showNotification('Connecting to IMU...', 'info');
            
            // Use the same Bluetooth connection logic as Grid Search
            const device = await navigator.bluetooth.requestDevice({
                filters: [{ namePrefix: 'HIP_EXO' }],
                optionalServices: ['0000ffe0-0000-1000-8000-00805f9b34fb'] // The service UUID
            });

            device.addEventListener('gattserverdisconnected', onInstabilityDisconnected);
            const bleServer = await device.gatt.connect();

            const service = await bleServer.getPrimaryService('0000ffe0-0000-1000-8000-00805f9b34fb');
            const bleCharacteristic = await service.getCharacteristic('0000ffe1-0000-1000-8000-00805f9b34fb');

            // Store the characteristic for data collection
            instabilityState.bleCharacteristic = bleCharacteristic;
            instabilityState.bleServer = bleServer;
            
            instabilityState.isConnected = true;
            updateInstabilityConnectionStatus();
            showNotification('IMU connected successfully!', 'success');
            
        } catch (error) {
            showNotification(`Bluetooth Error: ${error.message}`, 'danger');
        }
    }

    function onInstabilityDisconnected() {
        showNotification('IMU device disconnected.', 'warning');
        instabilityState.isConnected = false;
        updateInstabilityConnectionStatus();
    }

    async function startInstabilityDataCollection() {
        try {
            if (!instabilityState.bleCharacteristic) {
                showNotification('No Bluetooth connection available', 'danger');
                return;
            }
            
            instabilityState.running = true;
            instabilityState.collectedData = [];
            
            // Start stopwatch like Grid Search
            startInstabilityStopwatch();
            
            // Initialize data parser similar to Grid Search
            instabilityState.dataParser = {
                buffer: new Uint8Array(),
                HEADER_MARKER: 0xAA,
                FOOTER_MARKER: 0xBB,
                PACKET_SIZE: 14,
                
                append(data) {
                    const newBuffer = new Uint8Array(this.buffer.length + data.byteLength);
                    newBuffer.set(this.buffer);
                    newBuffer.set(new Uint8Array(data), this.buffer.length);
                    this.buffer = newBuffer;
                },

                parse() {
                    let packets = [];
                    let stillSearching = true;
                    while (stillSearching) {
                        if (this.buffer.length < this.PACKET_SIZE) {
                            stillSearching = false;
                            continue;
                        }

                        const headerIndex = this.buffer.indexOf(this.HEADER_MARKER);
                        if (headerIndex === -1) {
                            this.buffer = new Uint8Array();
                            stillSearching = false;
                            continue;
                        }

                        if (headerIndex + this.PACKET_SIZE > this.buffer.length) {
                            stillSearching = false;
                            continue;
                        }

                        const packet = this.buffer.slice(headerIndex, headerIndex + this.PACKET_SIZE);
                        if (packet[this.PACKET_SIZE - 1] !== this.FOOTER_MARKER) {
                            this.buffer = this.buffer.slice(headerIndex + 1);
                            continue;
                        }

                        // Parse the packet (same format as Grid Search)
                        const force = new Int16Array(packet.slice(1, 3).buffer)[0];
                        const accX = new Int16Array(packet.slice(3, 5).buffer)[0];
                        const accY = new Int16Array(packet.slice(5, 7).buffer)[0];

                        packets.push({ force, accX, accY });
                        this.buffer = this.buffer.slice(headerIndex + this.PACKET_SIZE);
                    }
                    return packets;
                }
            };
            
            updateInstabilityConnectionStatus();
            
            // Start notifications from the Bluetooth characteristic
            await instabilityState.bleCharacteristic.startNotifications();
            instabilityState.bleCharacteristic.addEventListener('characteristicvaluechanged', handleInstabilityCharacteristicValueChanged);
            
            showNotification('Data collection started', 'success');
            
        } catch (error) {
            showNotification(`Failed to start data collection: ${error.message}`, 'danger');
        }
    }

    function handleInstabilityCharacteristicValueChanged(event) {
        try {
            // Use the same approach as Grid Search
            instabilityState.dataParser.append(event.target.value.buffer);
            const newPackets = instabilityState.dataParser.parse();
            if (newPackets.length > 0) {
                instabilityState.collectedData.push(...newPackets);
                // No need to display sample count - we have the timer
            }
        } catch (error) {
            console.error('Error parsing instability data:', error);
        }
    }



    async function stopInstabilityDataCollection() {
        try {
            // Stop Bluetooth notifications
            if (instabilityState.bleCharacteristic) {
                await instabilityState.bleCharacteristic.stopNotifications();
                instabilityState.bleCharacteristic.removeEventListener('characteristicvaluechanged', handleInstabilityCharacteristicValueChanged);
            }
            
            // Stop stopwatch like Grid Search
            stopInstabilityStopwatch();
            
            instabilityState.running = false;
            updateInstabilityConnectionStatus();
            
            showNotification('Data collection stopped. Analyzing...', 'info');
            
            // Check if we have collected data
            if (instabilityState.collectedData.length === 0) {
                showNotification('No data collected. Please try again.', 'warning');
                return;
            }
            
            // Validate required data for analysis
            if (!appState.currentParticipant) {
                showNotification('No participant selected for analysis', 'danger');
                return;
            }
            
            if (!instabilityState.currentGeometry || !instabilityState.currentGeometry.alpha) {
                showNotification('No geometry selected for analysis', 'danger');
                return;
            }
            
            // Send data for analysis with proper timing
            const analysisData = {
                participantId: appState.currentParticipant,
                geometryId: instabilityState.currentGeometry.alpha, // Use alpha as geometry identifier for BO
                trialData: instabilityState.collectedData.map((sample, index) => ({
                    relative_time_ms: index * 5, // Use same 5ms intervals as Grid Search
                    force: sample.force,
                    accX: sample.accX,
                    accY: sample.accY
                }))
            };
            
            // Debug: Check data before sending
            if (analysisData.trialData.length === 0) {
                showNotification('No trial data to analyze', 'warning');
                return;
            }
            
            const response = await apiRequest('/api/trials/analyze', 'POST', analysisData);
            
            instabilityState.analysisResults = response;
            instabilityState.steps = response.steps || [];
            instabilityState.rawData = response.processed_data;
            
            // Show plots area like Grid Search
            instabilityOptimization.plotsArea.classList.remove('d-none');
            await displayInstabilityResults(response);
            
        } catch (error) {
            showNotification(`Failed to analyze data: ${error.message}`, 'danger');
        }
    }

    async function displayInstabilityResults(results) {
        // Show results card
        instabilityOptimization.resultsCard.classList.remove('d-none');
        
        // Update metrics
        instabilityOptimization.lossDisplay.textContent = results.metrics?.instability_loss?.toFixed(4) || '-';
        instabilityOptimization.stepsCount.textContent = results.metrics?.step_count || '-';
        
        // Display plots using same method as Grid Search
        if (results.plots) {
            // Use the same renderPlot function as Grid Search
            async function renderInstabilityPlot(plotDiv, plotPath) {
                try {
                    // Add a cache-busting query parameter
                    const plotResponse = await fetch(`${plotPath}?t=${new Date().getTime()}`);
                    if (!plotResponse.ok) throw new Error(`Failed to fetch plot: ${plotPath}`);
                    const plotHtml = await plotResponse.text();
                    const tempDiv = document.createElement('div');
                    tempDiv.innerHTML = plotHtml;
                    const plotlyGraphDiv = tempDiv.querySelector('.plotly-graph-div');
                    if (plotlyGraphDiv) {
                        const plotData = JSON.parse(plotlyGraphDiv.dataset.raw);
                        Plotly.newPlot(plotDiv, plotData.data, plotData.layout, { responsive: true });
                    } else {
                        plotDiv.innerHTML = plotHtml; // Fallback for simple HTML
                    }
                } catch (error) {
                    console.error('Error rendering instability plot:', error);
                    plotDiv.innerHTML = '<div class="alert alert-warning">Failed to load plot</div>';
                }
            }
            
            if (results.plots.timeseries) {
                await renderInstabilityPlot(instabilityOptimization.forcePlotDiv, results.plots.timeseries);
                
                // Add click listener for step editing (like Grid Search)
                if (instabilityOptimization.forcePlotDiv.removeListener) {
                    instabilityOptimization.forcePlotDiv.removeListener('plotly_click', onInstabilityPlotClick);
                }
                instabilityOptimization.forcePlotDiv.on('plotly_click', onInstabilityPlotClick);
            }
            
            if (results.plots.histogram) {
                await renderInstabilityPlot(instabilityOptimization.histPlotDiv, results.plots.histogram);
            }
        }
        
        // Update steps list
        renderInstabilityStepsList(results.steps || []);
        
        // Store steps in state for editing
        instabilityState.currentSteps = results.steps || [];
        
        // Show geometry action buttons (alternative geometry and manual loss)
        document.getElementById('instability-geometry-actions').style.display = 'block';
    }

    // Instability Alternative Geometry and Manual Loss Functions (copied from Pain)
    function showInstabilityAlternativeGeometryInput() {
        // Hide the current geometry display and show alternative input form
        instabilityOptimization.screen.querySelector('.card').style.display = 'none';
        instabilityOptimization.alternativeGeometryCard.style.display = 'block';
        
        // Set default values based on current geometry, ensuring they match discrete values
        if (instabilityState.currentGeometry) {
            // Use current geometry values but ensure they match discrete steps
            const currentAlpha = instabilityState.currentGeometry.alpha;
            const currentBeta = instabilityState.currentGeometry.beta;
            const currentGamma = instabilityState.currentGeometry.gamma;
            
            // Round to nearest discrete values
            instabilityOptimization.altAlphaInput.value = Math.round(currentAlpha / 5) * 5; // Steps of 5°
            instabilityOptimization.altBetaInput.value = Math.round(currentBeta / 5) * 5;   // Steps of 5°
            instabilityOptimization.altGammaInput.value = Math.round(currentGamma / 3) * 3;  // Steps of 3°
        } else {
            // Default to valid discrete values
            instabilityOptimization.altAlphaInput.value = 95;  // Valid discrete value
            instabilityOptimization.altBetaInput.value = 125;  // Valid discrete value
            instabilityOptimization.altGammaInput.value = 0;   // Valid discrete value
        }
    }
    
    function hideInstabilityAlternativeGeometryInput() {
        // Hide input form and show geometry display
        instabilityOptimization.alternativeGeometryCard.style.display = 'none';
        instabilityOptimization.screen.querySelector('.card').style.display = 'block';
    }
    
    function setInstabilityAlternativeGeometry() {
        const alpha = parseInt(instabilityOptimization.altAlphaInput.value);
        const beta = parseInt(instabilityOptimization.altBetaInput.value);
        const gamma = parseInt(instabilityOptimization.altGammaInput.value);
        
        // Validate inputs
        if (alpha < 70 || alpha > 120 || alpha % 5 !== 0) {
            showNotification('Alpha must be between 70-120° in steps of 5°', 'danger');
            return;
        }
        if (beta < 90 || beta > 140 || beta % 5 !== 0) {
            showNotification('Beta must be between 90-140° in steps of 5°', 'danger');
            return;
        }
        if (gamma < -12 || gamma > 12 || gamma % 3 !== 0) {
            showNotification('Gamma must be between -12 to +12° in steps of 3°', 'danger');
            return;
        }
        
        // Set the geometry
        instabilityState.currentGeometry = {
            alpha: alpha,
            beta: beta,
            gamma: gamma,
            trial_number: instabilityState.currentTrialNumber,
            is_alternative: true
        };
        
        displayInstabilityGeometry(instabilityState.currentGeometry);
        hideInstabilityAlternativeGeometryInput();
        
        showNotification('Alternative geometry set successfully!', 'success');
    }
    
    function showInstabilityManualLossInput() {
        instabilityOptimization.manualLossSection.style.display = 'block';
        instabilityOptimization.manualLossInput.focus();
    }
    
    function hideInstabilityManualLossInput() {
        instabilityOptimization.manualLossSection.style.display = 'none';
    }
    
    async function submitInstabilityManualLoss() {
        const manualLoss = parseInt(instabilityOptimization.manualLossInput.value);
        
        if (manualLoss < 11 || manualLoss > 20) {
            showNotification('Manual loss must be between 11-20', 'danger');
            return;
        }
        
        try {
            const response = await apiRequest('/api/instability-bo/record-trial', 'POST', {
                userId: instabilityState.userId,
                geometry: instabilityState.currentGeometry,
                instabilityLoss: manualLoss,
                susScore: 0  // Default SUS score for manual loss
            });
            
            if (response.trial) {
                instabilityState.trials.push(response.trial);
                instabilityState.currentTrialNumber++;
                
                // Reset UI
                hideInstabilityManualLossInput();
                instabilityOptimization.surveyCard.classList.add('d-none');
                
                // Update displays
                renderInstabilityHistory();
                renderInstabilityProgress();
                
                // Get next geometry
                await loadNextInstabilityGeometry();
                
                showNotification(`Manual loss ${manualLoss} recorded successfully!`, 'success');
            }
        } catch (error) {
            showNotification(`Failed to record manual loss: ${error.message}`, 'danger');
        }
    }

    // --- Effort Optimization Functions (copied from Pain) ---
    
    async function startEffortOptimization(userId) {
        try {
            effortState.userId = userId;
            
            console.log('Starting effort optimization for userId:', userId);
            
            // First, check if there's existing metabolic cost data for this participant
            const existingDataResponse = await apiRequest(`/api/effort-bo/check-existing-data?userId=${userId}`);
            
            console.log('Existing data response:', existingDataResponse);
            
            // ALWAYS show the choice screen, regardless of existing data
            showEffortOptimizationDecision(existingDataResponse);
            
        } catch (error) {
            console.error('Effort optimization error:', error);
            showNotification(`Failed to start effort optimization: ${error.message}`, 'danger');
            
            // Show choice screen even if there's an error
            showEffortOptimizationDecision({
                has_existing_data: false,
                trialCount: 0,
                gridSearchCount: 0,
                effortBoCount: 0
            });
        }
    }
    
    function showEffortOptimizationDecision(existingData) {
        console.log('Showing effort optimization decision with data:', existingData);
        
        // Hide objective selection
        bo.objectiveSelectCard.classList.add('d-none');
        // Show decision card
        document.getElementById('effort-bo-decision-card').classList.remove('d-none');
        
        console.log('Decision card should now be visible');
        
        // Update the summary text
        const gridSearchTrials = existingData.gridSearchCount || 0;
        const effortBoTrials = existingData.effortBoCount || 0;
        const totalTrials = existingData.trialCount || 0;
        
        let summaryText;
        if (totalTrials === 0) {
            summaryText = `No previous metabolic cost data found for this participant.`;
        } else {
            summaryText = `Found ${totalTrials} trial(s) with metabolic cost data: `;
            if (gridSearchTrials > 0 && effortBoTrials > 0) {
                summaryText += `${gridSearchTrials} from Grid Search mode and ${effortBoTrials} from previous Effort Optimization sessions.`;
            } else if (gridSearchTrials > 0) {
                summaryText += `${gridSearchTrials} from Grid Search mode.`;
            } else if (effortBoTrials > 0) {
                summaryText += `${effortBoTrials} from previous Effort Optimization sessions.`;
            }
        }
        
        document.getElementById('effort-existing-data-summary').textContent = summaryText;
    }
    
    async function startEffortOptimizationSession(userId, restartMode) {
        console.log(`startEffortOptimizationSession called with userId: ${userId}, restartMode: ${restartMode}`);
        
        try {
            // Start the effort optimization session with restart mode flag
            const response = await apiRequest('/api/effort-bo/start', 'POST', { 
                userId, 
                restartMode 
            });
            
            console.log('Effort BO start response:', response);
            
            effortState.active = true;  // Mark session as active
            effortState.userId = userId;
            effortState.trials = response.history || [];
            effortState.currentTrialNumber = response.trial_count + 1;
            effortState.restartMode = restartMode;
            
            console.log('Effort state after setup:', effortState);
            
            // Show the effort optimization screen
            console.log('About to call showScreen("effortOptimization")');
            showScreen('effortOptimization');
            
            // Hide decision card if it was shown
            document.getElementById('effort-bo-decision-card').classList.add('d-none');
            
            // Update home button state
            updateHomeButtonState();
            
            // Render the history immediately
            renderEffortHistory();
            renderEffortProgress();
            
            // If restart mode or no previous trials, show first geometry input
            if (restartMode || effortState.trials.length === 0) {
                console.log('Showing first geometry input');
                showFirstEffortGeometryInput();
            } else {
                console.log('Loading next geometry from BO');
                // Load next geometry from BO
                await loadNextEffortGeometry();
            }
            
            const modeText = restartMode ? 'restarted' : 'continued';
            showNotification(`Effort optimization ${modeText} for ${userId}`, 'success');
            
        } catch (error) {
            console.error('Error in startEffortOptimizationSession:', error);
            showNotification(`Failed to start effort optimization: ${error.message}`, 'danger');
        }
    }
    
    async function loadNextEffortGeometry() {
        try {
            let response;
            
            // Check if we have any trials yet
            if (effortState.trials.length === 0) {
                // First geometry - get from first-geometry endpoint
                response = await apiRequest(`/api/effort-bo/first-geometry?userId=${effortState.userId}`, 'GET');
            } else {
                // Get next geometry from BO
                response = await apiRequest(`/api/effort-bo/next-geometry?userId=${effortState.userId}`, 'GET');
            }
            
            if (response.geometry) {
                effortState.currentGeometry = response.geometry;
                displayEffortGeometry(response.geometry);
                showNotification('New geometry loaded!', 'info');
            } else {
                showNotification('Unable to load geometry', 'warning');
            }
        } catch (error) {
            showNotification(`Failed to load next geometry: ${error.message}`, 'danger');
        }
    }
    
    function displayEffortGeometry(geometry) {
        effortOptimization.alphaDisplay.textContent = `${geometry.alpha}°`;
        effortOptimization.betaDisplay.textContent = `${geometry.beta}°`;
        effortOptimization.gammaDisplay.textContent = `${geometry.gamma}°`;
        
        // Hide survey initially
        effortOptimization.surveyCard.style.display = 'none';
        
        // Show geometry actions immediately
        document.getElementById('effort-geometry-actions').style.display = 'block';
        
        // Show the geometry and enable test button
        showNotification(`Geometry: α=${geometry.alpha}°, β=${geometry.beta}°, γ=${geometry.gamma}°`, 'info');
    }
    
    function renderEffortHistory() {
        const tbody = effortOptimization.historyTbody;
        tbody.innerHTML = '';
        
        effortState.trials.forEach((trial, index) => {
            const row = document.createElement('tr');
            // Don't make effort trials clickable - they have no raw data to edit
            
            // Calculate effort score from metabolic cost
            let effortScore = '-';
            if (trial.metabolic_cost !== undefined && trial.metabolic_cost !== null) {
                effortScore = Number(trial.metabolic_cost).toFixed(2);
            } else if (trial.survey_responses && trial.survey_responses.metabolic_cost) {
                effortScore = Number(trial.survey_responses.metabolic_cost).toFixed(2);
            } else if (trial.processed_features && trial.processed_features.effort_score) {
                effortScore = Number(trial.processed_features.effort_score).toFixed(2);
            }
            
            // Status
            let status = trial.status || 'Normally Tested';
            let statusBadge = 'badge-success';
            if (trial.source === 'grid_search') {
                status = 'Grid Search';
                statusBadge = 'badge-info';
            }
            
            row.innerHTML = `
                <td>${index + 1}</td>
                <td>α${trial.alpha}° β${trial.beta}° γ${trial.gamma}°</td>
                <td>${effortScore}</td>
                <td><span class="badge ${statusBadge}">${status}</span></td>
            `;
            tbody.appendChild(row);
        });
    }
    
    function renderEffortProgress() {
        effortOptimization.totalTrials.textContent = effortState.trials.length;
        
                    // Find best effort score (lowest metabolic cost)
        let bestScore = null;
        effortState.trials.forEach(trial => {
            let metabolicCost = null;
            if (trial.metabolic_cost !== undefined && trial.metabolic_cost !== null) {
                metabolicCost = trial.metabolic_cost;
            } else if (trial.survey_responses && trial.survey_responses.metabolic_cost) {
                metabolicCost = trial.survey_responses.metabolic_cost;
            } else if (trial.processed_features && trial.processed_features.effort_score) {
                metabolicCost = trial.processed_features.effort_score;
            }
            
            if (metabolicCost !== null && (bestScore === null || metabolicCost < bestScore)) {
                bestScore = metabolicCost;
            }
        });
        
        effortOptimization.bestScore.textContent = bestScore !== null ? bestScore.toFixed(1) : '-';
    }
    
    function showEffortSurvey() {
        effortOptimization.surveyCard.style.display = 'block';
        
        // Reset form to defaults
        effortOptimization.surveyForm.querySelectorAll('input[type="range"]').forEach(input => {
            input.value = 11; // Middle value
            const valueDisplay = input.parentElement.querySelector('.effort-value');
            if (valueDisplay) {
                valueDisplay.textContent = '11';
            }
        });
        
        // Add value change listeners for real-time updates
        effortOptimization.surveyForm.querySelectorAll('input[type="range"]').forEach(input => {
            input.addEventListener('input', (e) => {
                const valueDisplay = e.target.parentElement.querySelector('.effort-value');
                if (valueDisplay) {
                    valueDisplay.textContent = e.target.value;
                }
            });
        });
    }
    
    async function submitEffortSurvey() {
        const metabolicCostInput = document.getElementById('effort-metabolic-cost-input');
        const metabolicCost = parseFloat(metabolicCostInput.value);
        
        if (!metabolicCost || isNaN(metabolicCost)) {
            showNotification('Please enter a valid metabolic cost value', 'danger');
            return;
        }
        
        try {
            const response = await apiRequest('/api/effort-bo/record-trial', 'POST', {
                userId: effortState.userId,
                geometry: effortState.currentGeometry,
                effortScore: metabolicCost
            });
            
            if (response.trial) {
                effortState.trials.push(response.trial);
                effortState.currentTrialNumber++;
                
                // Reset UI
                effortOptimization.surveyCard.style.display = 'none';
                hideEffortManualLossInput();
                
                // Update displays
                renderEffortHistory();
                renderEffortProgress();
                
                // Get next geometry
                await loadNextEffortGeometry();
                
                showNotification(`Metabolic cost ${metabolicCost.toFixed(2)} recorded successfully!`, 'success');
            }
        } catch (error) {
            showNotification(`Failed to record effort score: ${error.message}`, 'danger');
        }
    }
    
    async function exitEffortOptimization() {
        effortState.active = false;
        showScreen('bo');
        bo.objectiveSelectCard.classList.remove('d-none');
        resetEffortOptimizationUI();
        updateHomeButtonState();
        showNotification('Effort optimization session ended', 'info');
    }
    
    // Effort Alternative Geometry and Manual Loss Functions (copied from Pain)
    function showEffortAlternativeGeometryInput() {
        // Hide the current geometry display and show alternative input form
        effortOptimization.screen.querySelector('.card').style.display = 'none';
        effortOptimization.alternativeGeometryCard.style.display = 'block';
        
        // Set default values based on current geometry, ensuring they match discrete values
        if (effortState.currentGeometry) {
            // Use current geometry values but ensure they match discrete steps
            const currentAlpha = effortState.currentGeometry.alpha;
            const currentBeta = effortState.currentGeometry.beta;
            const currentGamma = effortState.currentGeometry.gamma;
            
            // Round to nearest discrete values
            effortOptimization.altAlphaInput.value = Math.round(currentAlpha / 5) * 5; // Steps of 5°
            effortOptimization.altBetaInput.value = Math.round(currentBeta / 5) * 5;   // Steps of 5°
            effortOptimization.altGammaInput.value = Math.round(currentGamma / 3) * 3;  // Steps of 3°
        } else {
            // Default to valid discrete values
            effortOptimization.altAlphaInput.value = 95;  // Valid discrete value
            effortOptimization.altBetaInput.value = 125;  // Valid discrete value
            effortOptimization.altGammaInput.value = 0;   // Valid discrete value
        }
    }
    
    function hideEffortAlternativeGeometryInput() {
        // Hide input form and show geometry display
        effortOptimization.alternativeGeometryCard.style.display = 'none';
        effortOptimization.screen.querySelector('.card').style.display = 'block';
    }
    
    function setEffortAlternativeGeometry() {
        const alpha = parseInt(effortOptimization.altAlphaInput.value);
        const beta = parseInt(effortOptimization.altBetaInput.value);
        const gamma = parseInt(effortOptimization.altGammaInput.value);
        
        // Validate inputs
        if (alpha < 70 || alpha > 120 || alpha % 5 !== 0) {
            showNotification('Alpha must be between 70-120° in steps of 5°', 'danger');
            return;
        }
        if (beta < 90 || beta > 140 || beta % 5 !== 0) {
            showNotification('Beta must be between 90-140° in steps of 5°', 'danger');
            return;
        }
        if (gamma < -12 || gamma > 12 || gamma % 3 !== 0) {
            showNotification('Gamma must be between -12 to +12° in steps of 3°', 'danger');
            return;
        }
        
        // Set the geometry
        effortState.currentGeometry = {
            alpha: alpha,
            beta: beta,
            gamma: gamma,
            trial_number: effortState.currentTrialNumber,
            is_alternative: true
        };
        
        displayEffortGeometry(effortState.currentGeometry);
        hideEffortAlternativeGeometryInput();
        
        showNotification('Alternative geometry set successfully!', 'success');
    }
    
    function showEffortManualLossInput() {
        effortOptimization.manualLossSection.style.display = 'block';
        effortOptimization.manualLossInput.focus();
    }
    
    function hideEffortManualLossInput() {
        effortOptimization.manualLossSection.style.display = 'none';
    }
    
    async function submitEffortManualLoss() {
        const manualLoss = parseInt(effortOptimization.manualLossInput.value);
        
        if (manualLoss < 11 || manualLoss > 20) {
            showNotification('Manual loss must be between 11-20', 'danger');
            return;
        }
        
        try {
            const response = await apiRequest('/api/effort-bo/record-trial', 'POST', {
                userId: effortState.userId,
                geometry: effortState.currentGeometry,
                effortScore: manualLoss,
                isHighLoss: true
            });
            
            if (response.trial) {
                effortState.trials.push(response.trial);
                effortState.currentTrialNumber++;
                
                // Reset UI
                hideEffortManualLossInput();
                effortOptimization.surveyCard.style.display = 'none';
                
                // Update displays
                renderEffortHistory();
                renderEffortProgress();
                
                // Get next geometry
                await loadNextEffortGeometry();
                
                showNotification(`Manual loss ${manualLoss} recorded successfully!`, 'success');
            }
        } catch (error) {
            showNotification(`Failed to record manual loss: ${error.message}`, 'danger');
        }
    }
    
    function showFirstEffortGeometryInput() {
        effortOptimization.firstGeometryCard.style.display = 'block';
        effortOptimization.testGeometryBtn.style.display = 'none';
    }
    
    function hideFirstEffortGeometryInput() {
        effortOptimization.firstGeometryCard.style.display = 'none';
        effortOptimization.testGeometryBtn.style.display = 'block';
    }
    
    function setFirstEffortGeometry() {
        const alpha = parseFloat(effortOptimization.firstAlphaInput.value);
        const beta = parseFloat(effortOptimization.firstBetaInput.value);
        const gamma = parseFloat(effortOptimization.firstGammaInput.value);
        
        if (isNaN(alpha) || isNaN(beta) || isNaN(gamma)) {
            showNotification('Please enter valid geometry values', 'warning');
            return;
        }
        
        effortState.currentGeometry = { alpha, beta, gamma };
        hideFirstEffortGeometryInput();
        showNotification('First geometry set! Click "Test This Geometry" to begin.', 'success');
    }

    // Plot click handler for step editing (like Grid Search)
    function onInstabilityPlotClick(data) {
        const point = data.points[0];
        // Check if the click is on the main force trace (usually trace 0)
        if (point.curveNumber !== 0) {
            return;
        }

        const clickedTime = point.x;

        // Avoid duplicates (within a small tolerance, e.g., 100ms)
        if (instabilityState.steps.some(step => Math.abs(step - clickedTime) < 0.1)) {
            showNotification("A step already exists near this time.", "warning");
            return;
        }

        // Add new step and sort
        instabilityState.steps.push(clickedTime);
        instabilityState.steps.sort((a, b) => a - b);

        // Trigger backend recalculation
        recalculateInstabilityAnalysis();
    }

    function renderInstabilityStepsList(steps) {
        instabilityOptimization.stepsList.innerHTML = '';
        
        if (steps.length === 0) {
            instabilityOptimization.stepsList.innerHTML = '<div class="text-muted small">No steps detected</div>';
            return;
        }
        
        steps.forEach((step, index) => {
            const stepDiv = document.createElement('div');
            stepDiv.className = 'd-flex justify-content-between align-items-center py-1 border-bottom';
            stepDiv.innerHTML = `
                <span class="small">Step ${index + 1}: ${step.toFixed(3)}s</span>
                <button class="btn btn-sm btn-outline-danger delete-instability-step-btn" data-step-index="${index}">
                    <i class="fas fa-times"></i>
                </button>
            `;
            instabilityOptimization.stepsList.appendChild(stepDiv);
        });
    }

    async function recalculateInstabilityAnalysis() {
        try {
            if (!instabilityState.currentSteps) {
                showNotification('No steps data available for recalculation', 'warning');
                return;
            }
            
            showNotification('Recalculating analysis...', 'info');
            
            const recalcData = {
                participantId: appState.currentParticipant,
                geometryId: instabilityState.currentGeometry.alpha,
                steps: instabilityState.currentSteps
            };
            
            const response = await apiRequest('/api/trials/recalculate', 'POST', recalcData);
            
            instabilityState.analysisResults = response;
            displayInstabilityResults(response);
            
            showNotification('Analysis recalculated successfully', 'success');
            
        } catch (error) {
            showNotification(`Failed to recalculate analysis: ${error.message}`, 'danger');
        }
    }

    function showInstabilitySurvey() {
        instabilityOptimization.surveyCard.classList.remove('d-none');
        
        // Reset survey values to default (3)
        instabilityOptimization.surveyForm.querySelectorAll('.sus-question').forEach(input => {
            input.value = 3;
            const valueDisplay = input.parentElement.querySelector('.sus-value');
            if (valueDisplay) valueDisplay.textContent = '3';
        });
    }

    async function submitInstabilitySurvey() {
        try {
            if (!instabilityState.analysisResults) {
                showNotification('No analysis results available', 'warning');
                return;
            }
            
            // Collect SUS survey responses
            const surveyResponses = {};
            instabilityOptimization.surveyForm.querySelectorAll('.sus-question').forEach(input => {
                surveyResponses[input.name] = parseInt(input.value);
            });
            
            // Calculate SUS score (simplified calculation)
            const susScore = Object.values(surveyResponses).reduce((sum, val) => sum + val, 0) * 2.5; // Scale to 0-100
            
            // Record trial with instability loss and SUS score using new API
            const trialData = {
                userId: instabilityState.userId,
                geometry: {
                    alpha: instabilityState.currentGeometry.alpha,
                    beta: instabilityState.currentGeometry.beta,
                    gamma: instabilityState.currentGeometry.gamma
                },
                instabilityLoss: instabilityState.analysisResults.metrics.instability_loss,
                susScore: susScore
            };
            
            const response = await apiRequest('/api/instability-bo/record-trial', 'POST', trialData);
            
            // Add to trials history
            instabilityState.trials.push(response.trial);
            instabilityState.currentTrialNumber++;
            
            // Update displays
            renderInstabilityHistory();
            renderInstabilityProgress();
            
            // Hide survey and show next geometry suggestion
            instabilityOptimization.surveyCard.classList.add('d-none');
            instabilityOptimization.suggestNextBtn.classList.remove('d-none');
            
            showNotification('Trial recorded successfully!', 'success');
            
        } catch (error) {
            showNotification(`Failed to record trial: ${error.message}`, 'danger');
        }
    }

    function renderInstabilityHistory() {
        instabilityOptimization.historyTbody.innerHTML = '';
        
        instabilityState.trials.forEach((trial, index) => {
            const row = document.createElement('tr');
            const susScore = trial.survey_responses?.sus_score || '-';
            const loss = trial.processed_features?.instability_loss?.toFixed(4) || '-';
            
            // Add source indicator for Grid Search trials
            const trialNumber = trial.source === 'grid_search' ? `${index + 1} (Grid Search)` : index + 1;
            
            // Make both Grid Search and Instability BO trials clickable (they have raw data for editing)
            if (trial.source === 'grid_search' || trial.source === 'instability_bo') {
                row.className = 'clickable-trial-row';
                row.style.cursor = 'pointer';
                row.dataset.trialId = trial.id;
            }
            
            row.innerHTML = `
                <td>${trialNumber}</td>
                <td>${trial.alpha}°</td>
                <td>${trial.beta}°</td>
                <td>${trial.gamma}°</td>
                <td><strong>${loss}</strong></td>
                <td>${susScore}</td>
            `;
            instabilityOptimization.historyTbody.appendChild(row);
        });
    }

    function renderInstabilityProgress() {
        if (instabilityState.trials.length === 0) {
            instabilityOptimization.bestLoss.textContent = '-';
            instabilityOptimization.totalTrials.textContent = '0';
            return;
        }
        
        // Find best (lowest) instability loss
        const bestLoss = Math.min(...instabilityState.trials.map(t => 
            t.processed_features?.instability_loss || Infinity
        ));
        
        instabilityOptimization.bestLoss.textContent = bestLoss === Infinity ? '-' : bestLoss.toFixed(4);
        instabilityOptimization.totalTrials.textContent = instabilityState.trials.length;
    }

    function exitInstabilityOptimization() {
        instabilityState.active = false;
        instabilityState = {
            active: false,
            userId: null,
            currentGeometry: null,
            trials: [],
            currentTrialNumber: 1,
            isConnected: false,
            isCollecting: false,
            collectedData: [],
            analysisResults: null
        };
        
        // Hide instability screen, show objective selection
        instabilityOptimization.screen.classList.add('d-none');
        bo.objectiveSelectCard.classList.remove('d-none');
        
        // Update home button state
        updateHomeButtonState();
    }

    // ========================================================================================
    // === INSTABILITY OPTIMIZATION EVENT LISTENERS ==========================================
    // ========================================================================================

    // Connect IMU button
    instabilityOptimization.connectBtn.addEventListener('click', () => {
        connectInstabilityIMU();
    });

    // Start/Stop data collection button (like Grid Search)
    instabilityOptimization.startStopBtn.addEventListener('click', async () => {
        if (instabilityState.running) {
            // Stopping the trial
            stopInstabilityStopwatch();
            await stopInstabilityDataCollection();
            instabilityOptimization.startStopBtn.disabled = true;
        } else {
            // Starting the trial
            startInstabilityStopwatch();
            await startInstabilityDataCollection();
        }
    });

    // Show survey button
    instabilityOptimization.showSurveyBtn.addEventListener('click', () => {
        showInstabilitySurvey();
    });

    // Survey form submit
    instabilityOptimization.surveyForm.addEventListener('submit', (e) => {
        e.preventDefault();
        submitInstabilitySurvey();
    });

    // Survey cancel button
    instabilityOptimization.surveyCancelBtn.addEventListener('click', () => {
        instabilityOptimization.surveyCard.classList.add('d-none');
    });

    // SUS question input changes (update display values)
    instabilityOptimization.surveyForm.querySelectorAll('.sus-question').forEach(input => {
        input.addEventListener('input', () => {
            const valueDisplay = input.parentElement.querySelector('.sus-value');
            if (valueDisplay) valueDisplay.textContent = input.value;
        });
    });

    // Delete step buttons (event delegation)
    instabilityOptimization.stepsList.addEventListener('click', (e) => {
        const deleteBtn = e.target.closest('.delete-instability-step-btn');
        if (deleteBtn) {
            const stepIndex = parseInt(deleteBtn.dataset.stepIndex);
            if (instabilityState.currentSteps) {
                instabilityState.currentSteps.splice(stepIndex, 1);
                renderInstabilityStepsList(instabilityState.currentSteps);
            }
        }
    });

    // Suggest next geometry button
    instabilityOptimization.suggestNextBtn.addEventListener('click', async () => {
        instabilityOptimization.suggestNextBtn.classList.add('d-none');
        await loadNextInstabilityGeometry();
    });

    // Exit optimization button
    instabilityOptimization.exitBtn.addEventListener('click', () => {
        exitInstabilityOptimization();
    });

    // ========================================================================================
    // === INSTABILITY DECISION EVENT LISTENERS ===============================================
    // ========================================================================================

    // Instability Restart button
    document.getElementById('instability-restart-btn').addEventListener('click', async () => {
        try {
            await startInstabilityOptimizationSession(instabilityState.userId, true); // restart mode
        } catch (error) {
            showNotification(`Failed to restart instability optimization: ${error.message}`, 'danger');
        }
    });

    // Instability Continue button
    document.getElementById('instability-continue-btn').addEventListener('click', async () => {
        try {
            await startInstabilityOptimizationSession(instabilityState.userId, false); // continue mode
        } catch (error) {
            showNotification(`Failed to continue instability optimization: ${error.message}`, 'danger');
        }
    });

    // Instability suggest alternative button
    instabilityOptimization.suggestAlternativeBtn.addEventListener('click', () => {
        showInstabilityAlternativeGeometryInput();
    });
    
    // Instability test geometry button (show survey like Pain/Effort)
    instabilityOptimization.testGeometryBtn.addEventListener('click', () => {
        showInstabilitySurvey();
    });
    
    // Instability manual loss button
    instabilityOptimization.manualLossBtn.addEventListener('click', () => {
        showInstabilityManualLossInput();
    });
    
    // Instability set alternative geometry button
    instabilityOptimization.setAlternativeGeometryBtn.addEventListener('click', () => {
        setInstabilityAlternativeGeometry();
    });
    
    // Instability cancel alternative geometry button
    instabilityOptimization.cancelAlternativeGeometryBtn.addEventListener('click', () => {
        hideInstabilityAlternativeGeometryInput();
    });
    
    // Instability manual loss input event listener
    instabilityOptimization.manualLossInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            submitInstabilityManualLoss();
        }
    });

    // Instability Back to Objectives button
    document.getElementById('instability-decision-back-btn').addEventListener('click', () => {
        // Hide decision card and go back to objective selection
        document.getElementById('instability-bo-decision-card').classList.add('d-none');
        bo.objectiveSelectCard.classList.remove('d-none');
    });

    // --- Effort Optimization Event Listeners ---
    
    // Effort test geometry button
    effortOptimization.testGeometryBtn.addEventListener('click', () => {
        showEffortSurvey();
    });
    
    // Effort suggest alternative button
    effortOptimization.suggestAlternativeBtn.addEventListener('click', () => {
        showEffortAlternativeGeometryInput();
    });
    
    // Effort manual loss button
    effortOptimization.manualLossBtn.addEventListener('click', () => {
        showEffortManualLossInput();
    });
    
    // Effort set alternative geometry button
    effortOptimization.setAlternativeGeometryBtn.addEventListener('click', () => {
        setEffortAlternativeGeometry();
    });
    
    // Effort cancel alternative geometry button
    effortOptimization.cancelAlternativeGeometryBtn.addEventListener('click', () => {
        hideEffortAlternativeGeometryInput();
    });
    
    // Effort manual loss input event listener
    effortOptimization.manualLossInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            submitEffortManualLoss();
        }
    });
    
    // Effort survey form
    effortOptimization.surveyForm.addEventListener('submit', (e) => {
        e.preventDefault();
        submitEffortSurvey();
    });
    
    // Effort survey cancel button
    effortOptimization.surveyCancelBtn.addEventListener('click', () => {
        effortOptimization.surveyCard.style.display = 'none';
    });
    

    
    // Effort exit button
    effortOptimization.exitBtn.addEventListener('click', () => {
        exitEffortOptimization();
    });
    
    // Effort decision buttons
    document.getElementById('effort-restart-btn').addEventListener('click', async () => {
        console.log('Effort restart button clicked');
        console.log('effortState.userId:', effortState.userId);
        try {
            await startEffortOptimizationSession(effortState.userId, true); // restart mode
        } catch (error) {
            console.error('Error in restart button handler:', error);
            showNotification(`Failed to restart effort optimization: ${error.message}`, 'danger');
        }
    });
    
    document.getElementById('effort-continue-btn').addEventListener('click', async () => {
        console.log('Effort continue button clicked');
        console.log('effortState.userId:', effortState.userId);
        try {
            await startEffortOptimizationSession(effortState.userId, false); // continue mode
        } catch (error) {
            console.error('Error in continue button handler:', error);
            showNotification(`Failed to continue effort optimization: ${error.message}`, 'danger');
        }
    });
    
    // Effort Back to Objectives button
    document.getElementById('effort-decision-back-btn').addEventListener('click', () => {
        // Hide decision card and go back to objective selection
        document.getElementById('effort-bo-decision-card').classList.add('d-none');
        bo.objectiveSelectCard.classList.remove('d-none');
    });
    
    // Effort first geometry input event listeners
    effortOptimization.setFirstGeometryBtn.addEventListener('click', () => {
        setFirstEffortGeometry();
    });
    
    effortOptimization.cancelFirstGeometryBtn.addEventListener('click', () => {
        hideFirstEffortGeometryInput();
        // Go back to objective selection
        exitEffortOptimization();
    });

    // --- Kick things off ---
    
    // Debug: Test if effort optimization screen element exists
    const effortScreen = document.getElementById('effort-optimization-screen');
    console.log('Effort optimization screen element:', effortScreen);
    console.log('Effort optimization screen classes:', effortScreen ? effortScreen.className : 'null');
    console.log('Effort optimization screen style:', effortScreen ? effortScreen.style.display : 'null');
    
    showScreen('modeSelection');
    loadInitialData();
    updateHomeButtonState();
}); 