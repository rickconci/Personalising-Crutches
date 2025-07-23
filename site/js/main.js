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
        Object.values(screens).forEach(screen => screen.classList.add('d-none'));
        screens[screenName].classList.remove('d-none');
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

    // Start with mode selection screen
    loadInitialData();

    // --- Initial UI Setup based on Config ---
    // Metabolic cost removed from grid search mode


    function populateParticipantSelects() {
        // Populate for participant selection screen and BO mode
        [systematic.participantSelect, bo.participantSelect].forEach(select => {
            select.innerHTML = '<option selected disabled>Choose...</option>';
            appState.participants.forEach(p => {
                const option = document.createElement('option');
                option.value = p.id;
                option.textContent = p.full_name;
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
        appState.mode = null;
        showScreen('modeSelection');
    });

    // --- Participant Selection Logic ---
    systematic.participantSelect.addEventListener('change', async (e) => {
        const participantId = e.target.value;
        systematic.deleteParticipantBtn.disabled = !participantId;
        
        if (!participantId) {
            systematic.gridSearchTables.innerHTML = `<div class="text-center text-muted">Select a participant to see grid search trials.</div>`;
            systematic.instabilityPlot3D.innerHTML = `<div class="text-center text-muted pt-5">Select a participant to view the 3D plot.</div>`;
            renderParticipantTrialsTable(null);
            return;
        }
        
        await refreshParticipantView(participantId);
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
            showNotification(`Participant "${newParticipant.full_name}" created successfully!`, 'success');

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
        const participantTrials = appState.trials.filter(t => t.participant_id === numericParticipantId);
        
        if (!participant) {
            systematic.participantTrialsTitle.textContent = 'Participant Trials';
            systematic.participantTrialsTableBody.innerHTML = `<tr><td colspan="13" class="text-center text-muted">Participant not found.</td></tr>`;
            return;
        }
        
        systematic.participantTrialsTitle.textContent = `${participant.name}'s Trials`;
        
        if (participantTrials.length === 0) {
            systematic.participantTrialsTableBody.innerHTML = `<tr><td colspan="13" class="text-center text-muted">No trials recorded yet for this participant.</td></tr>`;
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
                <td>${t.survey_responses?.tlx_score !== undefined ? t.survey_responses.tlx_score : '-'}</td>
                <td class="text-center">
                    <button class="btn btn-sm btn-outline-danger delete-trial-btn" data-trial-id="${t.id}" data-geometry-id="${t.geometry_id}">
                        🗑️
                    </button>
                </td>
            `;
            systematic.participantTrialsTableBody.appendChild(row);
        });
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
        surveyResponses['tlx_score'] = tlxScore;
        
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
            }
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
        const userId = bo.participantSelect.options[bo.participantSelect.selectedIndex].text;
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
            bo.userInfo.textContent = `Active BO Session for: ${userId}`;
            renderBOHistory();
            getNextBOSuggestion();
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

    // --- Kick things off ---
    showScreen('modeSelection');
    loadInitialData();
}); 