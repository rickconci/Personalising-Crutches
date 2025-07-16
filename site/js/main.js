document.addEventListener('DOMContentLoaded', function () {
    // By leaving this empty, the browser will make API requests to the same
    // origin that served the page, which eliminates all CORS issues.
    const SERVER_URL = '';

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
        remainingGeometriesList: document.getElementById('remaining-geometries'),
        trialsTableBody: document.querySelector('#all-trials-table tbody'),
        trialRunnerCol: document.getElementById('trial-runner-col'),
        trialRunnerTitle: document.getElementById('trial-runner-title'),
        trialForm: document.getElementById('systematic-trial-form'),
        connectDeviceBtn: document.getElementById('connect-device-btn'),
        deviceStatus: document.getElementById('device-status'),
        stopwatch: document.getElementById('stopwatch'),
        startStopBtn: document.getElementById('start-stop-btn'),
        analyzeDataBtn: document.getElementById('analyze-data-btn'),
        plotsArea: document.getElementById('plots-area'),
        forcePlotDiv: document.getElementById('force-plot-div'),
        histPlotDiv: document.getElementById('hist-plot-div'),
        stepInteractionArea: document.getElementById('step-interaction-area'),
        stepList: document.getElementById('step-list'),
        stepCount: document.getElementById('step-count'),
        metricsAndSurveyArea: document.getElementById('metrics-and-survey-area'),
        instabilityLossValue: document.getElementById('instability-loss-value'),
        effortLossValue: document.getElementById('effort-loss-value'),
        surveyArea: document.getElementById('survey-area'), // This is now part of the above
        uploadDataBtn: document.getElementById('upload-data-btn'),
        fileUploadInput: document.getElementById('file-upload-input'),
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

    // --- Initial Load ---
    async function loadInitialData() {
        try {
            appState.participants = await apiRequest('/api/participants');
            appState.geometries = await apiRequest('/api/geometries');
            appState.trials = await apiRequest('/api/trials');
            populateParticipantSelects();
            renderAllTrialsTable();
        } catch (error) {
            showNotification(`Failed to load initial data: ${error.message}`, 'danger');
        }
    }

    function populateParticipantSelects() {
        // Populate for both systematic and BO mode
        [systematic.participantSelect, bo.participantSelect].forEach(select => {
            select.innerHTML = '<option selected disabled>Choose...</option>';
            appState.participants.forEach(p => {
                const option = document.createElement('option');
                option.value = p.id;
                option.textContent = p.user_id;
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


    // --- Systematic Mode Logic ---
    function renderAllTrialsTable() {
        systematic.trialsTableBody.innerHTML = '';
        if (appState.trials.length === 0) {
            systematic.trialsTableBody.innerHTML = `<tr><td colspan="7" class="text-center text-muted">No trials recorded yet.</td></tr>`;
            return;
        }
        appState.trials.forEach(t => {
            const row = document.createElement('tr');
            row.dataset.participantId = t.participant_id; // Add data attribute for filtering
            row.innerHTML = `
                <td>${new Date(t.timestamp).toLocaleString()}</td>
                <td>${t.participant_user_id}</td>
                <td>${t.geometry_name}</td>
                <td>${t.alpha?.toFixed(1) ?? '-'}</td>
                <td>${t.beta?.toFixed(1) ?? '-'}</td>
                <td>${t.gamma?.toFixed(1) ?? '-'}</td>
                <td><span class="badge bg-${t.source === 'bo' ? 'success' : 'primary'}">${t.source}</span></td>
            `;
            systematic.trialsTableBody.appendChild(row);
        });
    }

    function filterTrialsTable(participantId) {
        const rows = systematic.trialsTableBody.querySelectorAll('tr');
        rows.forEach(row => {
            // If participantId is null/empty, show all rows.
            // Otherwise, show only rows that match the participantId.
            if (!participantId || row.dataset.participantId === participantId) {
                row.style.display = '';
            } else {
                row.style.display = 'none';
            }
        });
    }

    systematic.participantSelect.addEventListener('change', async (e) => {
        const participantId = e.target.value;
        filterTrialsTable(participantId); // Filter the main table

        if (!participantId) {
            systematic.remainingGeometriesList.innerHTML = `<li class="list-group-item text-center text-muted">Select a participant to see remaining trials.</li>`;
            return;
        };
        try {
            const data = await apiRequest(`/api/participants/${participantId}`);
            appState.currentParticipant = data.participant;
            renderRemainingGeometries(data.remaining_geometries);
        } catch (error) {
            showNotification(`Error fetching participant details: ${error.message}`, 'danger');
        }
    });

    function renderRemainingGeometries(geometries) {
        systematic.remainingGeometriesList.innerHTML = '';
        if (geometries.length === 0) {
            systematic.remainingGeometriesList.innerHTML = `<li class="list-group-item text-center text-muted">All systematic trials complete for this participant!</li>`;
            return;
        }
        geometries.forEach(g => {
            const li = document.createElement('li');
            li.className = 'list-group-item d-flex justify-content-between align-items-center';
            li.innerHTML = `
                <span><strong>${g.name}</strong> (α:${g.alpha}, β:${g.beta}, γ:${g.gamma})</span>
                <button class="btn btn-sm btn-primary" data-geom-id="${g.id}">Start Trial</button>
            `;
            systematic.remainingGeometriesList.appendChild(li);
        });
    }

    systematic.remainingGeometriesList.addEventListener('click', (e) => {
        if (e.target.tagName === 'BUTTON') {
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
    });

    function resetTrialState() {
        // Reset buttons
        systematic.connectDeviceBtn.disabled = false;
        systematic.uploadDataBtn.disabled = false;
        systematic.startStopBtn.disabled = true;
        systematic.startStopBtn.innerHTML = '<i class="fas fa-play"></i> Start Trial';
        systematic.startStopBtn.classList.replace('btn-danger', 'btn-success');
        systematic.analyzeDataBtn.disabled = true;
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
        systematic.effortLossValue.textContent = '-';
        systematic.stepInteractionArea.classList.add('d-none');
        systematic.stepList.innerHTML = '';
        uploadedFile = null;
        trialState.metrics = null;
        trialState.steps = [];
        trialState.rawData = null;
    }

    // --- Step Editing & Recalculation ---

    function recalculateAndUpdate() {
        if (!trialState.rawData || trialState.steps.length < 2) {
            // Not enough data to calculate, clear dependent views
            systematic.instabilityLossValue.textContent = 'N/A';
            systematic.effortLossValue.textContent = 'N/A';
            Plotly.purge(systematic.histPlotDiv);
            return;
        }

        const steps = trialState.steps.sort((a, b) => a - b);
        const df = trialState.rawData;

        // 1. Recalculate Metrics
        const durations = steps.slice(1).map((step, i) => step - steps[i]);
        const meanDuration = durations.reduce((a, b) => a + b, 0) / durations.length;
        const instability_loss = durations.map(d => Math.pow(d - meanDuration, 2)).reduce((a, b) => a + b, 0) / durations.length;

        const effort_loss = instability_loss * 1500 + Math.random() * 5; // Same simulation
        trialState.metrics = { instability_loss, effort_loss };

        // 2. Update Metric Displays
        systematic.instabilityLossValue.textContent = instability_loss.toFixed(4);
        systematic.effortLossValue.textContent = effort_loss.toFixed(4);

        // 3. Re-render Histogram
        Plotly.react(systematic.histPlotDiv, [{
            x: durations,
            type: 'histogram',
            nbinsx: 20,
            name: 'Step Durations'
        }], { title: "Step Duration Distribution", xaxis_title: "Duration (s)", yaxis_title: "Count" }, { responsive: true });

        // 4. Re-render Step List
        renderStepList();

        // 5. Update markers on the main plot
        const newStepX = [];
        const newStepY = [];

        function findClosestIndex(array, value) {
            let bestIndex = 0;
            let bestDiff = Infinity;
            for (let i = 0; i < array.length; i++) {
                const diff = Math.abs(array[i] - value);
                if (diff < bestDiff) {
                    bestDiff = diff;
                    bestIndex = i;
                }
            }
            return bestIndex;
        }

        steps.forEach(stepTime => {
            const index = findClosestIndex(df.timestamp, stepTime);
            newStepX.push(df.timestamp[index]); // Use the actual timestamp from data for precision
            newStepY.push(df.force[index]);
        });

        // Use Plotly.react for a more robust update.
        // Get a fresh copy of the plot's data traces, keeping the non-step traces.
        const originalTraces = systematic.forcePlotDiv.data.slice(0, 3);

        // Create a new trace object for the updated steps
        const newStepTrace = {
            type: 'scatter',
            mode: 'markers',
            name: 'Detected Steps',
            x: newStepX,
            y: newStepY,
            marker: { symbol: 'x', color: 'red', size: 10 }
        };

        // Combine the original traces with the new step trace
        const newData = [...originalTraces, newStepTrace];

        // Use Plotly.newPlot to force a complete redraw, which is more robust
        // than Plotly.react for this type of dynamic update.
        Plotly.newPlot(systematic.forcePlotDiv, newData, systematic.forcePlotDiv.layout);
    }

    function renderStepList() {
        systematic.stepList.innerHTML = '';
        systematic.stepCount.textContent = trialState.steps.length;
        trialState.steps.forEach((stepTime, index) => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${stepTime.toFixed(3)}s</td>
                <td class="text-end">
                    <button class="btn btn-sm btn-outline-info py-0 px-1 inspect-step-btn" data-time="${stepTime}"><i class="fas fa-search-plus"></i></button>
                    <button class="btn btn-sm btn-outline-danger py-0 px-1 delete-step-btn" data-index="${index}"><i class="fas fa-trash-alt"></i></button>
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
            recalculateAndUpdate();
        }

        if (target.classList.contains('inspect-step-btn')) {
            const time = parseFloat(target.dataset.time);
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

    systematic.uploadDataBtn.addEventListener('click', () => {
        systematic.fileUploadInput.click();
    });

    systematic.fileUploadInput.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (!file) return;

        uploadedFile = file;
        showNotification(`File "${file.name}" selected. Ready to analyze.`, 'info');

        // Update UI for file upload path
        systematic.connectDeviceBtn.disabled = true;
        systematic.uploadDataBtn.disabled = true;
        systematic.startStopBtn.disabled = true;
        systematic.analyzeDataBtn.disabled = false;
        systematic.deviceStatus.textContent = `File: ${file.name}`;
        systematic.deviceStatus.classList.replace('alert-secondary', 'alert-info');
    });

    function onDisconnected() {
        showNotification('Device disconnected.', 'warning');
        resetTrialState();
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
            systematic.analyzeDataBtn.disabled = false;
        } else {
            // Starting the trial
            await startDataCollection();
            startStopwatch();
            systematic.startStopBtn.innerHTML = '<i class="fas fa-stop"></i> Stop Trial';
            systematic.startStopBtn.classList.replace('btn-success', 'btn-danger');
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

    systematic.analyzeDataBtn.addEventListener('click', async () => {
        const participantId = parseInt(systematic.trialForm.querySelector('#systematic-participant-id').value);
        const geometryId = parseInt(systematic.trialForm.querySelector('#systematic-geometry-id').value);

        try {
            let results;
            if (uploadedFile) {
                // --- Handle File Upload ---
                const formData = new FormData();
                formData.append('file', uploadedFile);
                formData.append('participantId', participantId);
                formData.append('geometryId', geometryId);

                // We cannot use the apiRequest helper for multipart/form-data
                const response = await fetch(`${SERVER_URL}/api/trials/analyze`, {
                    method: 'POST',
                    body: formData,
                });
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.error);
                }
                results = await response.json();

            } else {
                // --- Handle Live Data ---
                const payload = {
                    participantId,
                    geometryId,
                    trialData: trialDataBuffer,
                };
                results = await apiRequest('/api/trials/analyze', 'POST', payload);
            }

            showNotification(results.message, 'info');

            // --- Render Plotly Charts ---
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
                    throw new Error("Could not find plot data in server response.");
                }
            }

            // This part changes significantly

            // 1. Store steps and raw data for editing
            trialState.steps = results.steps.sort((a, b) => a - b);
            const rawDataResponse = await fetch(`${SERVER_URL}${results.plots.timeseries}?t=${new Date().getTime()}`);
            const rawDataHtml = await rawDataResponse.text();
            const tempDiv = document.createElement('div');
            tempDiv.innerHTML = rawDataHtml;
            const plotDiv = tempDiv.querySelector('.plotly-graph-div');
            const plotData = JSON.parse(plotDiv.dataset.raw);

            trialState.rawData = {
                timestamp: plotData.data[0].x,
                force: plotData.data[0].y
            }

            // 2. Initial Plotting
            await renderPlot(systematic.forcePlotDiv, results.plots.timeseries);

            // Re-attach the plotly click listener now that the plot exists
            systematic.forcePlotDiv.on('plotly_click', (data) => {
                // Guard: Only add steps if the user clicks on the main force trace (curveNumber 0)
                const point = data.points[0];
                if (point.curveNumber !== 0) {
                    return;
                }
                const clickedTime = point.x;
                if (trialState.steps.includes(clickedTime)) return;
                trialState.steps.push(clickedTime);
                recalculateAndUpdate();
            });

            await renderPlot(systematic.histPlotDiv, results.plots.histogram);

            // 3. Initial Metric Display
            trialState.metrics = results.metrics;
            systematic.instabilityLossValue.textContent = results.metrics.instability_loss.toFixed(4);
            systematic.effortLossValue.textContent = results.metrics.effort_loss.toFixed(4);

            // 4. Render the interactive step list
            renderStepList();
            systematic.stepInteractionArea.classList.remove('d-none');

            systematic.plotsArea.classList.remove('d-none');
            systematic.metricsAndSurveyArea.classList.remove('d-none');
            systematic.analyzeDataBtn.disabled = true;

        } catch (error) {
            showNotification(`Analysis failed: ${error.message}`, 'danger');
        }
    });


    systematic.saveParticipantBtn.addEventListener('click', async () => {
        const userId = systematic.newParticipantForm.querySelector('#new-participant-id').value.trim();
        if (!userId) {
            showNotification('Participant ID is required.', 'danger');
            return;
        }

        const payload = {
            userId: userId,
            userCharacteristics: {
                age: parseInt(document.getElementById('char-age').value),
                sex: document.getElementById('char-sex').value,
                height: parseFloat(document.getElementById('char-height').value),
                weight: parseFloat(document.getElementById('char-weight').value),
                forearm_length: parseFloat(document.getElementById('char-forearm').value),
                activity_level: document.getElementById('char-activity').value,
                previous_crutch_experience: document.querySelector('input[name="crutch-experience"]:checked').value === 'true',
            }
        };

        try {
            const newParticipant = await apiRequest('/api/participants', 'POST', payload);
            showNotification(`Participant "${newParticipant.user_id}" created successfully!`, 'success');

            systematic.createParticipantModal.hide();
            systematic.newParticipantForm.reset();

            // Refresh participant list and select the new one
            await loadInitialData();
            systematic.participantSelect.value = newParticipant.id;
            systematic.participantSelect.dispatchEvent(new Event('change'));

        } catch (error) {
            showNotification(`Error creating participant: ${error.message}`, 'danger');
        }
    });

    systematic.trialForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        // This form now handles saving the final results after analysis
        const payload = {
            participantId: parseInt(systematic.trialForm.querySelector('#systematic-participant-id').value),
            geometryId: parseInt(systematic.trialForm.querySelector('#systematic-geometry-id').value),
            surveyResponses: {
                effort: parseInt(document.getElementById('systematic-effort').value),
                pain: parseInt(document.getElementById('systematic-pain').value),
                stability: parseInt(document.getElementById('systematic-instability').value),
            },
            metrics: trialState.metrics // Attach the calculated metrics
        };

        try {
            const newTrial = await apiRequest('/api/trials/save', 'POST', payload);
            showNotification('Systematic trial recorded successfully!', 'success');

            systematic.trialRunnerCol.classList.add('d-none'); // Hide trial runner
            systematic.trialForm.reset();
            resetTrialState(); // Fully reset the panel

            // Refresh data
            await loadInitialData();
            // Refresh the remaining geometries list
            systematic.participantSelect.dispatchEvent(new Event('change'));

        } catch (error) {
            showNotification(`Error submitting trial: ${error.message}`, 'danger');
        }
    });


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