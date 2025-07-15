document.addEventListener('DOMContentLoaded', function () {
    const SERVER_URL = 'http://localhost:5000';

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
        trialModal: new bootstrap.Modal(document.getElementById('trial-modal')),
        trialForm: document.getElementById('systematic-trial-form'),
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
            // Set hidden inputs in the modal form
            systematic.trialForm.querySelector('#systematic-participant-id').value = appState.currentParticipant.id;
            systematic.trialForm.querySelector('#systematic-geometry-id').value = geomId;
            systematic.trialModal.show();
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
        const formData = new FormData();
        const fileInput = systematic.trialForm.querySelector('input[type="file"]');

        const data = {
            participantId: parseInt(systematic.trialForm.querySelector('#systematic-participant-id').value),
            geometryId: parseInt(systematic.trialForm.querySelector('#systematic-geometry-id').value),
            surveyResponses: {
                effort: parseInt(systematic.trialForm.querySelector('#systematic-effort').value),
                pain: parseInt(systematic.trialForm.querySelector('#systematic-pain').value),
                stability: parseInt(systematic.trialForm.querySelector('#systematic-instability').value),
            }
        };

        formData.append('data', JSON.stringify(data));
        formData.append('file', fileInput.files[0]);

        try {
            // NOTE: Can't use apiRequest helper for multipart/form-data
            const response = await fetch(`${SERVER_URL}/api/trials`, { method: 'POST', body: formData });
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error);
            }
            const newTrial = await response.json();

            showNotification('Systematic trial recorded successfully!', 'success');
            systematic.trialModal.hide();
            systematic.trialForm.reset();

            // Refresh data
            loadInitialData();
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