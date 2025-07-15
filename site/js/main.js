document.addEventListener('DOMContentLoaded', function () {
    const SERVER_URL = 'http://localhost:5000';

    // --- State Management ---
    let appState = {
        userId: null,
        objective: null,
        suggestedGeometry: null,
        history: [],
    };

    // --- Element Selectors ---
    const screens = {
        setup: document.getElementById('setup-screen'),
        main: document.getElementById('main-screen'),
    };

    const setupForm = {
        userId: document.getElementById('user-id'),
        objective: document.getElementById('objective'),
        charHeight: document.getElementById('char-height'),
        charWeight: document.getElementById('char-weight'),
        charForearm: document.getElementById('char-forearm'),
        charFitness: document.getElementById('char-fitness'),
        startButton: document.getElementById('start-experiment'),
    };

    const mainScreen = {
        userInfo: document.getElementById('user-info'),
        suggestionBox: document.getElementById('suggestion-box'),
        acceptBtn: document.getElementById('accept-geometry'),
        rejectBtn: document.getElementById('reject-geometry'),
        alternativeBtn: document.getElementById('alternative-geometry'),
        trialCard: document.getElementById('trial-recording-card'),
        historyTable: document.querySelector('#experiment-history tbody'),
        bestTrialBox: document.getElementById('best-trial-box'),
        optPlot: document.getElementById('opt-plot'),
    };

    const trialForm = {
        alpha: document.getElementById('current-alpha'),
        beta: document.getElementById('current-beta'),
        gamma: document.getElementById('current-gamma'),
        fileInput: document.getElementById('accel-data'),
        pain: document.getElementById('pain'),
        effort: document.getElementById('effort'),
        instability: document.getElementById('instability'),
        painValue: document.getElementById('pain-value'),
        effortValue: document.getElementById('effort-value'),
        instabilityValue: document.getElementById('instability-value'),
        submitButton: document.getElementById('submit-trial'),
    };

    const rejectModal = {
        modal: new bootstrap.Modal(document.getElementById('reject-modal')),
        penaltyInput: document.getElementById('penalty-loss'),
        submitBtn: document.getElementById('submit-penalty'),
    };

    // --- UI Update Functions ---
    function showScreen(screenName) {
        Object.values(screens).forEach(screen => screen.classList.add('d-none'));
        screens[screenName].classList.remove('d-none');
    }

    function updateSuggestionBox(geometry) {
        if (!geometry) {
            mainScreen.suggestionBox.innerHTML = `<p class="text-muted">Could not get a suggestion.</p>`;
            return;
        }
        appState.suggestedGeometry = geometry;
        mainScreen.suggestionBox.innerHTML = `
            <div class="row g-2">
                <div class="col-4">
                    <div class="stat-box">
                        <div class="stat-value">${geometry.alpha.toFixed(1)}</div>
                        <div class="stat-label">Alpha</div>
                    </div>
                </div>
                <div class="col-4">
                    <div class="stat-box">
                        <div class="stat-value">${geometry.beta.toFixed(1)}</div>
                        <div class="stat-label">Beta</div>
                    </div>
                </div>
                <div class="col-4">
                    <div class="stat-box">
                        <div class="stat-value">${geometry.gamma.toFixed(1)}</div>
                        <div class="stat-label">Gamma</div>
                    </div>
                </div>
            </div>
        `;
    }

    function updateHistoryTable() {
        mainScreen.historyTable.innerHTML = '';
        if (appState.history.length === 0) {
            mainScreen.historyTable.innerHTML = `<tr><td colspan="5" class="text-center text-muted">No trials yet.</td></tr>`;
            return;
        }
        appState.history.forEach((entry, index) => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${index + 1} ${entry.is_penalty ? '(Penalty)' : ''}</td>
                <td>${entry.alpha?.toFixed(1) ?? '-'}</td>
                <td>${entry.beta?.toFixed(1) ?? '-'}</td>
                <td>${entry.gamma?.toFixed(1) ?? '-'}</td>
                <td>${entry.Total_Combined_Loss.toFixed(2)}</td>
            `;
            mainScreen.historyTable.appendChild(row);
        });
    }

    function updateBestTrial() {
        fetch(`${SERVER_URL}/api/experiment/best?userId=${appState.userId}`)
            .then(handleResponse)
            .then(data => {
                if (data.alpha === undefined) {
                    mainScreen.bestTrialBox.innerHTML = `<p class="text-muted">${data.message || 'No valid trials yet.'}</p>`;
                } else {
                    mainScreen.bestTrialBox.innerHTML = `
                        <ul class="list-group list-group-flush">
                            <li class="list-group-item d-flex justify-content-between"><strong>Alpha:</strong> ${data.alpha.toFixed(1)}</li>
                            <li class="list-group-item d-flex justify-content-between"><strong>Beta:</strong> ${data.beta.toFixed(1)}</li>
                            <li class="list-group-item d-flex justify-content-between"><strong>Gamma:</strong> ${data.gamma.toFixed(1)}</li>
                            <li class="list-group-item d-flex justify-content-between"><strong>Loss:</strong> <span class="badge bg-primary">${data.Total_Combined_Loss.toFixed(3)}</span></li>
                        </ul>
                    `;
                }
            });
    }

    function showTrialCard(show, geometry) {
        if (show) {
            trialForm.alpha.value = geometry.alpha.toFixed(1);
            trialForm.beta.value = geometry.beta.toFixed(1);
            trialForm.gamma.value = geometry.gamma.toFixed(1);
            mainScreen.trialCard.classList.remove('d-none');
        } else {
            mainScreen.trialCard.classList.add('d-none');
        }
    }

    // --- API Helper Functions ---
    async function handleResponse(response) {
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || `HTTP error! Status: ${response.status}`);
        }
        return response.json();
    }

    function getNextGeometry() {
        mainScreen.suggestionBox.innerHTML = `<div class="spinner-border" role="status"><span class="visually-hidden">Loading...</span></div>`;
        fetch(`${SERVER_URL}/api/experiment/next-geometry?userId=${appState.userId}`)
            .then(handleResponse)
            .then(updateSuggestionBox)
            .catch(error => {
                showNotification(error.message, 'danger');
                mainScreen.suggestionBox.innerHTML = `<p class="text-danger">Error: ${error.message}</p>`;
            });
    }

    // --- Event Listeners ---
    setupForm.startButton.addEventListener('click', () => {
        const userId = setupForm.userId.value.trim();
        if (!userId) {
            showNotification('User ID is required.', 'warning');
            return;
        }

        const payload = {
            userId: userId,
            objective: setupForm.objective.value,
            userCharacteristics: {
                height: parseFloat(setupForm.charHeight.value),
                weight: parseFloat(setupForm.charWeight.value),
                forearm_length: parseFloat(setupForm.charForearm.value),
                fitness_level: parseFloat(setupForm.charFitness.value),
            }
        };

        setupForm.startButton.disabled = true;
        setupForm.startButton.innerHTML = `<span class="spinner-border spinner-border-sm"></span> Starting...`;

        fetch(`${SERVER_URL}/api/experiment/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
            .then(handleResponse)
            .then(data => {
                appState.userId = data.userId;
                appState.objective = payload.objective;
                appState.history = data.history || [];
                mainScreen.userInfo.textContent = `User: ${appState.userId} | Objective: ${appState.objective}`;
                updateHistoryTable();
                showScreen('main');
                getNextGeometry();
                updateBestTrial();
            })
            .catch(error => showNotification(error.message, 'danger'))
            .finally(() => {
                setupForm.startButton.disabled = false;
                setupForm.startButton.textContent = 'Start Experiment';
            });
    });

    mainScreen.acceptBtn.addEventListener('click', () => {
        showTrialCard(true, appState.suggestedGeometry);
    });

    mainScreen.alternativeBtn.addEventListener('click', () => {
        const altGeometry = {
            alpha: parseFloat(prompt("Enter alternative Alpha:", appState.suggestedGeometry.alpha)),
            beta: parseFloat(prompt("Enter alternative Beta:", appState.suggestedGeometry.beta)),
            gamma: parseFloat(prompt("Enter alternative Gamma:", appState.suggestedGeometry.gamma)),
        };
        if (!isNaN(altGeometry.alpha) && !isNaN(altGeometry.beta) && !isNaN(altGeometry.gamma)) {
            showTrialCard(true, altGeometry);
        }
    });

    mainScreen.rejectBtn.addEventListener('click', () => {
        rejectModal.modal.show();
    });

    rejectModal.submitBtn.addEventListener('click', () => {
        const payload = {
            userId: appState.userId,
            crutchGeometry: appState.suggestedGeometry,
            penaltyLoss: parseFloat(rejectModal.penaltyInput.value)
        };

        fetch(`${SERVER_URL}/api/experiment/penalty`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
            .then(handleResponse)
            .then(data => {
                appState.history = data.history;
                updateHistoryTable();
                getNextGeometry();
                rejectModal.modal.hide();
                showNotification('Penalty recorded.');
            })
            .catch(error => showNotification(error.message, 'danger'));
    });

    trialForm.submitButton.addEventListener('click', () => {
        const file = trialForm.fileInput.files[0];
        if (!file) {
            showNotification('Please select an accelerometer data file.', 'warning');
            return;
        }

        const trialData = {
            userId: appState.userId,
            crutchGeometry: {
                alpha: parseFloat(trialForm.alpha.value),
                beta: parseFloat(trialForm.beta.value),
                gamma: parseFloat(trialForm.gamma.value),
            },
            subjectiveMetrics: {
                effort_survey_answer: parseFloat(trialForm.effort.value),
                pain_survey_answer: parseFloat(trialForm.pain.value),
                stability_survey_answer: parseFloat(trialForm.instability.value),
            }
        };

        const formData = new FormData();
        formData.append('file', file);
        formData.append('data', JSON.stringify(trialData));

        trialForm.submitButton.disabled = true;
        trialForm.submitButton.innerHTML = `<span class="spinner-border spinner-border-sm"></span> Processing...`;

        fetch(`${SERVER_URL}/api/experiment/trial`, {
            method: 'POST',
            body: formData
        })
            .then(handleResponse)
            .then(data => {
                showNotification('Trial processed successfully.');
                appState.history = data.history;
                updateHistoryTable();
                updateBestTrial();
                if (data.plot_path) {
                    // Add a cache-busting query param
                    mainScreen.optPlot.src = `${SERVER_URL}${data.plot_path}?t=${new Date().getTime()}`;
                }
                showTrialCard(false);
                getNextGeometry();
            })
            .catch(error => showNotification(error.message, 'danger'))
            .finally(() => {
                trialForm.submitButton.disabled = false;
                trialForm.submitButton.textContent = 'Submit Trial';
            });
    });

    // Sliders to update their displayed values
    ['pain', 'effort', 'instability'].forEach(id => {
        document.getElementById(id).addEventListener('input', function () {
            document.getElementById(`${id}-value`).textContent = this.value;
        });
    });

    // --- Toast Notification ---
    function showNotification(message, type = 'success') {
        const toastContainer = document.querySelector('.toast-container');
        const toastEl = document.createElement('div');
        toastEl.className = `toast align-items-center text-white bg-${type} border-0`;
        toastEl.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">${message}</div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        `;
        toastContainer.appendChild(toastEl);
        const toast = new bootstrap.Toast(toastEl, { delay: 3000 });
        toast.show();
        toastEl.addEventListener('hidden.bs.toast', () => toastEl.remove());
    }

    // --- Initial Setup ---
    showScreen('setup');
}); 