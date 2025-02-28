document.addEventListener('DOMContentLoaded', function () {
    // Server URL - update this with your Flask server address
    const SERVER_URL = 'http://localhost:5000';

    // Initialize data storage
    let experimentData = {
        currentGeometry: {
            alpha: 0,
            beta: 0,
            gamma: 0,
            delta: 0
        },
        subjectiveMetrics: {
            pain: 0,
            effort: 0,
            instability: 0
        },
        accelerometerData: null,
        analysisResults: {
            subjective: {
                pain: 0,
                effort: 0,
                instability: 0,
                weightedSum: 0
            },
            objective: {
                pain: 0,
                effort: 0,
                instability: 0,
                weightedSum: 0
            }
        },
        history: []
    };

    // Initialize Bootstrap tabs
    const triggerTabList = document.querySelectorAll('#main-tabs button');
    triggerTabList.forEach(triggerEl => {
        triggerEl.addEventListener('click', event => {
            event.preventDefault();
            new bootstrap.Tab(triggerEl).show();
        });
    });

    // Setup event listeners for range inputs to update displayed values
    document.getElementById('pain').addEventListener('input', function () {
        document.getElementById('pain-value').textContent = this.value;
    });

    document.getElementById('effort').addEventListener('input', function () {
        document.getElementById('effort-value').textContent = this.value;
    });

    document.getElementById('instability').addEventListener('input', function () {
        document.getElementById('instability-value').textContent = this.value;
    });

    // Save geometry
    document.getElementById('save-geometry').addEventListener('click', function () {
        experimentData.currentGeometry = {
            alpha: parseFloat(document.getElementById('alpha').value),
            beta: parseFloat(document.getElementById('beta').value),
            gamma: parseFloat(document.getElementById('gamma').value),
            delta: parseFloat(document.getElementById('delta').value)
        };

        showNotification('Geometry saved');
    });

    // Save subjective metrics
    document.getElementById('save-metrics').addEventListener('click', function () {
        experimentData.subjectiveMetrics = {
            pain: parseInt(document.getElementById('pain').value),
            effort: parseInt(document.getElementById('effort').value),
            instability: parseInt(document.getElementById('instability').value)
        };

        showNotification('Subjective metrics saved');
    });

    // Upload accelerometer data
    document.getElementById('upload-data').addEventListener('click', function () {
        const fileInput = document.getElementById('accel-data');
        const file = fileInput.files[0];

        if (!file) {
            showNotification('Please select a file first', 'danger');
            return;
        }

        // Create a FormData object for file upload
        const formData = new FormData();
        formData.append('file', file);

        // Show loading state
        const uploadBtn = document.getElementById('upload-data');
        const originalBtnText = uploadBtn.textContent;
        uploadBtn.disabled = true;
        uploadBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Uploading...';

        // Upload file to the server
        fetch(`${SERVER_URL}/api/upload`, {
            method: 'POST',
            body: formData
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                // Store the file ID returned by the server
                experimentData.accelerometerData = data.fileId;

                // Reset button state
                uploadBtn.disabled = false;
                uploadBtn.textContent = originalBtnText;

                showNotification('Data uploaded');

                // Auto-switch to Analysis tab after upload
                document.getElementById('analysis-tab').click();
            })
            .catch(error => {
                console.error('Error uploading file:', error);
                showNotification('Upload failed: ' + error.message, 'danger');

                // Reset button state
                uploadBtn.disabled = false;
                uploadBtn.textContent = originalBtnText;
            });
    });

    // Run analysis
    document.getElementById('run-analysis').addEventListener('click', function () {
        // Check if we have all required data
        if (!experimentData.accelerometerData) {
            showNotification('Please upload accelerometer data first', 'danger');
            return;
        }

        // Get analysis parameters
        const numBouts = parseInt(document.getElementById('num-bouts').value);
        const weightings = {
            pain: parseFloat(document.getElementById('pain-weight').value),
            effort: parseFloat(document.getElementById('effort-weight').value),
            instability: parseFloat(document.getElementById('instability-weight').value)
        };

        // Check if weights sum to approximately 1
        const totalWeight = Number((weightings.pain + weightings.effort + weightings.instability).toFixed(2));
        if (totalWeight < 0.99 || totalWeight > 1.01) {
            showNotification('Weights should sum to 1.0 (current: ' + totalWeight + ')', 'warning');
        }

        // Call the server API for analysis
        runAnalysis(numBouts, weightings);
    });

    // Run Bayesian Optimization
    document.getElementById('run-optimisation').addEventListener('click', function () {
        // Check if we have enough data
        if (experimentData.history.length < 2) {
            showNotification('Need at least 2 experiments for optimization', 'danger');
            return;
        }

        const kernelType = document.getElementById('kernel-type').value;
        const iterations = parseInt(document.getElementById('n-iterations').value);

        // Call the server API for optimization
        runOptimization(kernelType, iterations);
    });

    // Function to call the analysis API
    function runAnalysis(numBouts, weightings) {
        // Show loading state
        const runAnalysisBtn = document.getElementById('run-analysis');
        const originalBtnText = runAnalysisBtn.textContent;
        runAnalysisBtn.disabled = true;
        runAnalysisBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Analyzing...';

        // Prepare data for API call
        const requestData = {
            accelerometerData: experimentData.accelerometerData,
            numBouts: numBouts,
            weightings: weightings,
            subjectiveMetrics: experimentData.subjectiveMetrics
        };

        // Call the API
        fetch(`${SERVER_URL}/api/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                // Process results from Python backend
                const subjective = data.subjective;
                const objective = data.objective;

                // Store results
                experimentData.analysisResults = {
                    subjective: subjective,
                    objective: objective
                };

                // Update the UI
                updateAnalysisTable(subjective, objective);

                // Add to experiment history
                const historyEntry = {
                    trial: experimentData.history.length + 1,
                    geometry: { ...experimentData.currentGeometry },
                    weightedLoss: ((parseFloat(subjective.weightedSum) + parseFloat(objective.weightedSum)) / 2).toFixed(2)
                };

                experimentData.history.push(historyEntry);
                updateHistoryTable();

                // Reset button state
                runAnalysisBtn.disabled = false;
                runAnalysisBtn.textContent = originalBtnText;

                showNotification('Analysis complete');
            })
            .catch(error => {
                console.error('Error during analysis:', error);
                showNotification('Analysis failed: ' + error.message, 'danger');

                // Reset button state
                runAnalysisBtn.disabled = false;
                runAnalysisBtn.textContent = originalBtnText;
            });
    }

    // Function to call the optimization API
    function runOptimization(kernelType, iterations) {
        // Show loading state
        const optimizeBtn = document.getElementById('run-optimisation');
        const originalBtnText = optimizeBtn.textContent;
        optimizeBtn.disabled = true;
        optimizeBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Optimizing...';

        // Prepare data for API call
        const requestData = {
            history: experimentData.history,
            kernelType: kernelType,
            iterations: iterations
        };

        // Call the API
        fetch(`${SERVER_URL}/api/optimize`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                // Process results from Python backend
                const optimizedGeometry = data.geometry;
                const expectedLoss = data.expectedLoss;

                // Update the UI
                document.getElementById('opt-alpha').textContent = optimizedGeometry.alpha.toFixed(2);
                document.getElementById('opt-beta').textContent = optimizedGeometry.beta.toFixed(2);
                document.getElementById('opt-gamma').textContent = optimizedGeometry.gamma.toFixed(2);
                document.getElementById('opt-delta').textContent = optimizedGeometry.delta.toFixed(2);
                document.getElementById('expected-loss').textContent = expectedLoss.toFixed(2);

                // Reset button state
                optimizeBtn.disabled = false;
                optimizeBtn.textContent = originalBtnText;

                showNotification('Optimization complete');

                // Update the form with the optimized values
                document.getElementById('alpha').value = optimizedGeometry.alpha.toFixed(2);
                document.getElementById('beta').value = optimizedGeometry.beta.toFixed(2);
                document.getElementById('gamma').value = optimizedGeometry.gamma.toFixed(2);
                document.getElementById('delta').value = optimizedGeometry.delta.toFixed(2);
            })
            .catch(error => {
                console.error('Error during optimization:', error);
                showNotification('Optimization failed: ' + error.message, 'danger');

                // Reset button state
                optimizeBtn.disabled = false;
                optimizeBtn.textContent = originalBtnText;
            });
    }

    // Update the analysis results table
    function updateAnalysisTable(subjective, objective) {
        const subjRow = document.getElementById('subjective-loss');
        const objRow = document.getElementById('objective-loss');

        subjRow.children[1].textContent = subjective.pain.toFixed(2);
        subjRow.children[2].textContent = subjective.effort.toFixed(2);
        subjRow.children[3].textContent = subjective.instability.toFixed(2);
        subjRow.children[4].textContent = subjective.weightedSum.toFixed(2);

        objRow.children[1].textContent = objective.pain.toFixed(2);
        objRow.children[2].textContent = objective.effort.toFixed(2);
        objRow.children[3].textContent = objective.instability.toFixed(2);
        objRow.children[4].textContent = objective.weightedSum.toFixed(2);
    }

    // Update the experiment history table
    function updateHistoryTable() {
        const tbody = document.querySelector('#experiment-history tbody');
        tbody.innerHTML = '';

        experimentData.history.forEach(entry => {
            const row = document.createElement('tr');

            row.innerHTML = `
                <td>${entry.trial}</td>
                <td>${entry.geometry.alpha.toFixed(2)}</td>
                <td>${entry.geometry.beta.toFixed(2)}</td>
                <td>${entry.geometry.gamma.toFixed(2)}</td>
                <td>${entry.geometry.delta.toFixed(2)}</td>
                <td>${entry.weightedLoss}</td>
            `;

            tbody.appendChild(row);
        });

        // Auto-switch to Optimisation tab after adding to history
        if (experimentData.history.length >= 2) {
            document.getElementById('optimisation-tab').click();
        }
    }

    // Bootstrap Toast notification system
    function showNotification(message, type = 'success') {
        // Create toast container if it doesn't exist
        let toastContainer = document.querySelector('.toast-container');
        if (!toastContainer) {
            toastContainer = document.createElement('div');
            toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
            toastContainer.style.zIndex = "1050";
            document.body.appendChild(toastContainer);
        }

        // Create a Bootstrap toast
        const toastEl = document.createElement('div');
        toastEl.className = `toast align-items-center text-white bg-${type} border-0`;
        toastEl.setAttribute('role', 'alert');
        toastEl.setAttribute('aria-live', 'assertive');
        toastEl.setAttribute('aria-atomic', 'true');

        const toastBody = document.createElement('div');
        toastBody.className = 'd-flex';

        const messageDiv = document.createElement('div');
        messageDiv.className = 'toast-body';
        messageDiv.textContent = message;

        const closeButton = document.createElement('button');
        closeButton.type = 'button';
        closeButton.className = 'btn-close btn-close-white me-2 m-auto';
        closeButton.setAttribute('data-bs-dismiss', 'toast');
        closeButton.setAttribute('aria-label', 'Close');

        toastBody.appendChild(messageDiv);
        toastBody.appendChild(closeButton);
        toastEl.appendChild(toastBody);
        toastContainer.appendChild(toastEl);

        // Initialize and show toast
        const toast = new bootstrap.Toast(toastEl, {
            animation: true,
            autohide: true,
            delay: 2000
        });
        toast.show();

        // Remove from DOM after hiding
        toastEl.addEventListener('hidden.bs.toast', function () {
            toastEl.remove();
        });
    }
}); 