/**
 * API client for Personalising Crutches FastAPI backend
 */

class CrutchAPI {
    constructor(baseURL = '/api') {
        this.baseURL = baseURL;
    }

    async request(endpoint, method = 'GET', data = null) {
        const options = {
            method: method,
            headers: {
                'Content-Type': 'application/json',
            },
        };

        if (data) {
            options.body = JSON.stringify(data);
        }

        try {
            const response = await fetch(`${this.baseURL}${endpoint}`, options);

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                // Better error message formatting for validation errors
                let errorMessage = `HTTP error! status: ${response.status}`;
                if (errorData.detail) {
                    if (Array.isArray(errorData.detail)) {
                        // FastAPI validation errors
                        errorMessage = errorData.detail.map(err =>
                            `${err.loc.join('.')}: ${err.msg}`
                        ).join(', ');
                    } else if (typeof errorData.detail === 'string') {
                        errorMessage = errorData.detail;
                    } else {
                        errorMessage = JSON.stringify(errorData.detail);
                    }
                }
                console.error('API error details:', errorData);
                throw new Error(errorMessage);
            }

            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                return await response.json();
            } else {
                return await response.text();
            }
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    }

    // Participant endpoints
    async getParticipants() {
        return this.request('/experiments/participants');
    }

    async createParticipant(participantData) {
        return this.request('/experiments/participants', 'POST', participantData);
    }

    async getParticipant(participantId) {
        return this.request(`/experiments/participants/${participantId}`);
    }

    async updateParticipant(participantId, participantData) {
        return this.request(`/experiments/participants/${participantId}`, 'PUT', participantData);
    }

    async deleteParticipant(participantId) {
        return this.request(`/experiments/participants/${participantId}`, 'DELETE');
    }

    // Geometry endpoints
    async getGeometries() {
        return this.request('/experiments/geometries');
    }

    async createGeometry(geometryData) {
        return this.request('/experiments/geometries', 'POST', geometryData);
    }

    async getGeometry(geometryId) {
        return this.request(`/experiments/geometries/${geometryId}`);
    }

    // Trial endpoints
    async getTrials(participantId = null) {
        const endpoint = participantId ? `/experiments/trials?participant_id=${participantId}` : '/experiments/trials';
        return this.request(endpoint);
    }

    async createTrial(trialData) {
        return this.request('/experiments/trials', 'POST', trialData);
    }

    async getTrial(trialId) {
        return this.request(`/experiments/trials/${trialId}`);
    }

    async updateTrial(trialId, trialData) {
        return this.request(`/experiments/trials/${trialId}`, 'PUT', trialData);
    }

    async deleteTrial(trialId) {
        return this.request(`/experiments/trials/${trialId}`, 'DELETE');
    }

    async createTrialForParticipant(participantId, geometryData) {
        return this.request(`/experiments/participants/${participantId}/trials`, 'POST', geometryData);
    }

    // Data processing endpoints
    async uploadFile(file, trialId = null, participantId = null) {
        const formData = new FormData();
        formData.append('file', file);
        if (trialId) formData.append('trial_id', trialId);
        if (participantId) formData.append('participant_id', participantId);

        const response = await fetch(`${this.baseURL}/data/upload`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Upload failed: ${response.status}`);
        }

        return response.json();
    }

    async processFile(fileId, algorithm = 'algo6_javascript', useForceGradient = false) {
        const formData = new FormData();
        formData.append('algorithm', algorithm);
        formData.append('use_force_gradient', useForceGradient);

        const response = await fetch(`${this.baseURL}/data/process/${fileId}`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Processing failed: ${response.status}`);
        }

        return response.json();
    }

    async calculateLoss(trialId, objective, surveyResponses = null) {
        const formData = new FormData();
        formData.append('objective', objective);
        if (surveyResponses) {
            formData.append('survey_responses', JSON.stringify(surveyResponses));
        }

        const response = await fetch(`${this.baseURL}/data/calculate-loss/${trialId}`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Loss calculation failed: ${response.status}`);
        }

        return response.json();
    }

    // Optimization endpoints
    async suggestGeometry(participantId, objective, acquisitionFunction = 'EI', maxIterations = 10) {
        const formData = new FormData();
        formData.append('objective', objective);
        formData.append('acquisition_function', acquisitionFunction);
        formData.append('max_iterations', maxIterations);

        const response = await fetch(`${this.baseURL}/optimization/suggest-geometry/${participantId}`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Optimization failed: ${response.status}`);
        }

        return response.json();
    }

    async getOptimizationHistory(participantId, objective = null) {
        const endpoint = objective ?
            `/optimization/history/${participantId}?objective=${objective}` :
            `/optimization/history/${participantId}`;
        return this.request(endpoint);
    }

    async getOptimizationRecommendations(participantId, objective) {
        return this.request(`/optimization/recommendations/${participantId}?objective=${objective}`);
    }

    async compareGeometries(participantId, geometries) {
        return this.request(`/optimization/compare-geometries/${participantId}`, 'POST', { geometries });
    }

    // Utility endpoints
    async getAvailableAlgorithms() {
        return this.request('/data/algorithms');
    }

    async getAcquisitionFunctions() {
        return this.request('/optimization/acquisition-functions');
    }

    async getObjectives() {
        return this.request('/optimization/objectives');
    }

    async getHealthStatus() {
        return this.request('/health');
    }
}

// Create global API instance
window.api = new CrutchAPI();
