/**
 * Geometry Sequencer - Manages dynamic geometry loading and sequencing
 */

export class GeometrySequencer {
    constructor(apiClient, uiComponents) {
        this.api = apiClient;
        this.ui = uiComponents;
        this.currentGeometry = null;
        this.currentSequence = null;
    }

    /**
     * Get current geometry
     */
    getCurrentGeometry() {
        return this.currentGeometry;
    }

    /**
     * Set current geometry
     * @param {Object} geometry - Geometry object
     */
    setCurrentGeometry(geometry) {
        this.currentGeometry = geometry;
    }

    /**
     * Get current sequence
     */
    getCurrentSequence() {
        return this.currentSequence;
    }

    /**
     * Set current sequence
     * @param {string} sequenceName - Name of the sequence
     */
    setCurrentSequence(sequenceName) {
        this.currentSequence = sequenceName;
    }

    /**
     * Load the next geometry for the current participant and sequence
     * @param {number} participantId - Participant ID
     * @returns {Object|null} - Next geometry or null if all complete
     */
    async loadNextGeometry(participantId) {
        if (!participantId || !this.currentSequence) {
            return null;
        }

        try {
            const nextGeometry = await this.api.getNextGeometry(
                participantId,
                this.currentSequence
            );

            this.currentGeometry = nextGeometry;
            return nextGeometry;

        } catch (error) {
            console.error('Error loading next geometry:', error);
            throw error;
        }
    }

    /**
     * Get progress for the current sequence
     * @param {number} participantId - Participant ID
     * @returns {Object} - Progress information
     */
    async getSequenceProgress(participantId) {
        if (!participantId || !this.currentSequence) {
            return {
                completed_trials: 0,
                total_trials: 0,
                progress_percentage: 0
            };
        }

        try {
            const progress = await this.api.getSequenceProgress(
                participantId,
                this.currentSequence
            );
            return progress;

        } catch (error) {
            console.error('Error getting sequence progress:', error);
            throw error;
        }
    }

    /**
     * Create trial from current geometry
     * @param {number} participantId - Participant ID
     * @returns {Object} - Trial creation result
     */
    async createTrialFromGeometry(participantId) {
        if (!this.currentGeometry) {
            throw new Error('No geometry selected');
        }

        try {
            const result = await this.api.createTrialFromGeometry(
                participantId,
                this.currentGeometry
            );
            return result;

        } catch (error) {
            console.error('Error creating trial from geometry:', error);
            throw error;
        }
    }

    /**
     * Render next geometry UI
     * @param {HTMLElement} infoElement - Element to render geometry info
     * @param {HTMLElement} startButton - Start trial button
     */
    renderNextGeometry(infoElement, startButton) {
        if (!this.currentGeometry) {
            return false;
        }

        if (!infoElement) {
            return false;
        }

        const geometry = this.currentGeometry;
        const isBaseline = geometry.is_baseline;
        const sessionClass = isBaseline ? 'border-warning' : 'border-primary';
        const headerClass = isBaseline ? 'bg-warning text-dark' : 'bg-primary text-white';

        infoElement.innerHTML = `
            <div class="card ${sessionClass}">
                <div class="card-header ${headerClass}">
                    <h6 class="mb-0">${geometry.session}</h6>
                </div>
                <div class="card-body">
                    <h5>${geometry.name}</h5>
                    <p class="mb-2">${geometry.description}</p>
                    <div class="row">
                        <div class="col-3">
                            <strong>α (Alpha):</strong><br>
                            <span class="badge bg-primary">${geometry.alpha}°</span>
                        </div>
                        <div class="col-3">
                            <strong>β (Beta):</strong><br>
                            <span class="badge bg-success">${geometry.beta}°</span>
                        </div>
                        <div class="col-3">
                            <strong>γ (Gamma):</strong><br>
                            <span class="badge bg-info">${geometry.gamma}°</span>
                        </div>
                        <div class="col-3">
                            <strong>Trial:</strong><br>
                            <span class="badge bg-secondary">${geometry.trial_number}/${geometry.total_trials}</span>
                        </div>
                    </div>
                </div>
            </div>
        `;

        if (startButton) {
            startButton.disabled = false;
        }

        return true;
    }

    /**
     * Render completion message
     * @param {HTMLElement} infoElement - Element to render message
     * @param {HTMLElement} startButton - Start trial button
     */
    renderCompletionMessage(infoElement, startButton) {
        if (!infoElement) return;

        infoElement.innerHTML = `
            <div class="alert alert-success">
                <h6>🎉 All trials completed!</h6>
                <p>This participant has completed all geometries in the ${this.currentSequence} sequence.</p>
            </div>
        `;

        if (startButton) {
            startButton.disabled = true;
        }
    }

    /**
     * Update progress UI
     * @param {number} participantId - Participant ID
     * @param {HTMLElement} progressBar - Progress bar element
     * @param {HTMLElement} progressText - Progress text element
     */
    async updateProgressUI(participantId, progressBar, progressText) {
        if (!participantId || !this.currentSequence) {
            return;
        }

        try {
            const progress = await this.getSequenceProgress(participantId);

            const percentage = Math.round(progress.progress_percentage);

            if (progressBar) {
                progressBar.style.width = `${percentage}%`;
                progressBar.setAttribute('aria-valuenow', percentage);
            }

            if (progressText) {
                progressText.textContent =
                    `${progress.completed_trials} of ${progress.total_trials} trials completed (${percentage}%)`;
            }

        } catch (error) {
            console.error('Error updating progress:', error);
        }
    }

    /**
     * Clear current state
     */
    clear() {
        this.currentGeometry = null;
        this.currentSequence = null;
    }
}

