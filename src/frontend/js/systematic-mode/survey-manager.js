/**
 * Survey Manager - Handles survey collection and trial saving
 */

export class SurveyManager {
    constructor(apiClient, uiComponents) {
        this.api = apiClient;
        this.ui = uiComponents;
    }

    /**
     * Collect survey responses from the DOM
     * @param {string} [containerId='survey-sections'] - id of the parent element
     *     containing the SUS / NRS / TLX selects. Lets the same SurveyManager
     *     drive both the systematic-mode survey and the BO-mode survey.
     * @returns {Object} - Survey responses with calculated scores
     */
    collectSurveyResponses(containerId = 'survey-sections') {
        const surveyResponses = {};

        const surveySection = document.getElementById(containerId);
        if (!surveySection) {
            console.error(`Survey section '${containerId}' not found`);
            return surveyResponses;
        }

        // SUS Score Calculation (6 questions)
        const susQuestions = surveySection.querySelectorAll('.sus-question');
        let susScore = 0;
        susQuestions.forEach((q, index) => {
            const value = parseInt(q.value);
            const isPositive = q.dataset.positive === 'true';
            surveyResponses[`sus_q${index + 1}`] = value;
            if (isPositive) {
                susScore += (value - 1);
            } else {
                susScore += (5 - value);
            }
        });
        surveyResponses['sus_score'] = (susScore / 24) * 100;

        console.log(`[${containerId}] SUS: ${susQuestions.length} qs, raw=${susScore}, norm=${surveyResponses['sus_score']}`);

        // NRS Score (scoped to container; matches `.nrs-score`)
        const nrsScoreElement = surveySection.querySelector('.nrs-score');
        if (nrsScoreElement) {
            surveyResponses['nrs_score'] = parseInt(nrsScoreElement.value);
        }

        // TLX Score Calculation (5 questions, 0-20 each)
        const tlxQuestions = surveySection.querySelectorAll('.tlx-question');
        const tlxFieldNames = ['tlx_mental_demand', 'tlx_physical_demand', 'tlx_performance', 'tlx_effort', 'tlx_frustration'];
        let tlxScore = 0;
        tlxQuestions.forEach((q, index) => {
            const value = parseInt(q.value);
            if (index < tlxFieldNames.length) {
                surveyResponses[tlxFieldNames[index]] = value;
            }
            if (tlxFieldNames[index] === 'tlx_performance') {
                tlxScore += (20 - value);
            } else {
                tlxScore += value;
            }
        });
        surveyResponses['tlx_score'] = tlxScore / 5;

        console.log(`[${containerId}] TLX: mean=${surveyResponses['tlx_score']}`);

        return surveyResponses;
    }

    /**
     * Get metabolic cost from input
     * @returns {number|null} - Metabolic cost value or null
     */
    getMetabolicCost() {
        const metabolicCostInput = document.getElementById('metabolic-cost-input');
        return metabolicCostInput?.value ? parseFloat(metabolicCostInput.value) : null;
    }

    /**
     * Save trial with all collected data
     * @param {Object} trialData - Trial data to save
     * @returns {Object} - Saved trial response
     */
    async saveTrial(trialData) {
        const {
            trialId,  // If provided, update existing trial instead of creating new one
            participantId,
            geometry,
            steps,
            instabilityLoss,
            surveyResponses,
            metabolicCost,
            lapsCompleted,
            openCapEvents  // OpenCap toggle events
        } = trialData;

        // Validate required fields
        if (!participantId) {
            throw new Error('No participant selected');
        }

        if (!geometry) {
            throw new Error('No geometry selected');
        }

        if (!steps || steps.length === 0) {
            throw new Error('No step data available. Please run a trial first.');
        }

        console.log('Saving trial - trialId:', trialId, 'mode:', trialId ? 'UPDATE' : 'CREATE');

        try {
            if (trialId) {
                // UPDATE existing trial
                const updatePayload = {
                    survey_responses: surveyResponses,
                    processed_features: {
                        step_count: steps.length,
                        instability_loss: instabilityLoss,
                        step_variance: instabilityLoss,
                        laps_completed: lapsCompleted
                    },
                    steps: steps,
                    metabolic_cost: metabolicCost,
                    laps_completed: lapsCompleted,
                    opencap_events: openCapEvents || []
                };

                console.log('Updating trial', trialId, 'with payload:', updatePayload);
                const response = await this.api.updateTrial(trialId, updatePayload);
                console.log('Trial updated successfully:', response);

                // Calculate total combined loss using backend
                try {
                    const lossResult = await this.api.calculateLoss(trialId, 'stability', surveyResponses);
                    console.log('Loss calculated:', lossResult);
                } catch (error) {
                    console.warn('Failed to calculate loss:', error);
                }

                return response;

            } else {
                // CREATE new trial
                const createPayload = {
                    participant_id: participantId,
                    geometry_id: geometry.id,
                    alpha: geometry.alpha,
                    beta: geometry.beta,
                    gamma: geometry.gamma,
                    delta: geometry.delta || 0,
                    source: 'grid_search',
                    survey_responses: surveyResponses,
                    processed_features: {
                        step_count: steps.length,
                        instability_loss: instabilityLoss,
                        step_variance: instabilityLoss,
                        laps_completed: lapsCompleted
                    },
                    steps: steps,
                    metabolic_cost: metabolicCost,
                    laps_completed: lapsCompleted,
                    opencap_events: openCapEvents || []
                };

                console.log('Creating new trial with payload:', createPayload);
                const response = await this.api.createTrial(createPayload);
                console.log('Trial created successfully:', response);

                // Calculate total combined loss using backend
                try {
                    const lossResult = await this.api.calculateLoss(response.id, 'stability', surveyResponses);
                    console.log('Loss calculated:', lossResult);
                } catch (error) {
                    console.warn('Failed to calculate loss:', error);
                }

                return response;
            }

        } catch (error) {
            console.error('Error saving trial:', error);
            throw error;
        }
    }

    /**
     * Reset survey forms (scoped to a container) to default values.
     * @param {string} [containerId='survey-sections']
     */
    resetSurveyForms(containerId = 'survey-sections') {
        const container = document.getElementById(containerId) || document;

        container.querySelectorAll('.sus-question').forEach(q => q.value = '3');
        const nrsScore = container.querySelector('.nrs-score');
        if (nrsScore) nrsScore.value = '5';
        container.querySelectorAll('.tlx-question').forEach(q => q.value = '10');

        // Metabolic cost only exists on the systematic survey
        if (container === document || container.id === 'survey-sections') {
            const metabolicCostInput = document.getElementById('metabolic-cost-input');
            if (metabolicCostInput) metabolicCostInput.value = '';
        }
    }

    /**
     * Validate survey responses are present (scoped to a container).
     * @param {string} [containerId='survey-sections']
     * @returns {Object} - Validation result {valid: boolean, errors: Array}
     */
    validateSurveyResponses(containerId = 'survey-sections') {
        const container = document.getElementById(containerId) || document;
        const errors = [];

        if (container.querySelectorAll('.sus-question').length === 0) {
            errors.push('SUS questions not found');
        }
        const nrsScore = container.querySelector('.nrs-score');
        if (!nrsScore || !nrsScore.value) {
            errors.push('NRS score is required');
        }
        if (container.querySelectorAll('.tlx-question').length === 0) {
            errors.push('TLX questions not found');
        }

        return { valid: errors.length === 0, errors };
    }

    /**
     * Calculate SUS score preview
     * @returns {number} - Calculated SUS score (0-100)
     */
    calculateSUSScorePreview() {
        const susQuestions = document.querySelectorAll('.sus-question');
        let susScore = 0;

        susQuestions.forEach((q) => {
            const value = parseInt(q.value) || 3;
            const isPositive = q.dataset.positive === 'true';
            if (isPositive) {
                susScore += (value - 1);
            } else {
                susScore += (5 - value);
            }
        });

        return (susScore / 24) * 100;
    }

    /**
     * Calculate TLX score preview
     * @returns {number} - Calculated TLX score (0-20, mean of dimensions)
     */
    calculateTLXScorePreview() {
        const tlxQuestions = document.querySelectorAll('.tlx-question');
        const tlxFieldNames = ['tlx_mental_demand', 'tlx_physical_demand', 'tlx_performance', 'tlx_effort', 'tlx_frustration'];
        let tlxScore = 0;

        tlxQuestions.forEach((q, index) => {
            const value = parseInt(q.value) || 10;

            // Flip performance so that high performance = low workload
            if (index < tlxFieldNames.length && tlxFieldNames[index] === 'tlx_performance') {
                tlxScore += (20 - value);
            } else {
                tlxScore += value;
            }
        });

        return tlxScore / 5; // Mean of all dimensions (0-20 scale)
    }
}

