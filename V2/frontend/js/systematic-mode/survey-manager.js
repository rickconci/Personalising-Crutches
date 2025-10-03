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
     * @returns {Object} - Survey responses with calculated scores
     */
    collectSurveyResponses() {
        const surveyResponses = {};

        // Scope queries to the systematic mode survey section only
        const surveySection = document.getElementById('survey-sections');
        if (!surveySection) {
            console.error('Survey section not found');
            return surveyResponses;
        }

        // SUS Score Calculation (6 questions in systematic mode)
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
        surveyResponses['sus_score'] = (susScore / 24) * 100; // Normalize to 0-100 (6 questions × 4 = 24)

        console.log(`SUS: ${susQuestions.length} questions, raw score: ${susScore}, normalized: ${surveyResponses['sus_score']}`);

        // NRS Score
        const nrsScoreElement = document.getElementById('nrs-score');
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
            tlxScore += value;
        });
        surveyResponses['tlx_score'] = tlxScore; // Already on 0-100 scale

        console.log(`TLX: ${tlxQuestions.length} questions, total score: ${tlxScore}`);
        console.log('TLX breakdown:', {
            mental_demand: surveyResponses['tlx_mental_demand'],
            physical_demand: surveyResponses['tlx_physical_demand'],
            performance: surveyResponses['tlx_performance'],
            effort: surveyResponses['tlx_effort'],
            frustration: surveyResponses['tlx_frustration']
        });

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
            lapsCompleted
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
                    laps_completed: lapsCompleted
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
                    laps_completed: lapsCompleted
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
     * Reset survey forms to default values
     */
    resetSurveyForms() {
        // Reset SUS questions
        const susQuestions = document.querySelectorAll('.sus-question');
        susQuestions.forEach(q => q.value = '3');

        // Reset NRS score
        const nrsScore = document.getElementById('nrs-score');
        if (nrsScore) nrsScore.value = '5';

        // Reset TLX questions
        const tlxQuestions = document.querySelectorAll('.tlx-question');
        tlxQuestions.forEach(q => q.value = '10');

        // Reset metabolic cost
        const metabolicCostInput = document.getElementById('metabolic-cost-input');
        if (metabolicCostInput) metabolicCostInput.value = '';
    }

    /**
     * Validate survey responses
     * @returns {Object} - Validation result {valid: boolean, errors: Array}
     */
    validateSurveyResponses() {
        const errors = [];

        // Check SUS questions
        const susQuestions = document.querySelectorAll('.sus-question');
        if (susQuestions.length === 0) {
            errors.push('SUS questions not found');
        }

        // Check NRS score
        const nrsScore = document.getElementById('nrs-score');
        if (!nrsScore || !nrsScore.value) {
            errors.push('NRS score is required');
        }

        // Check TLX questions
        const tlxQuestions = document.querySelectorAll('.tlx-question');
        if (tlxQuestions.length === 0) {
            errors.push('TLX questions not found');
        }

        return {
            valid: errors.length === 0,
            errors: errors
        };
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
     * @returns {number} - Calculated TLX score (0-100)
     */
    calculateTLXScorePreview() {
        const tlxQuestions = document.querySelectorAll('.tlx-question');
        let tlxScore = 0;

        tlxQuestions.forEach((q) => {
            const value = parseInt(q.value) || 10;
            tlxScore += value;
        });

        return tlxScore;
    }
}

