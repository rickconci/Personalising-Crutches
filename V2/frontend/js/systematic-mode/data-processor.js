/**
 * Data Processor - Handles file upload and data processing
 */

export class DataProcessor {
    constructor(apiClient, uiComponents) {
        this.api = apiClient;
        this.ui = uiComponents;
    }

    /**
     * Upload and process a data file
     * @param {File} file - Data file to upload
     * @param {Object} trialContext - Context for the trial (participant, geometry)
     * @returns {Object} - Processing results
     */
    async uploadAndProcessFile(file, trialContext) {
        if (!trialContext.participantId) {
            throw new Error('No participant selected');
        }

        try {
            // Create trial record first
            const trialData = {
                participant_id: trialContext.participantId,
                geometry_id: trialContext.geometry?.id,  // Important: include geometry_id
                alpha: trialContext.geometry?.alpha || 95,
                beta: trialContext.geometry?.beta || 125,
                gamma: trialContext.geometry?.gamma || 0,
                delta: trialContext.geometry?.delta || 0,
                source: 'grid_search',
                survey_responses: {
                    effort_survey_answer: 3,
                    pain_survey_answer: 3,
                    stability_survey_answer: 3
                }
            };

            const newTrial = await this.api.createTrial(trialData);

            // Upload and process file
            const uploadResult = await this.api.uploadFile(file, newTrial.id);
            const processResult = await this.api.processFile(uploadResult.id);

            // Update trial with results
            const updateData = {
                processed_features: processResult.processing_results.data_info,
                steps: processResult.processing_results.step_detection.step_times,
                step_variance: processResult.processing_results.gait_metrics.step_variance,
                y_change: processResult.processing_results.gait_metrics.y_change,
                y_total: processResult.processing_results.gait_metrics.y_total,
                total_combined_loss: processResult.processing_results.gait_metrics.step_variance || 0
            };

            const updatedTrial = await this.api.updateTrial(newTrial.id, updateData);

            return {
                success: true,
                trial: updatedTrial || newTrial,  // Return updated trial, fallback to original
                results: processResult.processing_results
            };

        } catch (error) {
            console.error('Upload and process error:', error);
            throw error;
        }
    }

    /**
     * Process raw data from trial
     * @param {Array} rawData - Raw data points
     * @returns {File} - CSV file object
     */
    convertRawDataToFile(rawData) {
        if (!rawData || rawData.length === 0) {
            throw new Error('No data to process');
        }

        // Convert raw data to CSV format with accelerometer and force columns
        const csvContent = 'acc_x_time,acc_x_data,acc_z_data,force\n' +
            rawData.map(point =>
                `${point.acc_x_time},${point.acc_x_data},${point.acc_z_data},${point.force}`
            ).join('\n');

        const blob = new Blob([csvContent], { type: 'text/csv' });
        const file = new File([blob], 'trial_data.csv', { type: 'text/csv' });

        console.log('Created CSV file:', file.name, file.size, 'bytes');
        return file;
    }

    /**
     * Process trial data and upload
     * @param {Array} rawData - Raw data points
     * @param {Object} trialContext - Context for the trial
     * @returns {Object} - Processing results
     */
    async processTrialData(rawData, trialContext) {
        const file = this.convertRawDataToFile(rawData);
        return await this.uploadAndProcessFile(file, trialContext);
    }
}

