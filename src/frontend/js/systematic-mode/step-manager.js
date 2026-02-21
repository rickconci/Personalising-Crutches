/**
 * Step Manager - Handles step manipulation and variance calculations
 */

export class StepManager {
    constructor(uiComponents) {
        this.ui = uiComponents;
        this.steps = [];
        this.plotData = null;
    }

    /**
     * Get current steps
     */
    getSteps() {
        return [...this.steps];
    }

    /**
     * Set steps
     * @param {Array<number>} steps - Array of step times in seconds
     */
    setSteps(steps) {
        this.steps = [...steps];
    }

    /**
     * Get current plot data
     */
    getPlotData() {
        return this.plotData;
    }

    /**
     * Set plot data
     * @param {Object} plotData - Plot data object
     */
    setPlotData(plotData) {
        this.plotData = plotData;
    }

    /**
     * Add a step at a specific time
     * @param {number} stepTime - Time in seconds
     * @returns {boolean} - True if step was added, false if duplicate
     */
    addStep(stepTime) {
        // Check if a step already exists near this time (within 0.1 seconds)
        const existingStep = this.steps.find(step => Math.abs(step - stepTime) < 0.1);
        if (existingStep) {
            return false;
        }

        this.steps.push(stepTime);
        this.steps.sort((a, b) => a - b);
        return true;
    }

    /**
     * Remove a step by index
     * @param {number} stepIndex - Index of step to remove
     * @returns {boolean} - True if step was removed
     */
    removeStep(stepIndex) {
        if (stepIndex >= 0 && stepIndex < this.steps.length) {
            this.steps.splice(stepIndex, 1);
            return true;
        }
        return false;
    }

    /**
     * Calculate step variance (instability metric)
     * @param {Array<number>} steps - Optional step times, uses current steps if not provided
     * @returns {number|null} - Variance value or null if insufficient data
     */
    calculateVariance(steps = null) {
        const stepTimes = steps || this.steps;

        if (!stepTimes || stepTimes.length < 2) {
            return null;
        }

        // Calculate step intervals (time between consecutive steps)
        const stepIntervals = [];
        for (let i = 1; i < stepTimes.length; i++) {
            stepIntervals.push(stepTimes[i] - stepTimes[i - 1]);
        }

        // Calculate variance of step intervals
        const mean = stepIntervals.reduce((sum, val) => sum + val, 0) / stepIntervals.length;
        const squaredDiffs = stepIntervals.map(val => Math.pow(val - mean, 2));
        const variance = squaredDiffs.reduce((sum, val) => sum + val, 0) / stepIntervals.length;

        return variance;
    }

    /**
     * Get step count
     */
    getStepCount() {
        return this.steps.length;
    }

    /**
     * Clear all steps
     */
    clear() {
        this.steps = [];
        this.plotData = null;
    }

    /**
     * Handle plot click to add a step
     * @param {Object} data - Plotly click data
     * @param {Object} plots - Current plot data
     * @returns {Object} - Result of the operation
     */
    handlePlotClick(data, plots) {
        const point = data.points[0];

        // Only allow clicking on the force trace (trace 0)
        if (point.curveNumber !== 0) {
            return { success: false, reason: 'wrong_trace' };
        }

        const clickedTime = point.x;
        const added = this.addStep(clickedTime);

        if (!added) {
            return {
                success: false,
                reason: 'duplicate',
                message: 'A step already exists near this time'
            };
        }

        // Update plot data with new steps
        if (plots) {
            this.plotData = {
                ...plots,
                step_times: this.steps
            };
        }

        return {
            success: true,
            stepTime: clickedTime,
            steps: this.getSteps(),
            variance: this.calculateVariance()
        };
    }

    /**
     * Update steps from processing results
     * @param {Object} results - Processing results with step detection
     */
    updateFromResults(results) {
        if (results && results.step_detection && results.step_detection.step_times) {
            this.setSteps(results.step_detection.step_times);
        }

        if (results && results.plots) {
            this.setPlotData(results.plots);
        }
    }
}

