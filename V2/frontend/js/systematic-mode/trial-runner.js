/**
 * Trial Runner - Manages trial lifecycle, timers, and data collection
 */

export class TrialRunner {
    constructor(deviceManager, uiComponents) {
        this.device = deviceManager;
        this.ui = uiComponents;

        this.state = {
            timer: null,
            startTime: null,
            elapsed: 0,
            running: false,
            testMode: false,
            rawData: null
        };
    }

    /**
     * Check if a trial is currently running
     */
    isRunning() {
        return this.state.running;
    }

    /**
     * Get the current elapsed time
     */
    getElapsedTime() {
        return this.state.elapsed;
    }

    /**
     * Get raw data from the current/last trial
     */
    getRawData() {
        return this.state.rawData;
    }

    /**
     * Start a new trial
     * @param {Object} geometry - Geometry configuration
     * @param {boolean} testMode - Whether to use test mode (fake data)
     */
    async startTrial(geometry, testMode = false) {
        if (this.state.running) {
            throw new Error('A trial is already running');
        }

        if (!testMode && !this.device.getConnectionStatus()) {
            throw new Error('Device not connected. Please connect device or enable Test Mode');
        }

        try {
            this.state.running = true;
            this.state.startTime = Date.now();
            this.state.testMode = testMode;
            this.state.rawData = null;

            if (!testMode) {
                await this.device.startDataCollection();
            }

            this._startTimer();

            return {
                success: true,
                testMode: testMode,
                startTime: this.state.startTime
            };

        } catch (error) {
            this._resetState();
            throw error;
        }
    }

    /**
     * Stop the current trial
     */
    async stopTrial() {
        if (!this.state.running) {
            throw new Error('No trial is running');
        }

        try {
            let rawData;

            if (this.state.testMode) {
                rawData = await this._loadRealTestData();
            } else {
                await this.device.stopDataCollection();
                rawData = this.device.getDataBuffer();
            }

            this._stopTimer();
            this.state.running = false;
            this.state.rawData = rawData;

            return {
                success: true,
                rawData: rawData,
                duration: this.state.elapsed
            };

        } catch (error) {
            this._resetState();
            throw error;
        }
    }

    /**
     * Reset the trial state
     */
    reset() {
        this._resetState();
        this.device.clearDataBuffer();
    }

    /**
     * Start the timer
     * @private
     */
    _startTimer() {
        this.state.timer = setInterval(() => {
            this.state.elapsed = Date.now() - this.state.startTime;
            if (this.onTimerTick) {
                this.onTimerTick(this.state.elapsed);
            }
        }, 100);
    }

    /**
     * Stop the timer
     * @private
     */
    _stopTimer() {
        if (this.state.timer) {
            clearInterval(this.state.timer);
            this.state.timer = null;
        }
    }

    /**
     * Reset trial state
     * @private
     */
    _resetState() {
        this.state.running = false;
        this.state.startTime = null;
        this.state.elapsed = 0;
        this.state.testMode = false;
        this._stopTimer();
    }

    /**
     * Load real test data from CSV file via API
     * @private
     */
    async _loadRealTestData() {
        const testDataEndpoint = '/api/data/test-data';

        try {
            console.log('Loading real test data from API:', testDataEndpoint);

            const response = await fetch(testDataEndpoint);
            if (!response.ok) {
                throw new Error(`Failed to fetch test data: ${response.statusText}`);
            }

            const csvText = await response.text();
            const lines = csvText.split('\n');

            // Skip header line
            const data = [];
            for (let i = 1; i < lines.length; i++) {
                const line = lines[i].trim();
                if (!line) continue;

                const parts = line.split(',');
                if (parts.length >= 4) {
                    const force = parseFloat(parts[0]);
                    const accX = parseFloat(parts[1]);
                    const accY = parseFloat(parts[2]);
                    const time_ms = parseFloat(parts[3]);

                    // Convert to the format expected by the data processor
                    // CSV has: force, accX, accY, relative_time_ms
                    // Expected format: acc_x_time, acc_x_data, acc_z_data, force
                    data.push({
                        acc_x_time: time_ms,
                        acc_x_data: accX,
                        acc_z_data: accY,  // Using accY as vertical (Z) acceleration
                        force: force
                    });
                }
            }

            console.log(`Loaded ${data.length} real data points from CSV (duration: ${(data[data.length - 1].acc_x_time / 1000).toFixed(1)}s)`);
            return data;

        } catch (error) {
            console.error('Error loading real test data:', error);
            console.warn('Falling back to fake data generation');
            return this._generateFakeData();
        }
    }

    /**
     * Generate fake data for testing (fallback)
     * @private
     */
    _generateFakeData() {
        const duration = (Date.now() - this.state.startTime) / 1000; // seconds
        const sampleRate = 100; // Hz
        const numSamples = Math.floor(duration * sampleRate);

        const data = [];
        const stepInterval = 0.6; // seconds between steps
        const baseForce = 50; // Baseline force when crutch is in contact with ground

        for (let i = 0; i < numSamples; i++) {
            const time_ms = i * (1000 / sampleRate);
            const time_sec = i / sampleRate;

            // Generate step pattern for vertical acceleration (acc_z)
            const stepPhase = (time_sec % stepInterval) / stepInterval;
            let acc_z = 9.81; // Base gravity
            let force = baseForce; // Load cell force

            if (stepPhase < 0.3) {
                // Step impact - increased vertical acceleration and force
                const impactFactor = Math.sin(stepPhase * Math.PI / 0.3);
                acc_z += 5 * impactFactor;
                force += 100 * impactFactor; // Force spike during step
            } else if (stepPhase < 0.5) {
                // Unloading phase
                force = baseForce * (1 - 0.5 * Math.sin((stepPhase - 0.3) * Math.PI / 0.2));
            }

            // Generate forward acceleration (acc_x) - slight variation with steps
            let acc_x = 0.2 * Math.sin(2 * Math.PI * time_sec / stepInterval);

            // Add realistic noise
            acc_z += (Math.random() - 0.5) * 0.5;
            acc_x += (Math.random() - 0.5) * 0.3;
            force += (Math.random() - 0.5) * 5;

            data.push({
                acc_x_time: time_ms,
                acc_x_data: acc_x,
                acc_z_data: acc_z,
                force: Math.max(0, force) // Force can't be negative
            });
        }

        console.log(`Generated ${data.length} fake data points over ${duration.toFixed(1)}s`);
        return data;
    }

    /**
     * Set a callback for timer updates
     * @param {Function} callback - Called with elapsed time in ms
     */
    setTimerCallback(callback) {
        this.onTimerTick = callback;
    }
}

