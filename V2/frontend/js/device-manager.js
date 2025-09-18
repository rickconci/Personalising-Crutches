/**
 * Device Manager - Handles Bluetooth/BLE device communication
 * Responsible for connecting to and managing data from crutch sensors
 */

class DeviceManager {
    constructor() {
        this.CHARACTERISTIC_UUID = '0000ffe1-0000-1000-8000-00805f9b34fb';
        this.SERVICE_UUID = '0000ffe0-0000-1000-8000-00805f9b34fb';
        this.DEVICE_NAME_PREFIX = 'HIP_EXO';

        this.bleServer = null;
        this.bleCharacteristic = null;
        this.isConnected = false;
        this.dataBuffer = [];

        // Event callbacks
        this.onDataReceived = null;
        this.onConnectionChange = null;
        this.onError = null;
    }

    /**
     * Check if Web Bluetooth is supported
     * @returns {boolean} True if Web Bluetooth is available
     */
    isBluetoothSupported() {
        return 'bluetooth' in navigator;
    }

    /**
     * Connect to the crutch device
     * @returns {Promise<boolean>} True if connection successful
     */
    async connect() {
        if (!this.isBluetoothSupported()) {
            this._handleError('Web Bluetooth is not available on this browser.');
            return false;
        }

        try {
            this._notifyConnectionChange('searching', 'Searching for device...');

            // Request device
            const device = await navigator.bluetooth.requestDevice({
                filters: [{ namePrefix: this.DEVICE_NAME_PREFIX }],
                optionalServices: [this.SERVICE_UUID]
            });

            this._notifyConnectionChange('connecting', 'Connecting to device...');

            // Set up disconnect handler
            device.addEventListener('gattserverdisconnected', () => {
                this._handleDisconnection();
            });

            // Connect to GATT server
            this.bleServer = await device.gatt.connect();

            // Get service and characteristic
            const service = await this.bleServer.getPrimaryService(this.SERVICE_UUID);
            this.bleCharacteristic = await service.getCharacteristic(this.CHARACTERISTIC_UUID);

            this.isConnected = true;
            this._notifyConnectionChange('connected', 'Device connected successfully');

            return true;

        } catch (error) {
            this._handleError(`Bluetooth connection failed: ${error.message}`);
            return false;
        }
    }

    /**
     * Disconnect from the device
     */
    async disconnect() {
        if (this.bleServer && this.bleServer.connected) {
            await this.stopDataCollection();
            this.bleServer.disconnect();
        }
        this._handleDisconnection();
    }

    /**
     * Start collecting data from the device
     * @returns {Promise<boolean>} True if data collection started successfully
     */
    async startDataCollection() {
        if (!this.isConnected || !this.bleCharacteristic) {
            this._handleError('Device not connected');
            return false;
        }

        try {
            this.dataBuffer = []; // Clear previous data
            await this.bleCharacteristic.startNotifications();
            this.bleCharacteristic.addEventListener('characteristicvaluechanged',
                this._handleCharacteristicValueChanged.bind(this));
            return true;
        } catch (error) {
            this._handleError(`Failed to start data collection: ${error.message}`);
            return false;
        }
    }

    /**
     * Stop collecting data from the device
     */
    async stopDataCollection() {
        if (this.bleCharacteristic) {
            try {
                await this.bleCharacteristic.stopNotifications();
                this.bleCharacteristic.removeEventListener('characteristicvaluechanged',
                    this._handleCharacteristicValueChanged.bind(this));
            } catch (error) {
                console.warn('Error stopping notifications:', error);
            }
        }
    }

    /**
     * Get the current data buffer
     * @returns {Array} Array of data points collected from the device
     */
    getDataBuffer() {
        return [...this.dataBuffer];
    }

    /**
     * Clear the data buffer
     */
    clearDataBuffer() {
        this.dataBuffer = [];
    }

    /**
     * Get connection status
     * @returns {boolean} True if device is connected
     */
    getConnectionStatus() {
        return this.isConnected;
    }

    /**
     * Set callback for data received events
     * @param {Function} callback - Function to call when data is received
     */
    setDataReceivedCallback(callback) {
        this.onDataReceived = callback;
    }

    /**
     * Set callback for connection status changes
     * @param {Function} callback - Function to call when connection status changes
     */
    setConnectionChangeCallback(callback) {
        this.onConnectionChange = callback;
    }

    /**
     * Set callback for error events
     * @param {Function} callback - Function to call when errors occur
     */
    setErrorCallback(callback) {
        this.onError = callback;
    }

    /**
     * Handle incoming characteristic value changes (data from device)
     * @private
     */
    _handleCharacteristicValueChanged(event) {
        try {
            const value = event.target.value;
            const decoder = new TextDecoder('utf-8');
            const dataString = decoder.decode(value);

            // Parse the data (assuming CSV format: timestamp,force)
            const lines = dataString.trim().split('\n');

            for (const line of lines) {
                if (line.trim()) {
                    const parts = line.split(',');
                    if (parts.length >= 2) {
                        const timestamp = parseFloat(parts[0]);
                        const force = parseFloat(parts[1]);

                        if (!isNaN(timestamp) && !isNaN(force)) {
                            const dataPoint = { timestamp, force };
                            this.dataBuffer.push(dataPoint);

                            // Notify listeners
                            if (this.onDataReceived) {
                                this.onDataReceived(dataPoint);
                            }
                        }
                    }
                }
            }
        } catch (error) {
            this._handleError(`Data parsing error: ${error.message}`);
        }
    }

    /**
     * Handle device disconnection
     * @private
     */
    _handleDisconnection() {
        this.isConnected = false;
        this.bleServer = null;
        this.bleCharacteristic = null;
        this._notifyConnectionChange('disconnected', 'Device disconnected');
    }

    /**
     * Notify connection status change
     * @private
     */
    _notifyConnectionChange(status, message) {
        if (this.onConnectionChange) {
            this.onConnectionChange(status, message);
        }
    }

    /**
     * Handle errors
     * @private
     */
    _handleError(message) {
        console.error('DeviceManager Error:', message);
        if (this.onError) {
            this.onError(message);
        }
    }

    /**
     * Create a downloadable file from the current data buffer
     * @param {string} filename - Name for the downloaded file
     * @returns {string} Data URL for download
     */
    exportDataAsCSV(filename = 'crutch_data.csv') {
        if (this.dataBuffer.length === 0) {
            throw new Error('No data to export');
        }

        // Create CSV content
        let csvContent = 'timestamp,force\n';
        this.dataBuffer.forEach(point => {
            csvContent += `${point.timestamp},${point.force}\n`;
        });

        // Create blob and URL
        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);

        // Trigger download
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        return url;
    }

    /**
     * Get data collection statistics
     * @returns {Object} Statistics about the collected data
     */
    getDataStatistics() {
        if (this.dataBuffer.length === 0) {
            return {
                count: 0,
                duration: 0,
                minForce: 0,
                maxForce: 0,
                avgForce: 0
            };
        }

        const forces = this.dataBuffer.map(point => point.force);
        const timestamps = this.dataBuffer.map(point => point.timestamp);

        return {
            count: this.dataBuffer.length,
            duration: Math.max(...timestamps) - Math.min(...timestamps),
            minForce: Math.min(...forces),
            maxForce: Math.max(...forces),
            avgForce: forces.reduce((sum, force) => sum + force, 0) / forces.length
        };
    }
}

// Export for use in other modules
window.DeviceManager = DeviceManager;
