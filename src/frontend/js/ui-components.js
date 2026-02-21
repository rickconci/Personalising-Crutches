/**
 * UI Components - Reusable UI functionality and utilities
 * Handles notifications, modals, plots, and common UI interactions
 */

class UIComponents {
    constructor() {
        this.toastContainer = this._createToastContainer();
    }

    /**
     * Show a notification toast
     * @param {string} message - Message to display
     * @param {string} type - Bootstrap alert type (success, danger, warning, info)
     * @param {number} duration - Duration in milliseconds (0 for persistent)
     */
    showNotification(message, type = 'success', duration = 5000) {
        const toastEl = document.createElement('div');
        toastEl.className = `toast align-items-center text-white bg-${type} border-0`;
        toastEl.setAttribute('role', 'alert');
        toastEl.setAttribute('aria-live', 'assertive');
        toastEl.setAttribute('aria-atomic', 'true');

        toastEl.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">${message}</div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" 
                        data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
        `;

        this.toastContainer.appendChild(toastEl);

        const toast = new bootstrap.Toast(toastEl, {
            autohide: duration > 0,
            delay: duration
        });

        toast.show();

        // Clean up after toast is hidden
        toastEl.addEventListener('hidden.bs.toast', () => {
            toastEl.remove();
        });
    }

    /**
     * Show a confirmation dialog
     * @param {string} message - Confirmation message
     * @param {string} title - Dialog title
     * @returns {Promise<boolean>} True if confirmed, false if cancelled
     */
    async showConfirmDialog(message, title = 'Confirm Action') {
        return new Promise((resolve) => {
            // Determine if this is a delete action to style accordingly
            const isDeleteAction = title.toLowerCase().includes('delete');
            const confirmBtnClass = isDeleteAction ? 'btn-danger' : 'btn-primary';
            const confirmBtnText = isDeleteAction ? 'Delete' : 'Confirm';

            const modalHtml = `
                <div class="modal fade" id="confirmModal" tabindex="-1">
                    <div class="modal-dialog">
                        <div class="modal-content">
                            <div class="modal-header ${isDeleteAction ? 'bg-danger text-white' : ''}">
                                <h5 class="modal-title">${title}</h5>
                                <button type="button" class="btn-close ${isDeleteAction ? 'btn-close-white' : ''}" data-bs-dismiss="modal"></button>
                            </div>
                            <div class="modal-body">
                                <p>${message}</p>
                            </div>
                            <div class="modal-footer">
                                <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                                <button type="button" class="btn ${confirmBtnClass}" id="confirmBtn">${confirmBtnText}</button>
                            </div>
                        </div>
                    </div>
                </div>
            `;

            // Remove existing modal if present
            const existingModal = document.getElementById('confirmModal');
            if (existingModal) {
                existingModal.remove();
            }

            // Add modal to DOM
            document.body.insertAdjacentHTML('beforeend', modalHtml);
            const modalElement = document.getElementById('confirmModal');
            const modal = new bootstrap.Modal(modalElement);

            // Handle confirm button
            document.getElementById('confirmBtn').addEventListener('click', () => {
                modal.hide();
                resolve(true);
            });

            // Handle modal close (cancel)
            modalElement.addEventListener('hidden.bs.modal', () => {
                modalElement.remove();
                resolve(false);
            });

            modal.show();
        });
    }

    /**
     * Create a loading spinner element
     * @param {string} message - Loading message
     * @returns {HTMLElement} Loading spinner element
     */
    createLoadingSpinner(message = 'Loading...') {
        const spinner = document.createElement('div');
        spinner.className = 'text-center p-4';
        spinner.innerHTML = `
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <div class="mt-2">${message}</div>
        `;
        return spinner;
    }

    /**
     * Show/hide loading state on an element
     * @param {HTMLElement} element - Element to show loading on
     * @param {boolean} show - True to show loading, false to hide
     * @param {string} message - Loading message
     */
    toggleLoading(element, show, message = 'Loading...') {
        if (show) {
            element.dataset.originalContent = element.innerHTML;
            element.innerHTML = this.createLoadingSpinner(message).outerHTML;
            element.style.pointerEvents = 'none';
        } else {
            if (element.dataset.originalContent) {
                element.innerHTML = element.dataset.originalContent;
                delete element.dataset.originalContent;
            }
            element.style.pointerEvents = '';
        }
    }

    /**
     * Format time duration for display
     * @param {number} milliseconds - Duration in milliseconds
     * @returns {string} Formatted time string (MM:SS.s)
     */
    formatDuration(milliseconds) {
        const minutes = Math.floor(milliseconds / 60000);
        const seconds = Math.floor((milliseconds % 60000) / 1000);
        const deciseconds = Math.floor((milliseconds % 1000) / 100);
        return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}.${deciseconds}`;
    }

    /**
     * Format number for display with specified decimal places
     * @param {number} value - Number to format
     * @param {number} decimals - Number of decimal places
     * @returns {string} Formatted number string
     */
    formatNumber(value, decimals = 2) {
        if (value === null || value === undefined || isNaN(value)) {
            return 'N/A';
        }
        return Number(value).toFixed(decimals);
    }

    /**
     * Create a simple data table
     * @param {Array} data - Array of objects to display
     * @param {Array} columns - Column definitions [{ key, label, formatter? }]
     * @returns {HTMLElement} Table element
     */
    createDataTable(data, columns) {
        const table = document.createElement('table');
        table.className = 'table table-striped table-hover';

        // Create header
        const thead = document.createElement('thead');
        thead.className = 'table-light';
        const headerRow = document.createElement('tr');

        columns.forEach(col => {
            const th = document.createElement('th');
            th.textContent = col.label;
            headerRow.appendChild(th);
        });

        thead.appendChild(headerRow);
        table.appendChild(thead);

        // Create body
        const tbody = document.createElement('tbody');

        data.forEach(row => {
            const tr = document.createElement('tr');

            columns.forEach(col => {
                const td = document.createElement('td');
                const value = row[col.key];

                if (col.formatter && typeof col.formatter === 'function') {
                    td.innerHTML = col.formatter(value, row);
                } else {
                    td.textContent = value ?? 'N/A';
                }

                tr.appendChild(td);
            });

            tbody.appendChild(tr);
        });

        table.appendChild(tbody);
        return table;
    }

    /**
     * Create a progress bar
     * @param {number} value - Progress value (0-100)
     * @param {string} label - Progress label
     * @param {string} variant - Bootstrap color variant
     * @returns {HTMLElement} Progress bar element
     */
    createProgressBar(value, label = '', variant = 'primary') {
        const container = document.createElement('div');
        container.className = 'mb-2';

        if (label) {
            const labelEl = document.createElement('div');
            labelEl.className = 'small text-muted mb-1';
            labelEl.textContent = label;
            container.appendChild(labelEl);
        }

        const progressContainer = document.createElement('div');
        progressContainer.className = 'progress';

        const progressBar = document.createElement('div');
        progressBar.className = `progress-bar bg-${variant}`;
        progressBar.style.width = `${Math.min(100, Math.max(0, value))}%`;
        progressBar.setAttribute('role', 'progressbar');
        progressBar.setAttribute('aria-valuenow', value);
        progressBar.setAttribute('aria-valuemin', '0');
        progressBar.setAttribute('aria-valuemax', '100');
        progressBar.textContent = `${value}%`;

        progressContainer.appendChild(progressBar);
        container.appendChild(progressContainer);

        return container;
    }

    /**
     * Create a badge element
     * @param {string} text - Badge text
     * @param {string} variant - Bootstrap color variant
     * @returns {HTMLElement} Badge element
     */
    createBadge(text, variant = 'secondary') {
        const badge = document.createElement('span');
        badge.className = `badge bg-${variant}`;
        badge.textContent = text;
        return badge;
    }

    /**
     * Create a card element
     * @param {string} title - Card title
     * @param {string} content - Card content (HTML)
     * @param {Array} actions - Array of action buttons [{ text, class, onclick }]
     * @returns {HTMLElement} Card element
     */
    createCard(title, content, actions = []) {
        const card = document.createElement('div');
        card.className = 'card';

        if (title) {
            const header = document.createElement('div');
            header.className = 'card-header';
            header.innerHTML = `<h5 class="card-title mb-0">${title}</h5>`;
            card.appendChild(header);
        }

        const body = document.createElement('div');
        body.className = 'card-body';
        body.innerHTML = content;
        card.appendChild(body);

        if (actions.length > 0) {
            const footer = document.createElement('div');
            footer.className = 'card-footer';

            actions.forEach(action => {
                const btn = document.createElement('button');
                btn.className = action.class || 'btn btn-primary';
                btn.textContent = action.text;
                if (action.onclick) {
                    btn.addEventListener('click', action.onclick);
                }
                footer.appendChild(btn);
                footer.appendChild(document.createTextNode(' '));
            });

            card.appendChild(footer);
        }

        return card;
    }

    /**
     * Toggle visibility of multiple elements
     * @param {Object} elements - Object with element keys and boolean values
     */
    toggleElements(elements) {
        Object.entries(elements).forEach(([element, show]) => {
            const el = typeof element === 'string' ? document.getElementById(element) : element;
            if (el) {
                if (show) {
                    el.classList.remove('d-none');
                } else {
                    el.classList.add('d-none');
                }
            }
        });
    }

    /**
     * Animate element appearance
     * @param {HTMLElement} element - Element to animate
     * @param {string} animation - Animation type ('fadeIn', 'slideIn', etc.)
     */
    animateElement(element, animation = 'fadeIn') {
        element.classList.add('animate__animated', `animate__${animation}`);

        // Clean up animation classes after completion
        element.addEventListener('animationend', () => {
            element.classList.remove('animate__animated', `animate__${animation}`);
        }, { once: true });
    }

    /**
     * Create the toast container if it doesn't exist
     * @private
     */
    _createToastContainer() {
        let container = document.getElementById('toast-container');
        if (!container) {
            container = document.createElement('div');
            container.id = 'toast-container';
            container.className = 'toast-container position-fixed top-0 end-0 p-3';
            container.style.zIndex = '1055';
            document.body.appendChild(container);
        }
        return container;
    }

    /**
     * Debounce function calls
     * @param {Function} func - Function to debounce
     * @param {number} wait - Wait time in milliseconds
     * @returns {Function} Debounced function
     */
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    /**
     * Throttle function calls
     * @param {Function} func - Function to throttle
     * @param {number} limit - Time limit in milliseconds
     * @returns {Function} Throttled function
     */
    throttle(func, limit) {
        let inThrottle;
        return function () {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }
}

// Create global instance
window.UIComponents = UIComponents;
window.ui = new UIComponents();
