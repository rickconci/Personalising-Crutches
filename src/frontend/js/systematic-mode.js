/**
 * Systematic Mode - Barrel file for backward compatibility
 * 
 * This file re-exports the modular SystematicMode implementation
 * to maintain backward compatibility with existing code.
 * 
 * The implementation has been refactored into smaller, focused modules:
 * - core.js: Main orchestrator
 * - trial-runner.js: Trial lifecycle management
 * - step-manager.js: Step manipulation and variance calculations
 * - data-processor.js: File upload and data processing
 * - ui-renderer.js: All rendering logic
 * - geometry-sequencer.js: Dynamic geometry loading
 * - survey-manager.js: Survey collection and trial saving
 */

import { SystematicMode } from './systematic-mode/core.js';

// Export for ES6 modules
export { SystematicMode };

// Export to global window for backward compatibility
window.SystematicMode = SystematicMode;

