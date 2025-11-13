"""
EMG Interactive Step Editor

Provides interactive step editing functionality for EMG data analysis,
similar to the systematic mode but specifically for EMG envelope data.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.utils
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import json
from emg_parser import EMGData
from emg_visualizer import EMGVisualizer


class EMGStepManager:
    """
    Manages step detection and editing for EMG data.
    Similar to the frontend StepManager but for EMG analysis.
    """
    
    def __init__(self):
        self.steps: List[float] = []
        self.plot_data: Optional[Dict] = None
        self.emg_data: Optional[EMGData] = None
        self.processing_info: Optional[Dict] = None
        
    def clear(self):
        """Clear all steps and reset state."""
        self.steps = []
        self.plot_data = None
        self.emg_data = None
        self.processing_info = None
    
    def load_emg_data(self, emg_data: EMGData, processing_info: Dict):
        """Load EMG data and processing results."""
        self.emg_data = emg_data
        self.processing_info = processing_info
        
        # Extract initial steps from processing results
        if 'step_times' in processing_info:
            self.steps = processing_info['step_times'].tolist()
        else:
            self.steps = []
    
    def add_step(self, time: float, tolerance: float = 0.5) -> bool:
        """
        Add a step at the specified time.
        
        Args:
            time: Time in seconds to add the step
            tolerance: Minimum time difference to consider steps distinct
            
        Returns:
            True if step was added, False if too close to existing step
        """
        # Check for duplicates within tolerance
        for existing_time in self.steps:
            if abs(existing_time - time) < tolerance:
                return False
        
        # Add step and sort
        self.steps.append(time)
        self.steps.sort()
        return True
    
    def remove_step(self, time: float, tolerance: float = 0.5) -> bool:
        """
        Remove a step near the specified time.
        
        Args:
            time: Time in seconds to remove the step
            tolerance: Maximum time difference to consider steps the same
            
        Returns:
            True if step was removed, False if no step found
        """
        for i, existing_time in enumerate(self.steps):
            if abs(existing_time - time) < tolerance:
                self.steps.pop(i)
                return True
        return False
    
    def get_steps(self) -> List[float]:
        """Get current list of step times."""
        return self.steps.copy()
    
    def calculate_variance(self) -> float:
        """
        Calculate step-to-step variance (instability loss).
        
        Returns:
            Variance of step intervals
        """
        if len(self.steps) < 2:
            return 0.0
        
        intervals = np.diff(self.steps)
        return float(np.var(intervals))
    
    def get_plot_data(self) -> Dict:
        """Get current plot data with steps."""
        if self.plot_data is None:
            return {}
        
        return {
            **self.plot_data,
            'step_times': self.steps
        }


class EMGInteractiveEditor:
    """
    Interactive EMG step editor with Plotly visualization.
    
    Provides the same functionality as the frontend systematic mode
    but for EMG data analysis.
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.step_manager = EMGStepManager()
        self.visualizer = EMGVisualizer()
        
    def create_interactive_plot(self, 
                               emg_data: EMGData, 
                               processing_info: Dict,
                               trial_name: str) -> str:
        """
        Create an interactive EMG plot with step editing capabilities.
        
        Args:
            emg_data: EMG data object
            processing_info: Processing results from EMG analysis
            trial_name: Name of the trial
            
        Returns:
            Path to the generated HTML file
        """
        # Load data into step manager
        self.step_manager.load_emg_data(emg_data, processing_info)
        
        # Create the interactive plot
        fig = self._create_interactive_figure(emg_data, processing_info)
        
        # Generate HTML with interactive functionality
        html_content = self._generate_interactive_html(fig, trial_name)
        
        # Save to file
        output_file = self.output_dir / f"{trial_name}_interactive_emg.html"
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        return str(output_file)
    
    def _create_interactive_figure(self, emg_data: EMGData, processing_info: Dict) -> go.Figure:
        """Create the interactive Plotly figure."""
        # Get time signal
        time_signal = emg_data.data['time_s'].values
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=[
                'Original/Cleaned/Rectified EMG',
                'EMG Envelope', 
                'Normalized Envelope with Peaks'
            ],
            vertical_spacing=0.08,
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}]]
        )
        
        # Plot 1: Signal processing pipeline
        if 'cleaned_emg_sample' in processing_info:
            sample_time = time_signal[::10]  # Match the sampling
            fig.add_trace(go.Scatter(
                x=sample_time,
                y=processing_info['cleaned_emg_sample'],
                mode='lines',
                name='Cleaned EMG',
                line=dict(color='blue', width=1),
                opacity=0.7
            ), row=1, col=1)
        
        # Plot 2: EMG Envelope
        if 'envelope_ds' in processing_info:
            ds_time = processing_info['time_s']
            ds_env = processing_info['envelope_ds']
            
            fig.add_trace(go.Scatter(
                x=ds_time,
                y=ds_env,
                mode='lines',
                name='EMG Envelope',
                line=dict(color='green', width=2)
            ), row=2, col=1)
        
        # Plot 3: Normalized Envelope with Peaks
        if 'envelope_ds' in processing_info:
            ds_time = processing_info['time_s']
            ds_env = processing_info['envelope_ds']
            ds_thr = processing_info['thr_ds']
            ds_height = processing_info.get('height_ds', ds_thr)
            
            # Envelope
            fig.add_trace(go.Scatter(
                x=ds_time,
                y=ds_env,
                mode='lines',
                name='Envelope + Threshold',
                line=dict(color='orange', width=2)
            ), row=3, col=1)
            
            # Adaptive threshold
            fig.add_trace(go.Scatter(
                x=ds_time,
                y=ds_thr,
                mode='lines',
                name='Adaptive Threshold',
                line=dict(color='red', width=1, dash='dash'),
                opacity=0.7
            ), row=3, col=1)
            
            # Hybrid threshold (if different)
            if not np.array_equal(ds_thr, ds_height):
                fig.add_trace(go.Scatter(
                    x=ds_time,
                    y=ds_height,
                    mode='lines',
                    name='Hybrid Threshold',
                    line=dict(color='purple', width=1, dash='dot'),
                    opacity=0.8
                ), row=3, col=1)
        
        # Add step markers to all plots
        steps = self.step_manager.get_steps()
        if len(steps) > 0:
            for i, (row, col) in enumerate([(1, 1), (2, 1), (3, 1)]):
                # Get values at step times
                if row == 1 and 'cleaned_emg_sample' in processing_info:
                    step_values = np.interp(steps, sample_time, processing_info['cleaned_emg_sample'])
                elif row == 2 and 'envelope_ds' in processing_info:
                    step_values = np.interp(steps, ds_time, ds_env)
                elif row == 3 and 'envelope_ds' in processing_info:
                    step_values = np.interp(steps, ds_time, ds_env)
                else:
                    step_values = [0] * len(steps)
                
                fig.add_trace(go.Scatter(
                    x=steps,
                    y=step_values,
                    mode='markers',
                    name=f'Steps ({len(steps)})',
                    marker=dict(
                        color='red',
                        size=8,
                        symbol='x',
                        line=dict(width=2, color='darkred')
                    ),
                    showlegend=(i == 0)  # Only show legend for first plot
                ), row=row, col=col)
        
        # Update layout
        fig.update_layout(
            height=900,
            title=f"Interactive EMG Analysis - {emg_data.trial_name}",
            showlegend=True,
            hovermode='closest'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Time (s)", row=3, col=1)
        fig.update_yaxes(title_text="Amplitude (mV)", row=1, col=1)
        fig.update_yaxes(title_text="Envelope (mV)", row=2, col=1)
        fig.update_yaxes(title_text="Normalized Envelope", row=3, col=1)
        
        return fig
    
    def _generate_interactive_html(self, fig: go.Figure, trial_name: str) -> str:
        """Generate HTML with interactive functionality."""
        # Extract plot data and layout for dynamic rendering
        plot_data = fig.to_dict()
        traces = plot_data['data']
        layout = plot_data['layout']
        
        # Convert to JSON for JavaScript
        traces_json = json.dumps(traces, cls=plotly.utils.PlotlyJSONEncoder)
        layout_json = json.dumps(layout, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Create full interactive HTML
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive EMG Analysis - {trial_name}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
    <style>
        .plotly-chart-container {{
            width: 100%;
            height: 600px;
        }}
        .step-item {{
            cursor: pointer;
        }}
        .step-item:hover {{
            background-color: #f8f9fa;
        }}
    </style>
</head>
<body>
    <div class="container-fluid py-3">
        <div class="row">
            <div class="col-md-8">
                <div class="card">
                    <div class="card-header">
                        <h4>Interactive EMG Analysis - {trial_name}</h4>
                        <p class="mb-0 text-muted">Click on the plot to add steps, or use the table below to remove steps.</p>
                    </div>
                    <div class="card-body">
                        <div id="emg-plot" class="plotly-chart-container"></div>
                    </div>
                </div>
            </div>
            
            <div class="col-md-4">
                <div class="card">
                    <div class="card-header">
                        <h5>Step Management</h5>
                    </div>
                    <div class="card-body">
                        <div class="mb-3">
                            <h6>Detected Steps (<span id="step-count">{len(self.step_manager.get_steps())}</span>)</h6>
                            <p class="small text-muted">Click on the plot to add steps, or use the delete buttons below to remove steps.</p>
                        </div>
                        
                        <div class="table-responsive" style="max-height: 300px; overflow-y: auto;">
                            <table class="table table-sm table-striped">
                                <thead>
                                    <tr>
                                        <th>Step #</th>
                                        <th>Time (s)</th>
                                        <th>Action</th>
                                    </tr>
                                </thead>
                                <tbody id="step-list">
                                    <!-- Steps will be populated here -->
                                </tbody>
                            </table>
                        </div>
                        
                        <div class="mt-3">
                            <div class="row text-center">
                                <div class="col-6">
                                    <div class="border rounded p-2">
                                        <div class="text-muted small">Instability Loss</div>
                                        <div class="h5 mb-0" id="instability-loss">0.00</div>
                                    </div>
                                </div>
                                <div class="col-6">
                                    <div class="border rounded p-2">
                                        <div class="text-muted small">Total Steps</div>
                                        <div class="h5 mb-0" id="total-steps">0</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="mt-3 d-grid gap-2">
                            <button class="btn btn-primary" id="save-steps-btn">
                                <i class="fas fa-save me-2"></i>Save Steps
                            </button>
                            <button class="btn btn-secondary" id="clear-all-btn">
                                <i class="fas fa-trash me-2"></i>Clear All Steps
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Step management
        let steps = {json.dumps(self.step_manager.get_steps())};
        let plotData = {traces_json};
        let plotLayout = {layout_json};
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {{
            renderPlot();
            updateStepsList();
            updateMetrics();
            setupPlotInteractions();
        }});
        
        function renderPlot() {{
            const plotDiv = document.getElementById('emg-plot');
            if (plotDiv) {{
                // Create a copy of the plot data
                let traces = JSON.parse(JSON.stringify(plotData));
                
                // Add step markers to all subplots
                if (steps.length > 0) {{
                    // Get the data for each subplot
                    const subplotData = [
                        traces[0], // First subplot (cleaned EMG)
                        traces[1], // Second subplot (envelope)
                        traces[2]  // Third subplot (normalized envelope)
                    ];
                    
                    // Add step markers to each subplot
                    subplotData.forEach((data, subplotIndex) => {{
                        if (data && data.x && data.y) {{
                            const stepX = steps;
                            const stepY = steps.map(step => {{
                                // Interpolate step values from the data
                                const x = data.x;
                                const y = data.y;
                                if (x.length === 0 || y.length === 0) return 0;
                                
                                // Simple linear interpolation
                                for (let i = 0; i < x.length - 1; i++) {{
                                    if (step >= x[i] && step <= x[i + 1]) {{
                                        const t = (step - x[i]) / (x[i + 1] - x[i]);
                                        return y[i] + t * (y[i + 1] - y[i]);
                                    }}
                                }}
                                return y[0]; // Fallback
                            }});
                            
                            // Add step marker trace
                            traces.push({{
                                x: stepX,
                                y: stepY,
                                mode: 'markers',
                                name: subplotIndex === 0 ? `Steps (${{steps.length}})` : '',
                                marker: {{
                                    color: 'red',
                                    size: 8,
                                    symbol: 'x',
                                    line: {{ width: 2, color: 'darkred' }}
                                }},
                                showlegend: subplotIndex === 0,
                                xaxis: subplotIndex === 0 ? 'x' : subplotIndex === 1 ? 'x2' : 'x3',
                                yaxis: subplotIndex === 0 ? 'y' : subplotIndex === 1 ? 'y2' : 'y3'
                            }});
                        }}
                    }});
                }}
                
                // Render the plot
                Plotly.newPlot(plotDiv, traces, plotLayout, {{
                    displayModeBar: true,
                    displaylogo: false
                }});
            }}
        }}
        
        function setupPlotInteractions() {{
            const plotDiv = document.getElementById('emg-plot');
            if (plotDiv) {{
                plotDiv.on('plotly_click', function(data) {{
                    const point = data.points[0];
                    const clickedTime = point.x;
                    
                    // Check if clicking on a step marker (to remove it)
                    if (point.curveNumber >= 3) {{ // Step markers start at curve 3
                        const stepIndex = point.curveNumber - 3;
                        if (stepIndex < steps.length) {{
                            removeStep(stepIndex);
                            return;
                        }}
                    }}
                    
                    // Otherwise, add a new step
                    if (addStep(clickedTime)) {{
                        updateStepsList();
                        updateMetrics();
                        updatePlot();
                        showNotification(`Step added at ${{clickedTime.toFixed(2)}}s`, 'success');
                    }} else {{
                        showNotification('A step already exists near this time', 'warning');
                    }}
                }});
            }}
        }}
        
        function addStep(time, tolerance = 0.5) {{
            // Check for duplicates
            for (let existingTime of steps) {{
                if (Math.abs(existingTime - time) < tolerance) {{
                    return false;
                }}
            }}
            
            steps.push(time);
            steps.sort((a, b) => a - b);
            return true;
        }}
        
        function removeStep(index) {{
            steps.splice(index, 1);
            updateStepsList();
            updateMetrics();
            updatePlot();
            showNotification('Step removed', 'info');
        }}
        
        function updateStepsList() {{
            const tbody = document.getElementById('step-list');
            tbody.innerHTML = '';
            
            steps.forEach((step, index) => {{
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${{index + 1}}</td>
                    <td>${{step.toFixed(3)}}</td>
                    <td>
                        <button class="btn btn-sm btn-outline-danger" onclick="removeStep(${{index}})">
                            <i class="fas fa-trash"></i>
                        </button>
                    </td>
                `;
                tbody.appendChild(row);
            }});
            
            document.getElementById('step-count').textContent = steps.length;
            document.getElementById('total-steps').textContent = steps.length;
        }}
        
        function updateMetrics() {{
            if (steps.length < 2) {{
                document.getElementById('instability-loss').textContent = '0.00';
                return;
            }}
            
            // Calculate variance of intervals
            const intervals = [];
            for (let i = 1; i < steps.length; i++) {{
                intervals.push(steps[i] - steps[i-1]);
            }}
            
            const mean = intervals.reduce((a, b) => a + b, 0) / intervals.length;
            const variance = intervals.reduce((sum, interval) => sum + Math.pow(interval - mean, 2), 0) / intervals.length;
            
            document.getElementById('instability-loss').textContent = variance.toFixed(3);
        }}
        
        function updatePlot() {{
            // Simply re-render the entire plot with updated steps
            renderPlot();
        }}
        
        function saveSteps() {{
            const data = {{
                trial_name: '{trial_name}',
                steps: steps,
                timestamp: new Date().toISOString()
            }};
            
            // Create and download JSON file
            const blob = new Blob([JSON.stringify(data, null, 2)], {{type: 'application/json'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = '{trial_name}_steps.json';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            showNotification('Steps saved successfully!', 'success');
        }}
        
        function clearAllSteps() {{
            if (confirm('Are you sure you want to clear all steps?')) {{
                steps = [];
                updateStepsList();
                updateMetrics();
                updatePlot();
                showNotification('All steps cleared', 'info');
            }}
        }}
        
        function showNotification(message, type = 'info') {{
            // Simple notification - you could enhance this
            const alertClass = type === 'success' ? 'alert-success' : 
                             type === 'warning' ? 'alert-warning' : 
                             type === 'error' ? 'alert-danger' : 'alert-info';
            
            const notification = document.createElement('div');
            notification.className = `alert ${{alertClass}} alert-dismissible fade show position-fixed`;
            notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
            notification.innerHTML = `
                ${{message}}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            `;
            
            document.body.appendChild(notification);
            
            // Auto-remove after 3 seconds
            setTimeout(() => {{
                if (notification.parentNode) {{
                    notification.remove();
                }}
            }}, 3000);
        }}
        
        // Event listeners
        document.getElementById('save-steps-btn').addEventListener('click', saveSteps);
        document.getElementById('clear-all-btn').addEventListener('click', clearAllSteps);
    </script>
</body>
</html>
        """
        
        return html_template
    
    def load_saved_steps(self, json_file: Path) -> List[float]:
        """Load steps from a saved JSON file."""
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                return data.get('steps', [])
        except Exception as e:
            print(f"Error loading steps: {e}")
            return []
    
    def save_steps(self, steps: List[float], trial_name: str) -> Path:
        """Save steps to a JSON file."""
        data = {
            'trial_name': trial_name,
            'steps': steps,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        output_file = self.output_dir / f"{trial_name}_steps.json"
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        return output_file
