"""
EMG Analysis Package

A comprehensive package for analyzing EMG data from Trigno systems,
including step detection, gait analysis, and visualization.
"""

from .emg_parser import EMGParser
from .emg_visualizer import EMGVisualizer
from .emg_gait_analyzer import EMGGaitAnalyzer

__version__ = "1.0.0"
__author__ = "Riccardo Conci"

__all__ = [
    "EMGParser",
    "EMGVisualizer", 
    "EMGGaitAnalyzer"
]
