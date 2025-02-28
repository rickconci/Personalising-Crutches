# Personalised Crutches Optimisation Web Application

This web application provides an interface for collecting data, analyzing accelerometer information, and optimizing crutch geometry based on both subjective and objective metrics.

## Project Structure

```
project/
├── SITE/               # Frontend
│   ├── css/
│   │   └── styles.css
│   ├── js/
│   │   └── main.js
│   ├── index.html
│   └── README.md
└── server/             # Backend
    ├── app.py          # Flask server
    ├── analysis.py     # Data analysis functions
    ├── optimization.py # Bayesian optimization functions
    ├── requirements.txt
    └── uploads/        # Directory for uploaded files
```

## Features

The application is divided into three main sections:

### A) Data Collection

1. **Crutch Geometry Input**: Record crutch parameters (alpha, beta, gamma, delta)
2. **Subjective Metrics**: Rate pain, effort, and instability on a scale of 0-6
3. **Accelerometer Data Upload**: Upload accelerometer data files

### B) Analysis

1. **Analysis Controls**: Configure the number of bouts and metric weightings
2. **Results Visualization**: Display subjective and objective losses with weighted sums

### C) Optimisation

1. **Experiment History**: Track all crutch geometry and loss pairs
2. **Bayesian Optimisation**: Run optimization with configurable options
3. **Optimisation Results**: Display recommended crutch geometry

## Setup and Usage

### Frontend

The frontend is a static web application that can be opened directly in a browser or served with a simple HTTP server:

```bash
# Python 3
python -m http.server 8000

# Then visit http://localhost:8000 in your browser
```

### Backend

The backend requires Python with Flask and other dependencies. See the README in the server directory for detailed setup instructions.

Basic setup:

```bash
cd server
pip install -r requirements.txt
python app.py

# Backend will run at http://localhost:5000
```

## How It Works

1. **Data Collection**: Users input crutch parameters and subjective metrics, then upload accelerometer data
2. **Analysis**: The backend processes the accelerometer data and combines it with subjective metrics
3. **Optimization**: Bayesian optimization is used to find optimal crutch parameters based on previous results

## Technical Details

- Frontend: HTML, CSS (Bootstrap), JavaScript
- Backend: Python with Flask, NumPy, Pandas, scikit-learn
- Communication: RESTful API calls
- Optimization: Bayesian optimization with Gaussian Process Regression

## Future Work

- Implement actual backend API endpoints for analysis and optimization
- Add data visualization for accelerometer data
- Improve the optimization algorithm with more advanced features

## Notes

This is a basic implementation that needs to be connected to the actual analysis and optimization functions from the Jupyter notebook. The UI provides the structure and interaction points for the complete system.
