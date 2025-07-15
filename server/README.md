# Personalised Crutches Optimisation - Backend Server

This backend server provides API endpoints for the Personalised Crutches Optimisation web application. It handles file uploads, data analysis, and Bayesian optimization.

## Setup

1. Create a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Create an uploads directory (if it doesn't exist):

   ```bash
   mkdir uploads
   ```

## Running the Server

Start the Flask development server:

```bash
python app.py
```

The server will run at `http://localhost:5000` by default.

## API Endpoints

- `POST /api/upload`: Upload accelerometer data files
- `POST /api/analyze`: Analyze accelerometer data with subjective metrics
- `POST /api/optimize`: Run Bayesian optimization on experiment history
- `GET /api/history`: Get the current experiment history

## Integration with Frontend

The frontend (in the `../SITE` directory) communicates with this backend server using fetch API calls. Make sure the server is running when using the web application.

## Note on Data Storage

For simplicity, this demo server stores data in memory. In a production environment, you would use a database to persist data between server restarts.
