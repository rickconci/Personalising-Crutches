"""
Basic API tests for Personalising Crutches.

These tests verify that the API endpoints are working correctly.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data

def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200

def test_get_participants():
    """Test getting participants."""
    response = client.get("/api/experiments/participants")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_get_geometries():
    """Test getting geometries."""
    response = client.get("/api/experiments/geometries")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_get_trials():
    """Test getting trials."""
    response = client.get("/api/experiments/trials")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_create_participant():
    """Test creating a participant."""
    participant_data = {
        "name": "Test Participant",
        "characteristics": {
            "height": 175.0,
            "weight": 70.0,
            "forearm_length": 25.0,
            "fitness_level": 3
        }
    }
    
    response = client.post("/api/experiments/participants", json=participant_data)
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Test Participant"
    assert "id" in data

def test_get_algorithms():
    """Test getting available algorithms."""
    response = client.get("/api/data/algorithms")
    assert response.status_code == 200
    data = response.json()
    assert "algorithms" in data
    assert isinstance(data["algorithms"], list)

def test_get_acquisition_functions():
    """Test getting acquisition functions."""
    response = client.get("/api/optimization/acquisition-functions")
    assert response.status_code == 200
    data = response.json()
    assert "acquisition_functions" in data
    assert isinstance(data["acquisition_functions"], list)

def test_get_objectives():
    """Test getting optimization objectives."""
    response = client.get("/api/optimization/objectives")
    assert response.status_code == 200
    data = response.json()
    assert "objectives" in data
    assert isinstance(data["objectives"], list)

if __name__ == "__main__":
    pytest.main([__file__])
