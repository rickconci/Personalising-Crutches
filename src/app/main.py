"""
Main FastAPI application for the Personalising Crutches system.

This module provides the main application entry point and configuration.
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
from pathlib import Path

from .core.config import settings
from .api import experiments, data, optimization

# Create FastAPI application
app = FastAPI(
    title="Personalising Crutches API",
    description="Bayesian Optimization for Personalized Crutch Design",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(experiments.router, prefix="/api/experiments", tags=["experiments"])
app.include_router(data.router, prefix="/api/data", tags=["data"])
app.include_router(optimization.router, prefix="/api/optimization", tags=["optimization"])

# Mount static files
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

@app.get("/")
async def root():
    """Root endpoint - serve the frontend."""
    frontend_file = frontend_path / "index.html"
    if frontend_file.exists():
        return FileResponse(frontend_file)
    else:
        return {"message": "Personalising Crutches API", "version": "0.1.0"}


# Create necessary directories
def create_directories():
    """Create necessary directories for the application."""
    directories = [
        settings.data_directory,
        settings.raw_data_directory,
        settings.processed_data_directory,
        settings.results_directory,
        settings.plots_directory,
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Initialize directories on startup
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    create_directories()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )
