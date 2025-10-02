# V2 Architecture Documentation

## Overview

The V2 system is a **FastAPI-based web application** for personalized crutch design using Bayesian Optimization. It features a modular architecture with separate backend API, frontend interface, and research tools.

## 🚀 Startup Flow: From `main.py` to Running Application

### 1. **Entry Point: `app/main.py`**

```python
# FastAPI application initialization
app = FastAPI(title="Personalising Crutches API", ...)

# CORS middleware for frontend communication
app.add_middleware(CORSMiddleware, ...)

# API routers registration
app.include_router(experiments.router, prefix="/api/experiments")
app.include_router(data.router, prefix="/api/data") 
app.include_router(optimization.router, prefix="/api/optimization")
app.include_router(geometry_sequences.router, prefix="/api/geometry-sequences")
app.include_router(dynamic_geometry.router, prefix="/api/dynamic-geometry")

# Static file serving (frontend)
app.mount("/static", StaticFiles(directory=frontend_path))

# Startup event: create directories
@app.on_event("startup")
async def startup_event():
    create_directories()
```

**Role**: Application entry point, middleware setup, API routing, static file serving

---

## 🏗️ Backend Architecture

### **Core Configuration Layer**

#### `app/core/config.py`

- **Role**: Central configuration management
- **Contains**:
  - `Settings`: Database URL, API settings, data directories
  - `ExperimentConfig`: Crutch parameters, objectives, boundaries
  - `DataProcessingConfig`: Signal processing parameters
  - `StepDetectionConfig`: Algorithm configurations
- **Used by**: All services, API endpoints, frontend

#### `app/core/config/geometry_sequences.py`

- **Role**: Experimental protocol definitions
- **Contains**:
  - `GeometryPoint`: Single crutch configuration
  - `GeometrySequence`: Ordered list of geometries
  - `ExperimentProtocol`: Available protocols (grid_search, baseline_comparison, custom)
  - Predefined sequences: 3×3×3 grid, baseline comparison (14 trials)
- **Used by**: Dynamic geometry service, frontend sequence selection

### **Database Layer**

#### `database/connection.py`

- **Role**: Database connection management
- **Contains**:
  - SQLAlchemy engine configuration
  - Session factory (`SessionLocal`)
  - `get_db()` dependency for FastAPI
- **Used by**: All API endpoints, services

#### `database/models.py`

- **Role**: SQLAlchemy ORM models
- **Contains**:
  - `Participant`: Study participants
  - `CrutchGeometry`: Predefined geometries (legacy)
  - `Trial`: Experimental trials with detailed survey responses
  - `ExperimentSession`: Trial groupings
  - `DataFile`: Uploaded files
  - `OptimizationRun`: BO sessions
- **Used by**: All services, API endpoints

### **Business Logic Layer (Services)**

#### `app/core/services/dynamic_geometry_service.py`

- **Role**: Calculate next geometry based on sequence and progress
- **Key Methods**:
  - `get_next_geometry()`: Returns next geometry to test
  - `get_sequence_progress()`: Progress tracking
  - `create_trial_from_geometry()`: Create trial record
- **Used by**: Dynamic geometry API, frontend

#### `app/core/services/geometry_sequence_service.py`

- **Role**: Manage predefined geometry sequences
- **Key Methods**:
  - `create_geometries_from_sequence()`: Generate database records
  - `get_sequence_geometries()`: Retrieve sequence details
- **Used by**: Geometry sequences API, setup scripts

#### `app/core/services/survey_service.py`

- **Role**: Process detailed survey responses
- **Key Methods**:
  - `process_survey_responses()`: Parse SUS/NRS/TLX data
  - `update_trial_survey_data()`: Store survey results
  - Score calculation methods
- **Used by**: Trial creation, data processing

#### `app/core/services/experiment_service.py`

- **Role**: CRUD operations for experiments
- **Used by**: Experiments API

#### `app/core/services/data_processing_service.py`

- **Role**: Process accelerometer data, step detection
- **Used by**: Data API, trial processing

#### `app/core/services/optimization_service.py`

- **Role**: Bayesian optimization logic
- **Used by**: Optimization API

### **API Layer**

#### `app/api/experiments.py`

- **Role**: Participant and trial management
- **Endpoints**: `/api/experiments/participants`, `/api/experiments/trials`
- **Uses**: Experiment service, database models

#### `app/api/data.py`

- **Role**: Data upload and processing
- **Endpoints**: `/api/data/upload`, `/api/data/process`
- **Uses**: Data processing service

#### `app/api/optimization.py`

- **Role**: Bayesian optimization
- **Endpoints**: `/api/optimization/suggest`, `/api/optimization/run`
- **Uses**: Optimization service

#### `app/api/geometry_sequences.py`

- **Role**: Predefined sequence management
- **Endpoints**: `/api/geometry-sequences/sequences`
- **Uses**: Geometry sequence service

#### `app/api/dynamic_geometry.py`

- **Role**: Dynamic geometry calculation
- **Endpoints**: `/api/dynamic-geometry/next/{participant_id}/{sequence}`
- **Uses**: Dynamic geometry service

---

## 🎨 Frontend Architecture

### **HTML Structure: `frontend/index.html`**

```html
<!-- Mode Selection Screen -->
<div id="mode-selection-screen">
  <button id="select-systematic-mode">Systematic Mode</button>
  <button id="select-bo-mode">Bayesian Optimization</button>
</div>

<!-- Systematic Mode Screen -->
<div id="systematic-screen">
  <!-- Participant Selection -->
  <!-- Geometry Sequence Selection -->
  <!-- Next Geometry Display -->
  <!-- Trial Runner -->
  <!-- Data Visualization -->
</div>
```

**Role**: UI structure, screen management, form elements

### **JavaScript Modules**

#### `frontend/js/app.js`

- **Role**: Main application controller
- **Responsibilities**:
  - Screen navigation
  - Mode selection
  - Component initialization
  - Global state management
- **Uses**: All other JS modules

#### `frontend/js/api.js`

- **Role**: Backend communication
- **Key Methods**:
  - `getParticipants()`, `createParticipant()`
  - `getNextGeometry()`, `getSequenceProgress()`
  - `createTrialFromGeometry()`
  - `uploadData()`, `processData()`
- **Used by**: All frontend modules

#### `frontend/js/systematic-mode.js`

- **Role**: Systematic experiment management
- **Key Features**:
  - Dynamic geometry loading
  - Trial execution
  - Progress tracking
  - Data visualization
- **Classes**: `SystematicMode`
- **Uses**: API client, UI components, device manager

#### `frontend/js/optimization.js`

- **Role**: Bayesian optimization interface
- **Uses**: API client, UI components

#### `frontend/js/device-manager.js`

- **Role**: Device connection management
- **Uses**: Web Serial API for device communication

#### `frontend/js/ui-components.js`

- **Role**: Reusable UI components
- **Contains**: Notification system, modals, form helpers

---

## 🔄 Data Flow: Complete User Journey

### **1. Application Startup**

```
main.py → FastAPI app → CORS middleware → API routers → Static files
```

### **2. User Opens Frontend**

```
Browser → index.html → Load JS modules → app.js → Initialize screens
```

### **3. Systematic Mode: Select Participant**

```
Frontend → api.js → experiments.py → experiment_service.py → database
```

### **4. Select Geometry Sequence**

```
Frontend → api.js → dynamic_geometry.py → dynamic_geometry_service.py
```

### **5. Get Next Geometry**

```
Frontend → api.js → dynamic_geometry.py → dynamic_geometry_service.py → 
Calculate based on progress → Return geometry details (α, β, γ, δ)
```

### **6. Start Trial**

```
Frontend → api.js → dynamic_geometry.py → dynamic_geometry_service.py →
Check if geometry exists in crutch_geometries → Create if needed → 
Create trial with geometry_id reference → Set geometry in form → Load next geometry
```

### **7. Run Trial (Device Connection)**

```
Frontend → device-manager.js → Web Serial API → Device → 
Collect data → Process steps → Calculate metrics
```

### **8. Submit Survey Data**

```
Frontend → api.js → experiments.py → survey_service.py → 
Process SUS/NRS/TLX → Store in database
```

### **9. Data Processing**

```
Frontend → api.js → data.py → data_processing_service.py → 
Step detection → Feature extraction → Store results
```

---

## 📊 Database Schema Integration

### **Trial Lifecycle**

1. **Next Geometry Calculation**: `dynamic_geometry_service.get_next_geometry()` → Calculate based on sequence and progress
2. **Geometry Storage**: Check if geometry exists in `crutch_geometries` table, create if needed
3. **Trial Creation**: `dynamic_geometry_service.create_trial_from_geometry()` → Link trial to geometry
4. **Data Collection**: Device → Raw data storage
5. **Processing**: `data_processing_service` → Step detection, features
6. **Survey**: `survey_service` → SUS/NRS/TLX responses
7. **Completion**: All data stored in `trials` table with `geometry_id` reference

### **Dynamic Geometry System**

The system uses **on-demand geometry creation** rather than pre-stored sequences:

#### **How It Works:**

1. **Sequence Definition**: Geometries defined in `geometry_sequences.py` (not in database)
2. **Dynamic Calculation**: Backend calculates next geometry based on:
   - Selected sequence (baseline_comparison, grid_search_3x3x3, etc.)
   - Participant's completed trials
   - Sequence progress
3. **Geometry Storage**: When trial is created:
   - Check if geometry (α, β, γ, δ) exists in `crutch_geometries`
   - If not found → Create new `CrutchGeometry` record
   - Create `Trial` record with `geometry_id` reference
4. **Data Persistence**: All trial data (survey, IMU, metrics) stored in `trials` table

#### **Example Data Flow:**

```
Participant "P001" starts baseline_comparison sequence:

Trial 1: 
- Dynamic service: "Next = α=95°, β=95°, γ=0°"
- Check CrutchGeometry: Not found
- Create: CrutchGeometry(id=1, α=95, β=95, γ=0, δ=0)
- Create: Trial(id=1, participant_id=1, geometry_id=1, survey_data=...)

Trial 2:
- Dynamic service: "Next = α=85°, β=95°, γ=0°" 
- Check CrutchGeometry: Not found
- Create: CrutchGeometry(id=2, α=85, β=95, γ=0, δ=0)
- Create: Trial(id=2, participant_id=1, geometry_id=2, survey_data=...)

Trial 3:
- Dynamic service: "Next = α=95°, β=95°, γ=0°" (baseline repeat)
- Check CrutchGeometry: Found (id=1)
- Create: Trial(id=3, participant_id=1, geometry_id=1, survey_data=...)
```

#### **Benefits:**

- ✅ **No preset order** - Sequences calculated dynamically
- ✅ **Efficient storage** - No duplicate geometries
- ✅ **Full traceability** - Every tested geometry is stored
- ✅ **Flexible** - Easy to modify sequences without database changes
- ✅ **Scalable** - Works for any number of participants and sequences

### **Key Tables**

- **`participants`**: User information (ID, height, weight, characteristics)
- **`crutch_geometries`**: Individual geometry configurations (α, β, γ, δ) - created on-demand
- **`trials`**: Main experimental data (links to geometry, stores metrics, survey responses)
- **`experiment_sessions`**: Trial groupings (e.g., "baseline_comparison" session)
- **`data_files`**: Uploaded files (IMU data, etc.)
- **`optimization_runs`**: BO sessions

---

## 🔧 Configuration Management

### **Environment Variables** (`.env`)

```bash
DATABASE_URL=sqlite:///./experiments.db
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=False
```

### **Sequence Configuration**

- **Baseline Comparison**: 14 trials (baseline → 3 sessions → baseline)
- **Grid Search**: 27 trials (3×3×3 grid)
- **Custom**: User-defined sequences

---

## 🚀 Running the Application

### **Development**

```bash
cd /Users/riccardoconci/Local_documents/Personalising-Crutches/V2
uvicorn app.main:app --reload
```

### **Production**

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### **Access Points**

- **Frontend**: `http://localhost:8000`
- **API Docs**: `http://localhost:8000/api/docs`
- **Health Check**: `http://localhost:8000/health`

---

## 🔄 Key Integration Points

### **1. Frontend ↔ Backend**

- **API Client** (`api.js`) handles all communication
- **RESTful endpoints** for data exchange
- **JSON** for data serialization

### **2. Database ↔ Services**

- **SQLAlchemy ORM** for database operations
- **Dependency injection** for database sessions
- **Transaction management** for data integrity

### **3. Configuration ↔ All Components**

- **Centralized config** in `app/core/config.py`
- **Environment-based** settings
- **Type validation** with Pydantic

### **4. Device ↔ Frontend**

- **Web Serial API** for device communication
- **Real-time data** streaming
- **Error handling** for connection issues

---

## 📁 File Organization Summary

```
V2/
├── app/                    # FastAPI backend
│   ├── main.py            # Application entry point
│   ├── api/               # API endpoints
│   └── core/              # Business logic
│       ├── config/        # Configuration
│       ├── models/        # Pydantic models
│       └── services/      # Business services
├── database/              # Data layer
│   ├── models.py         # SQLAlchemy models
│   └── connection.py     # Database connection
├── frontend/              # Web interface
│   ├── index.html        # Main HTML
│   └── js/               # JavaScript modules
├── research/              # Research tools
├── scripts/               # Setup scripts
└── docs/                  # Documentation
```

This architecture provides **separation of concerns**, **modularity**, and **scalability** while maintaining a clean, understandable codebase.

---

## 🎯 **Key Architectural Decisions**

### **Dynamic vs. Pre-stored Geometries**

- **Decision**: Use dynamic geometry calculation with on-demand database storage
- **Rationale**:
  - No preset order - sequences calculated based on progress
  - Efficient storage - no duplicate geometries
  - Full traceability - every tested geometry is stored
  - Flexible - easy to modify sequences without database changes

### **Database Design**

- **`crutch_geometries`**: Stores individual geometry configurations (α, β, γ, δ)
- **`trials`**: Links to geometries via `geometry_id`, stores all trial data
- **No sequence tables**: Sequences defined in code, not database

### **Frontend Architecture**

- **Modular JavaScript**: Separate files for different concerns
- **Dynamic UI**: "Next Geometry" display instead of static grid
- **Real-time updates**: Progress tracking and sequence management

### **API Design**

- **RESTful endpoints**: Clear separation of concerns
- **Dynamic calculation**: Backend determines next geometry
- **Flexible sequences**: Easy to add new experimental protocols
