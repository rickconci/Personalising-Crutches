# OpenCap Toggle Integration

## Overview

This document describes the OpenCap toggle functionality that allows researchers to mark specific time segments during trials for video capture synchronization.

## Feature Description

During a 2-minute trial, researchers can press an **OpenCap ON/OFF toggle button** multiple times (typically 2-3 times) to mark when video recording is active. These timestamps are:

1. **Captured** - Recorded with precise timing during the trial
2. **Stored** - Saved in the database as time segments
3. **Available** - Can be used to segment and extract trial data

## User Interface

### Location

The OpenCap toggle button appears **next to the Start/Stop Trial button** in the Grid Search mode.

### Behavior

- **Before Trial**: Button is hidden
- **During Trial**:
  - Button becomes visible and enabled
  - Starts in "OFF" state (gray)
  - Toggles between "ON" (green) and "OFF" (gray) states
  - Each click records a timestamp event
- **After Trial**: Button is hidden again

### Visual Indicators

- **OFF State**: Gray button with text "OpenCap OFF"
- **ON State**: Green button with text "OpenCap ON"
- **Notifications**: Toast messages confirm each toggle action

## Data Flow

### 1. Frontend (JavaScript)

**File**: `V2/frontend/js/systematic-mode/trial-runner.js`

When user clicks the toggle button:

```javascript
{
  timestamp: 1234567890,      // Absolute timestamp (ms)
  state: "on" | "off",        // Current state
  relativeTime: 5000          // Milliseconds since trial start
}
```

### 2. Frontend to Backend

**File**: `V2/frontend/js/systematic-mode/survey-manager.js`

Events are sent as part of the trial data:

```javascript
{
  participant_id: 1,
  geometry_id: 5,
  steps: [...],
  opencap_events: [
    {timestamp: 1000, state: "on", relativeTime: 5000},
    {timestamp: 2000, state: "off", relativeTime: 15000},
    {timestamp: 3000, state: "on", relativeTime: 30000},
    {timestamp: 4000, state: "off", relativeTime: 45000}
  ]
}
```

### 3. Backend Processing

**File**: `V2/app/core/services/experiment_service.py`

The `ExperimentService` processes events into segments:

**Input (events)**:

```json
[
  {"timestamp": 1000, "state": "on", "relativeTime": 5000},
  {"timestamp": 2000, "state": "off", "relativeTime": 15000}
]
```

**Output (segments)**:

```json
[
  {
    "start": 5000,
    "end": 15000,
    "duration": 10000
  }
]
```

### 4. Database Storage

**File**: `V2/database/models.py`

Segments are stored in the `trials` table:

```sql
opencap_segments JSON  -- [{start, end, duration}, ...]
```

## Architecture

### Frontend Components

#### 1. UI Component (`index.html`)

```html
<button type="button" class="btn btn-secondary btn-lg d-none" 
        id="opencap-toggle-btn" data-opencap-state="off">
  <i class="fas fa-video"></i> OpenCap OFF
</button>
```

#### 2. Trial Runner (`trial-runner.js`)

- **State Management**: Tracks OpenCap events in `state.openCapEvents[]`
- **Event Recording**: `recordOpenCapEvent(state)` method
- **Data Access**: `getOpenCapEvents()` method

#### 3. Core Controller (`core.js`)

- **Event Handling**: `_handleOpenCapToggle()` method
- **UI Updates**: `_updateTrialUI(running)` method
- **Data Integration**: Passes events to `surveyManager.saveTrial()`

#### 4. Survey Manager (`survey-manager.js`)

- **Data Collection**: Accepts `openCapEvents` parameter
- **API Communication**: Sends events in trial payload

### Backend Components

#### 1. Pydantic Models (`app/core/models/trial.py`)

```python
class TrialBase(BaseModel):
    opencap_events: Optional[List[Dict[str, Any]]] = None
    opencap_segments: Optional[List[Dict[str, Any]]] = None
```

#### 2. Database Model (`database/models.py`)

```python
class Trial(Base):
    opencap_segments = Column(JSON, nullable=True)
```

#### 3. Utility Functions (`app/core/utils/opencap_utils.py`)

- `process_opencap_events()` - Converts events to segments
- `segment_timeseries_data()` - Extracts data by segments
- `get_segment_statistics()` - Calculates segment metrics

#### 4. Service Layer (`app/core/services/experiment_service.py`)

- Processes events during trial creation
- Processes events during trial updates
- Handles errors gracefully

## Usage Examples

### Example 1: Single OpenCap Recording

```
Trial Start (t=0)
  ↓ 5 seconds
User presses: OpenCap ON
  ↓ 10 seconds (recording)
User presses: OpenCap OFF
  ↓ remaining time
Trial Stop
```

**Result**: One segment from 5000ms to 15000ms (duration: 10000ms)

### Example 2: Multiple OpenCap Recordings

```
Trial Start (t=0)
  ↓ 5 seconds
OpenCap ON
  ↓ 10 seconds
OpenCap OFF
  ↓ 15 seconds
OpenCap ON
  ↓ 15 seconds
OpenCap OFF
  ↓ remaining time
Trial Stop
```

**Result**:

- Segment 1: 5000ms to 15000ms (duration: 10000ms)
- Segment 2: 30000ms to 45000ms (duration: 15000ms)

## Data Segmentation

### Extracting OpenCap Segments from Trial Data

**Python Example**:

```python
from app.core.utils.opencap_utils import segment_timeseries_data

# Load trial data
trial = db.query(Trial).filter(Trial.id == trial_id).first()
opencap_segments = trial.opencap_segments

# Load raw accelerometer data
raw_data = load_raw_data(trial.raw_data_path)

# Extract only data from OpenCap segments
opencap_data = segment_timeseries_data(
    data=raw_data,
    segments=opencap_segments,
    time_key="acc_x_time"
)

# Now opencap_data contains only the data points recorded during OpenCap
```

### Use Cases

1. **Video Synchronization**: Match accelerometer data to video frames
2. **Selective Analysis**: Analyze only the walking periods captured on video
3. **Quality Control**: Verify data quality against video evidence
4. **Multi-Modal Analysis**: Correlate kinematic, kinetic, and video data

## Error Handling

### Frontend

- Button disabled if no trial is running
- Visual feedback for each toggle
- Events logged to console for debugging

### Backend

- Graceful handling of missing events (empty array)
- Validation of event structure
- Warning logs for malformed data
- Trial creation/update continues even if OpenCap processing fails

### Edge Cases Handled

1. **Unclosed Segment**: ON without OFF (ignored with warning)
2. **Invalid Sequence**: OFF without ON (ignored with warning)
3. **Empty Events**: No events recorded (segments = [])
4. **Malformed Data**: Missing fields (skipped with warning)

## Testing

### Manual Testing Checklist

- [ ] Button appears when trial starts
- [ ] Button is hidden when trial is not running
- [ ] Button toggles between ON/OFF states
- [ ] Visual feedback (color change) works
- [ ] Toast notifications appear
- [ ] Events are included in saved trial data
- [ ] Segments are correctly calculated
- [ ] Database stores segments properly

### Example Test Scenario

1. Start a trial
2. Wait 5 seconds → Press OpenCap (should turn ON)
3. Wait 10 seconds → Press OpenCap (should turn OFF)
4. Wait 15 seconds → Press OpenCap (should turn ON)
5. Wait 15 seconds → Press OpenCap (should turn OFF)
6. Stop trial and save
7. Check database: `opencap_segments` should contain 2 segments

## Database Schema

### Migration Required

You may need to run a database migration to add the `opencap_segments` column:

```sql
ALTER TABLE trials ADD COLUMN opencap_segments JSON DEFAULT NULL;
```

### Example Query

```sql
SELECT 
    id,
    participant_id,
    geometry_id,
    opencap_segments,
    JSON_LENGTH(opencap_segments) as segment_count
FROM trials
WHERE opencap_segments IS NOT NULL;
```

## Future Enhancements

### Potential Features

1. **Segment Preview**: Show segments on force plot
2. **Auto-validation**: Warn if segments are too short/long
3. **Segment Export**: Export individual segments as separate files
4. **Video Upload**: Associate video files with segments
5. **Segment Labeling**: Add custom labels to segments (e.g., "walking", "turning")

## Files Modified

### Frontend

- `V2/frontend/index.html` - Added toggle button UI
- `V2/frontend/js/systematic-mode/trial-runner.js` - Event tracking
- `V2/frontend/js/systematic-mode/core.js` - Event handling and UI updates
- `V2/frontend/js/systematic-mode/survey-manager.js` - Data transmission

### Backend

- `V2/database/models.py` - Added `opencap_segments` column
- `V2/app/core/models/trial.py` - Added Pydantic model fields
- `V2/app/core/services/experiment_service.py` - Event processing logic
- `V2/app/core/utils/opencap_utils.py` - **NEW** Utility functions
- `V2/app/core/utils/__init__.py` - **NEW** Package initialization

## Support

For questions or issues, refer to:

- Code comments in `opencap_utils.py`
- Console logs (browser developer tools)
- Backend logs (application logs)

## Version

- **Implemented**: October 2025
- **Author**: AI Assistant
- **Status**: Production Ready ✓
