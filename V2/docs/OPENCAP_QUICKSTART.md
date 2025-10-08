# OpenCap Toggle - Quick Start Guide

## What Changed?

A new OpenCap toggle button has been added to help you mark time points during trials for video synchronization.

## ✅ Database Migration Complete

The database has been updated with the new `opencap_events` column.

## How It Works

### During a Trial

1. **Start a trial** (Grid Search mode)
2. **OpenCap button appears** next to the Start/Stop button
3. **Click to toggle ON** (button turns green) when you start video recording
4. **Click to toggle OFF** (button turns gray) when you stop video recording
5. Repeat 2-3 times during the 2-minute trial as needed
6. **Stop the trial** and save as normal

### What Gets Saved

✅ **ALL trial data** (complete accelerometer/force data)  
✅ **OpenCap events** (timestamps of each ON/OFF toggle)  
✅ **Steps** (as before)  
✅ **Survey responses** (as before)

**Important**: The OpenCap marks are just timestamps. They don't filter or remove any data. All data from the entire trial is saved.

## Data Structure

Each trial now has an `opencap_events` field containing:

```json
[
  {
    "timestamp": 1234567890,
    "state": "on",
    "relativeTime": 5000
  },
  {
    "timestamp": 1234568890,
    "state": "off",
    "relativeTime": 15000
  }
]
```

- `timestamp`: Absolute timestamp (milliseconds)
- `state`: Either "on" or "off"
- `relativeTime`: Milliseconds since trial start

## Using the Data Later

When you're ready to analyze the data with video, you can:

```python
from app.core.utils.opencap_utils import process_opencap_events, segment_timeseries_data

# Load trial
trial = db.query(Trial).filter(Trial.id == trial_id).first()

# Get raw events
events = trial.opencap_events

# Convert to time segments
segments = process_opencap_events(events)
# Result: [{"start": 5000, "end": 15000, "duration": 10000}, ...]

# Load all trial data
all_data = load_trial_data(trial.raw_data_path)

# Extract just the OpenCap segments if needed
opencap_data = segment_timeseries_data(all_data, segments)
```

## Next Steps

1. **Restart your backend server** (if running)
2. **Refresh your browser**
3. **Try it out!**
   - Start a trial
   - Toggle OpenCap ON/OFF a few times
   - Save the trial
   - Check the database to see the events

## Files Modified

### Frontend

- `frontend/index.html` - Toggle button UI
- `frontend/js/systematic-mode/trial-runner.js` - Event tracking
- `frontend/js/systematic-mode/core.js` - UI handling
- `frontend/js/systematic-mode/survey-manager.js` - Data saving

### Backend

- `database/models.py` - Added `opencap_events` column
- `app/core/models/trial.py` - Pydantic models
- `app/core/services/experiment_service.py` - Data storage
- `app/core/utils/opencap_utils.py` - **NEW** - Analysis utilities

### Database

- Added `opencap_events` JSON column to `trials` table

## Troubleshooting

### Button doesn't appear

- Make sure you're in Grid Search mode
- Start a trial first (button only appears during trials)
- Check browser console for errors

### Data not saving

- Check backend logs for errors
- Verify database migration ran successfully
- Ensure backend server was restarted after migration

### Old trials

- Trials created before this update will have `null` or empty `opencap_events`
- This is normal and doesn't affect those trials

## Questions?

See the complete documentation in `docs/OPENCAP_INTEGRATION.md`
