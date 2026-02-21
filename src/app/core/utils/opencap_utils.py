"""
OpenCap utilities for processing video capture events and segmenting data.

This module provides functions to convert OpenCap toggle events into
time-based segments for video analysis and data extraction.
"""

from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def process_opencap_events(events: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Process OpenCap toggle events into time-based segments.
    
    Converts a list of ON/OFF toggle events into segments with start/end times.
    Each segment represents a period when OpenCap recording was active (ON state).
    
    Args:
        events: List of OpenCap events with format:
            [{
                "timestamp": absolute_timestamp_ms,
                "state": "on" | "off",
                "relativeTime": ms_since_trial_start
            }, ...]
    
    Returns:
        List of segments with format:
            [{
                "start": start_time_ms,
                "end": end_time_ms,
                "duration": duration_ms
            }, ...]
    
    Example:
        >>> events = [
        ...     {"timestamp": 1000, "state": "on", "relativeTime": 5000},
        ...     {"timestamp": 2000, "state": "off", "relativeTime": 15000},
        ...     {"timestamp": 3000, "state": "on", "relativeTime": 30000},
        ...     {"timestamp": 4000, "state": "off", "relativeTime": 45000}
        ... ]
        >>> process_opencap_events(events)
        [
            {"start": 5000, "end": 15000, "duration": 10000},
            {"start": 30000, "end": 45000, "duration": 15000}
        ]
    """
    if not events or len(events) == 0:
        logger.info("No OpenCap events to process")
        return []
    
    segments = []
    current_segment_start = None
    
    for event in events:
        state = event.get("state", "").lower()
        relative_time = event.get("relativeTime")
        
        if relative_time is None:
            logger.warning(f"Event missing relativeTime: {event}")
            continue
        
        if state == "on":
            if current_segment_start is not None:
                logger.warning(
                    f"Found 'on' event at {relative_time}ms without closing previous segment "
                    f"that started at {current_segment_start}ms. Ignoring previous segment."
                )
            current_segment_start = relative_time
            
        elif state == "off":
            if current_segment_start is None:
                logger.warning(
                    f"Found 'off' event at {relative_time}ms without corresponding 'on' event. "
                    "Ignoring this event."
                )
                continue
            
            # Create segment
            segment = {
                "start": current_segment_start,
                "end": relative_time,
                "duration": relative_time - current_segment_start
            }
            segments.append(segment)
            logger.info(
                f"Created OpenCap segment: {segment['start']}ms - {segment['end']}ms "
                f"(duration: {segment['duration']}ms)"
            )
            current_segment_start = None
        
        else:
            logger.warning(f"Unknown OpenCap event state: '{state}'")
    
    # Handle unclosed segment (ON event without OFF)
    if current_segment_start is not None:
        logger.warning(
            f"OpenCap segment started at {current_segment_start}ms was never closed. "
            "This segment will be ignored."
        )
    
    logger.info(f"Processed {len(events)} OpenCap events into {len(segments)} segments")
    return segments


def segment_timeseries_data(
    data: List[Dict[str, Any]], 
    segments: List[Dict[str, Any]],
    time_key: str = "acc_x_time"
) -> List[Dict[str, Any]]:
    """
    Extract time-series data points that fall within OpenCap segments.
    
    Args:
        data: List of time-series data points (e.g., accelerometer readings)
        segments: List of OpenCap segments from process_opencap_events()
        time_key: Key in data dict that contains the timestamp (default: "acc_x_time")
    
    Returns:
        List of data points that fall within any of the OpenCap segments.
        Each point includes an additional "segment_index" field indicating which segment it belongs to.
    
    Example:
        >>> data = [
        ...     {"acc_x_time": 1000, "force": 10},
        ...     {"acc_x_time": 6000, "force": 20},
        ...     {"acc_x_time": 12000, "force": 30},
        ...     {"acc_x_time": 35000, "force": 40},
        ... ]
        >>> segments = [
        ...     {"start": 5000, "end": 15000, "duration": 10000},
        ...     {"start": 30000, "end": 45000, "duration": 15000}
        ... ]
        >>> segment_timeseries_data(data, segments)
        [
            {"acc_x_time": 6000, "force": 20, "segment_index": 0},
            {"acc_x_time": 12000, "force": 30, "segment_index": 0},
            {"acc_x_time": 35000, "force": 40, "segment_index": 1}
        ]
    """
    if not segments or len(segments) == 0:
        logger.info("No segments provided, returning empty list")
        return []
    
    if not data or len(data) == 0:
        logger.info("No data provided, returning empty list")
        return []
    
    segmented_data = []
    
    for segment_idx, segment in enumerate(segments):
        start_time = segment["start"]
        end_time = segment["end"]
        
        # Extract data points within this segment
        segment_points = [
            {**point, "segment_index": segment_idx}
            for point in data
            if start_time <= point.get(time_key, -1) <= end_time
        ]
        
        segmented_data.extend(segment_points)
        logger.info(
            f"Segment {segment_idx}: Found {len(segment_points)} data points "
            f"between {start_time}ms and {end_time}ms"
        )
    
    logger.info(
        f"Extracted {len(segmented_data)} total data points from "
        f"{len(data)} original points across {len(segments)} segments"
    )
    return segmented_data


def get_segment_statistics(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate statistics about OpenCap segments.
    
    Args:
        segments: List of OpenCap segments
    
    Returns:
        Dictionary with segment statistics:
        {
            "count": total number of segments,
            "total_duration": sum of all segment durations (ms),
            "average_duration": average segment duration (ms),
            "min_duration": shortest segment duration (ms),
            "max_duration": longest segment duration (ms)
        }
    """
    if not segments or len(segments) == 0:
        return {
            "count": 0,
            "total_duration": 0,
            "average_duration": 0,
            "min_duration": 0,
            "max_duration": 0
        }
    
    durations = [seg["duration"] for seg in segments]
    
    return {
        "count": len(segments),
        "total_duration": sum(durations),
        "average_duration": sum(durations) / len(durations),
        "min_duration": min(durations),
        "max_duration": max(durations)
    }

