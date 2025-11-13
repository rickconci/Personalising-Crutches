"""
EMG Data Parser for Trigno CSV Files

This module provides clean, modular functionality for parsing Trigno CSV files
with proper column mapping and sensor identification using Pydantic for validation.
"""

import re
import pandas as pd
import numpy as np
from io import StringIO
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
from pydantic import BaseModel, Field, validator
import warnings

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


class EMGData(BaseModel):
    """Container for parsed EMG data with Pydantic validation."""
    
    data: pd.DataFrame = Field(..., description="EMG and IMU data with time_s column")
    metadata: Dict[str, str] = Field(default_factory=dict, description="File metadata")
    roster: List[str] = Field(..., description="Sensor roster from file")
    sampling_rate: float = Field(..., gt=0, description="Sampling rate in Hz")
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            pd.DataFrame: lambda v: v.to_dict('records'),
            np.ndarray: lambda v: v.tolist()
        }
    
    @validator('roster')
    def validate_roster(cls, v):
        """Validate that roster has expected number of sensors."""
        if len(v) < 8:
            raise ValueError(f"Expected at least 8 sensors in roster, got {len(v)}")
        return v
    
    @validator('sampling_rate')
    def validate_sampling_rate(cls, v):
        """Validate sampling rate is reasonable."""
        if v <= 0 or v > 10000:
            raise ValueError(f"Sampling rate {v} Hz is not reasonable")
        return v
    
    def get_muscle_data(self, muscle: str) -> Optional[pd.Series]:
        """Get EMG data for a specific muscle."""
        column = f"{muscle}_mV"
        return self.data[column] if column in self.data.columns else None
    
    def get_available_muscles(self) -> List[str]:
        """Get list of available muscle channels."""
        return [col.replace('_mV', '') for col in self.data.columns if col.endswith('_mV')]
    
    def get_trial_duration(self) -> float:
        """Get trial duration in seconds."""
        return float(self.data['time_s'].max() - self.data['time_s'].min())
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the data."""
        return {
            'duration_s': self.get_trial_duration(),
            'sampling_rate_hz': self.sampling_rate,
            'n_samples': len(self.data),
            'available_muscles': self.get_available_muscles(),
            'has_imu': any('ACC' in col or 'GYRO' in col for col in self.data.columns)
        }


class EMGParser:
    """
    Parser for Trigno EMG CSV files.
    
    Handles the complex structure of Trigno CSV exports and provides
    clean access to EMG and IMU data with Pydantic validation.
    """
    
    def __init__(self, imu_rate_hint: float = 370.3704):
        """
        Initialize the EMG parser.
        
        Args:
            imu_rate_hint: Expected IMU sampling rate for tolerance calculation
        """
        self.imu_rate_hint = imu_rate_hint
    
    def parse_file(
        self, 
        path: Union[str, Path], 
        preferred_order: Optional[List[str]] = None
    ) -> EMGData:
        """
        Parse a Trigno CSV file.
        
        Args:
            path: Path to the CSV file
            preferred_order: Optional list of 8 sensor names in preferred order
            
        Returns:
            EMGData object with parsed data and metadata
            
        Raises:
            RuntimeError: If file structure is unexpected
            ValueError: If preferred_order is invalid
        """
        path = Path(path)
        
        # Read and parse the file
        df_multi, metadata, roster = self._parse_csv_structure(path, preferred_order)
        
        # Collapse to single time-based DataFrame
        wide_df = self._collapse_to_single_time(df_multi)
        
        # Calculate sampling rate
        sampling_rate = self._calculate_sampling_rate(wide_df)
        
        # Create and validate EMGData object
        return EMGData(
            data=wide_df,
            metadata=metadata,
            roster=roster,
            sampling_rate=sampling_rate
        )
    
    def _parse_csv_structure(
        self, 
        path: Path, 
        preferred_order: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
        """Parse the CSV file structure and extract data."""
        with path.open('r', encoding='utf-8', errors='ignore') as f:
            lines = [l.rstrip('\n') for l in f]

        # Extract metadata until the roster line
        meta = {}
        i = 0
        while i < len(lines) and not lines[i].lower().startswith('forearm'):
            if ',' in lines[i]:
                k, v = [x.strip() for x in lines[i].split(',', 1)]
                meta[k.rstrip(':')] = v
            i += 1

        roster_line = lines[i]
        modes_line = lines[i+1]
        header_line = lines[i+2]
        freq_line = lines[i+3]
        data_lines = lines[i+4:]

        roster = self._split_roster(roster_line)
        
        if len(roster) < 8:
            raise RuntimeError(f"Unexpected roster: {roster}")

        # Apply preferred order if provided
        if preferred_order is not None:
            roster = self._apply_preferred_order(roster, preferred_order)

        # Read the raw table
        raw_df = pd.read_csv(StringIO('\n'.join([header_line] + data_lines)), engine='python')
        raw_df.columns = [c.strip() for c in raw_df.columns]

        # Build column specifications
        spec = self._build_column_specs(raw_df, roster)
        
        # Create MultiIndex DataFrame
        arrays, data = self._build_multichannel_data(raw_df, spec)
        multi_cols = pd.MultiIndex.from_tuples(arrays, names=['sensor', 'channel', 'field'])
        df_multi = pd.DataFrame(data, columns=multi_cols)

        return df_multi, meta, roster
    
    def _split_roster(self, line: str) -> List[str]:
        """Extract sensor names from roster line."""
        parts = [p.strip() for p in line.split(',')]
        names = []
        for p in parts:
            if not p or p == '':
                continue
            names.append(re.sub(r'\s*\(\d+\)\s*', '', p))
        return names
    
    def _apply_preferred_order(self, roster: List[str], preferred_order: List[str]) -> List[str]:
        """Apply preferred sensor order."""
        emg_names_preferred = [n for n in preferred_order if n.upper() != 'IMU'][:7]
        if len(emg_names_preferred) != 7:
            raise ValueError("preferred_order must specify 7 EMG names plus 'IMU'.")
        return emg_names_preferred + ['IMU']
    
    def _build_column_specs(self, raw_df: pd.DataFrame, roster: List[str]) -> List[Tuple]:
        """Build column specifications for EMG and IMU data."""
        spec = []
        
        # EMG sensors
        emg_time_cols = [col for col in raw_df.columns if 'EMG 1 Time Series (s)' in col]
        emg_value_cols = [col for col in raw_df.columns if 'EMG 1 (mV)' in col]
        
        if len(emg_time_cols) != 7 or len(emg_value_cols) != 7:
            raise RuntimeError(f"Expected 7 EMG time and value columns, got {len(emg_time_cols)} and {len(emg_value_cols)}")
        
        for k in range(7):
            sensor = roster[k]
            t_col = emg_time_cols[k]
            v_col = emg_value_cols[k]
            spec.append((sensor, 'EMG', t_col, v_col, 'mV'))

        # IMU sensors
        imu_pairs = [
            ('ACC X', r'ACC X Time Series \(s\)', r'ACC X \(G\)', 'G'),
            ('ACC Y', r'ACC Y Time Series \(s\)', r'ACC Y \(G\)', 'G'),
            ('ACC Z', r'ACC Z Time Series \(s\)', r'ACC Z \(G\)', 'G'),
            ('GYRO X', r'GYRO X Time Series \(s\)', r'GYRO X \(deg/s\)', 'deg/s'),
            ('GYRO Y', r'GYRO Y Time Series \(s\)', r'GYRO Y \(deg/s\)', 'deg/s'),
            ('GYRO Z', r'GYRO Z Time Series \(s\)', r'GYRO Z \(deg/s\)', 'deg/s'),
        ]
        
        for subch, t_pat, v_pat, unit in imu_pairs:
            t_col = next((c for c in raw_df.columns if re.fullmatch(t_pat, c, re.I)), None)
            v_col = next((c for c in raw_df.columns if re.fullmatch(v_pat, c, re.I)), None)
            if t_col is None or v_col is None:
                raise RuntimeError(f"Couldn't find IMU columns for {subch}.")
            spec.append(('IMU', subch, t_col, v_col, unit))
        
        return spec
    
    def _build_multichannel_data(self, raw_df: pd.DataFrame, spec: List[Tuple]) -> Tuple[List, Dict]:
        """Build multichannel data arrays."""
        arrays = []
        data = {}
        
        for sensor, sub, t_col, v_col, unit in spec:
            time_key = (sensor, sub, 'time_s')
            value_key = (sensor, sub, f'value_{unit}')
            
            # Handle duplicate column names
            if isinstance(raw_df[t_col], pd.DataFrame):
                time_series = pd.to_numeric(raw_df[t_col].iloc[:, 0], errors='coerce')
            else:
                time_series = pd.to_numeric(raw_df[t_col], errors='coerce')
                
            if isinstance(raw_df[v_col], pd.DataFrame):
                value_series = pd.to_numeric(raw_df[v_col].iloc[:, 0], errors='coerce')
            else:
                value_series = pd.to_numeric(raw_df[v_col], errors='coerce')
            
            arrays.append(time_key)
            data[time_key] = time_series
            arrays.append(value_key)
            data[value_key] = value_series
        
        return arrays, data
    
    def _collapse_to_single_time(self, df_multi: pd.DataFrame) -> pd.DataFrame:
        """Collapse MultiIndex DataFrame to single time-based DataFrame."""
        # Choose EMG master time
        emg_time_cols = [c for c in df_multi.columns if c[1] == 'EMG' and c[2] == 'time_s']
        if not emg_time_cols:
            raise ValueError("No EMG time columns found.")
        
        master_time_col = emg_time_cols[0]
        base = pd.DataFrame({'time_s': df_multi[master_time_col].astype('float64')})
        base = base.dropna().drop_duplicates(subset=['time_s']).sort_values('time_s').reset_index(drop=True)

        # Add EMG values
        emg_val_cols = [c for c in df_multi.columns if c[1] == 'EMG' and c[2].startswith('value_')]
        for c in emg_val_cols:
            muscle = c[0]
            base[muscle + '_mV'] = pd.to_numeric(df_multi[c], errors='coerce').values

        # Add IMU data if present
        imu_time_cols = [c for c in df_multi.columns if c[0] == 'IMU' and c[2] == 'time_s']
        if imu_time_cols:
            imu_time = df_multi[imu_time_cols[0]].astype('float64')
            imu = pd.DataFrame({'imu_time_s': imu_time})
            
            imu_val_cols = [c for c in df_multi.columns if c[0] == 'IMU' and c[2].startswith('value_')]
            for c in imu_val_cols:
                sub = c[1]
                unit = c[2].split('_', 1)[1]
                colname = f"{sub.replace(' ', '_')}_{unit}"
                imu[colname] = pd.to_numeric(df_multi[c], errors='coerce')

            imu = imu.dropna(subset=['imu_time_s']).drop_duplicates(subset=['imu_time_s']).sort_values('imu_time_s')

            # Merge IMU onto EMG time
            tol = 0.5 / self.imu_rate_hint
            base = pd.merge_asof(
                base.sort_values('time_s'),
                imu.sort_values('imu_time_s'),
                left_on='time_s',
                right_on='imu_time_s',
                direction='nearest',
                tolerance=tol
            ).drop(columns=['imu_time_s'])

        return base    
    def _calculate_sampling_rate(self, df: pd.DataFrame) -> float:
        """Calculate sampling rate from time data."""
        if 'time_s' not in df.columns or len(df) < 2:
            return 2000.0  # Default fallback
        
        time_diffs = np.diff(df['time_s'].values)
        return float(1.0 / np.median(time_diffs))
