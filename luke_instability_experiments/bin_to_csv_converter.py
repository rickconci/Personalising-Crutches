#!/usr/bin/env python3
"""
BIN to CSV Converter
Converts BIN files to CSV format compatible with data_analysis.py
Similar to how it is done in data_processing_imu.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import struct

def count_elements_between_first_pairs(data: np.ndarray, pair1: bytes, pair2: bytes) -> int:
    """
    Find the first occurrence of pair1 and the first occurrence of pair2 after it,
    and return the number of 4-byte elements between them.
    """
    # Find start of pair1
    start_idx = -1
    for i in range(len(data) - 1):
        if data[i] == pair1[0] and data[i+1] == pair1[1]:
            start_idx = i + 1
            break
    if start_idx < 0:
        raise ValueError("Start pair not found")

    # Find start of pair2 after start_idx
    end_idx = -1
    for j in range(start_idx + 1, len(data) - 1):
        if data[j] == pair2[0] and data[j+1] == pair2[1]:
            end_idx = j
            break
    if end_idx < 0 or end_idx <= start_idx:
        raise ValueError("End pair not found or occurs before start pair")

    byte_count = end_idx - start_idx - 1
    if byte_count % 4 != 0:
        raise ValueError("Data between markers is not a multiple of 4 bytes")

    return byte_count // 4

def check_data_packet(data: np.ndarray, sample_idx: int, packet_size: int):
    """
    Raise if packet would overrun the buffer.
    """
    if (sample_idx + 1) * packet_size > len(data):
        raise ValueError(f"Data packet {sample_idx+1} is invalid (truncation)")

def convert_bin_to_csv(bin_file_path, csv_file_path=None):
    """Convert BIN file to CSV format using the correct parsing method."""
    try:
        # --- load raw bytes
        with open(bin_file_path, 'rb') as f:
            raw = f.read()
        data_all = np.frombuffer(raw, dtype=np.uint8)

        # --- infer number of float32 fields by scanning markers
        #    Marker pairs [170,170] and [187,187] correspond to 0xAA 0xAA and 0xBB 0xBB
        num_fields = count_elements_between_first_pairs(data_all, b'\xAA\xAA', b'\xBB\xBB')

        # --- every field is single precision float (4 bytes)
        field_bytes = [4] * num_fields
        # --- offsets: first 4 padding bytes, then fields start at byte-offsets 3, 3+4, 3+8, …
        packet_size = 4 + sum(field_bytes)
        field_offsets = [2 + sum(field_bytes[:i]) for i in range(num_fields)]

        # --- prepare output containers
        data = {f"data{i+1}": [] for i in range(num_fields)}

        # --- loop over samples
        n_samples = len(data_all) // packet_size
        for i in range(n_samples):
            check_data_packet(data_all, i, packet_size)
            base = i * packet_size
            for j in range(num_fields):
                start = base + field_offsets[j]
                chunk = data_all[start : start + field_bytes[j]].tobytes()
                # little-endian float32 unpack
                value = struct.unpack('<f', chunk)[0]
                data[f"data{j+1}"].append(value)

        # --- convert lists to NumPy arrays
        for key in data:
            data[key] = np.array(data[key], dtype=np.float32)

        # --- build DataFrame
        df = pd.DataFrame(data)

        # ─── Human‑friendly column names ───────────────────────────────
        rename_map = {
            "data1": "acc_x_time",
            "data2": "data_2",
            "data3": "roll",
            "data4": "pitch",
            "data5": "yaw",
            "data6": "acc_x_data",
            "data7": "acc_y_data",
            "data8": "acc_z_data",
            "data9": "gyro_x_data",
            "data10": "gyro_y_data",
            "data11": "gyro_z_data",
            "data12": "force",
        }
        # Only rename columns that actually exist in the current DataFrame
        df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns},
                  inplace=True)

        # Add remaining columns with zeros if they don't exist
        for i in range(13, 24):
            col_name = f'data{i}'
            if col_name not in df.columns:
                df[col_name] = 0.0

        # Save to CSV
        if csv_file_path is None:
            csv_file_path = str(bin_file_path).replace('.BIN', '.csv').replace('.bin', '.csv')
        
        df.to_csv(csv_file_path, index=False)
        print(f"Converted {bin_file_path} to {csv_file_path}")
        print(f"  Samples: {len(df)}")
        print(f"  Duration: {len(df) * 0.01:.1f} seconds")
        print(f"  Fields: {num_fields}")
        
        return csv_file_path
        
    except Exception as e:
        print(f"Error converting {bin_file_path}: {e}")
        return None

def main():
    """Convert all BIN files in LukeCorrelation directory."""
    bin_dir = Path("LukeCorrelation")
    
    if not bin_dir.exists():
        print("LukeCorrelation directory not found")
        return
    
    bin_files = list(bin_dir.glob("*.BIN")) + list(bin_dir.glob("*.bin"))
    
    if not bin_files:
        print("No BIN files found in LukeCorrelation")
        return
    
    print(f"Found {len(bin_files)} BIN files to convert")
    
    for bin_file in bin_files:
        convert_bin_to_csv(bin_file)
    
    print("Conversion complete!")

if __name__ == "__main__":
    main() 