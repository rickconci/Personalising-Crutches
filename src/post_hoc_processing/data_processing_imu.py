import numpy as np
import struct
import pandas as pd
import os
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


def data_processing_bin(file_name: str) -> dict:
    """
    Read the binary file and unpack samples of (num_fields) float32 fields,
    preceded by 4 bytes of padding per sample.
    Returns a dict mapping 'data1', 'data2', … → numpy arrays of shape (n_samples,).
    """
    # --- load raw bytes
    with open(file_name, 'rb') as f:
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

    return data


def data_processing_bin(file_name: str, save_npz: bool = True, save_df: bool = False) -> dict:
    """
    Read the binary file, unpack float32 fields, and optionally save:
      - as an .npz archive (default)
      - as a DataFrame (if save_df=True)
    Returns:
      dict of numpy arrays, e.g. {'data1': array([...]), ...}
    """
    # --- load raw bytes
    with open(file_name, 'rb') as f:
        raw = f.read()
    data_all = np.frombuffer(raw, dtype=np.uint8)

    # --- infer number of float32 fields
    num_fields = count_elements_between_first_pairs(data_all, b'\xAA\xAA', b'\xBB\xBB')
    field_bytes = [4] * num_fields
    packet_size = 4 + sum(field_bytes)
    field_offsets = [2 + sum(field_bytes[:i]) for i in range(num_fields)]

    # --- unpack
    data = {f"data{i+1}": [] for i in range(num_fields)}
    n_samples = len(data_all) // packet_size
    for i in range(n_samples):
        check_data_packet(data_all, i, packet_size)
        base = i * packet_size
        for j in range(num_fields):
            start = base + field_offsets[j]
            chunk = data_all[start:start+4].tobytes()
            value = struct.unpack('<f', chunk)[0]
            data[f"data{j+1}"].append(value)

    # --- to numpy arrays
    for k in data:
        data[k] = np.array(data[k], dtype=np.float32)

    # --- save to .npz
    if save_npz:
        npz_path = os.path.splitext(file_name)[0] + '.npz'
        np.savez(npz_path, **data)
        print(f"Saved data to {npz_path}")

    # --- build & save DataFrame
    if save_df:
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

        # ─── Save files ────────────────────────────────────────────────
        base = os.path.splitext(file_name)[0]
        parquet_path = base + '.parquet'
        csv_path     = base + '.csv'

        df.to_parquet(parquet_path)
        print(f"Saved DataFrame to {parquet_path} (Parquet)")

        df.to_csv(csv_path, index=False)
        print(f"Saved DataFrame to {csv_path} (CSV)")

    return data

if __name__ == "__main__":

    data_path = '/Users/riccardoconci/Library/Mobile Documents/com~apple~CloudDocs/HQ_2024/Projects/2024_Harvard_AIM/Research/OPMO/Personalising-Crutches/2025.06.18/test2.BIN'
    data = data_processing_bin(data_path, save_npz=True, save_df=True)
    print(data)