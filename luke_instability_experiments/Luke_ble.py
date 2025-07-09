#!/usr/bin/env python3
"""
Luke_ble.py - Bluetooth Data Collection for Instability Experiments

USAGE:
    Single trial mode:    python Luke_ble.py
    Continuous mode:      python Luke_ble.py --continuous

DESCRIPTION:
    This script collects force, accX, and accY data from the MCU device.
    
    Single trial mode:
    - Records one trial and exits
    - Auto-generates filename (recorded_data_X.csv)
    - Good for testing or single measurements
    
    Continuous mode:
    - Records 3 trials for one participant
    - Uses proper naming (Name1_data.csv, Name2_data.csv, Name3_data.csv)
    - Asks for participant name once, geometry for each trial
    - Perfect for experimental data collection
    
    Valid geometries: 10.5, 18.2, 14
    
    Press ENTER to start recording, ENTER to stop recording.
"""

import asyncio
from bleak import BleakScanner, BleakClient
import struct
import csv
from datetime import datetime
import argparse
import os
import glob
import subprocess

CHARACTERISTIC_UUID = '0000ffe1-0000-1000-8000-00805f9b34fb'

def get_next_filename():
    """Get the next available filename with successive numbering."""
    # Check for existing files in current folder
    pattern = "recorded_data_*.csv"
    existing_files = glob.glob(pattern)
    
    if not existing_files:
        return "recorded_data_1.csv"
    
    # Extract numbers from existing filenames
    numbers = []
    for file in existing_files:
        try:
            # Extract number from filename like "recorded_data_1.csv"
            number = int(file.split("_")[-1].replace(".csv", ""))
            numbers.append(number)
        except ValueError:
            continue
    
    if not numbers:
        return "recorded_data_1.csv"
    
    # Get the next number
    next_number = max(numbers) + 1
    return f"recorded_data_{next_number}.csv"

async def record_trial_data(output_filename):
    """
    Connects to BLE device, receives force and IMU data (accX, accY), and saves to CSV.
    """
    buffer = bytearray()
    recorded_data = []
    data_labels = ["force", "accX", "accY"]

    # --- Packet Structure ---
    HEADER_MARKER = 0xAA
    FOOTER_MARKER = 0xBB
    FLOAT_SIZE = 4
    NUM_FLOATS = len(data_labels)
    PAYLOAD_SIZE = NUM_FLOATS * FLOAT_SIZE
    PACKET_SIZE = 1 + PAYLOAD_SIZE + 1  # header + payload + footer

    def notification_handler(sender, data):
        nonlocal buffer, recorded_data
        buffer += data

        while len(buffer) >= PACKET_SIZE:
            start_index = buffer.find(bytes([HEADER_MARKER]))
            if start_index == -1 or start_index + PACKET_SIZE > len(buffer):
                break

            footer_index = start_index + PACKET_SIZE - 1
            if buffer[footer_index] == FOOTER_MARKER:
                try:
                    payload = buffer[start_index + 1:footer_index]
                    floats = struct.unpack('<' + 'f' * NUM_FLOATS, payload)
                    timestamp = datetime.now().isoformat()
                    recorded_data.append([timestamp, *floats])
                    output = ', '.join([f"{label}: {value:.4f}" for label, value in zip(data_labels, floats)])
                    print(output)
                    buffer = buffer[start_index + PACKET_SIZE:]
                except struct.error as e:
                    print(f"[Unpack Error] {e}")
                    buffer = buffer[start_index + 1:]
            else:
                buffer = buffer[start_index + 1:]

    async def wait_for_spacebar(stop_event):
        loop = asyncio.get_running_loop()
        print("Press SPACEBAR to stop recording and save CSV...")
        while True:
            try:
                # Use a non-blocking input method
                if os.name == 'nt':  # Windows
                    import msvcrt
                else:  # Unix/Linux/Mac
                    import sys, tty, termios
                if os.name == 'nt':  # Windows
                    if msvcrt.kbhit():
                        key = msvcrt.getch()
                        if key == b' ':
                            print("\nSpacebar pressed - stopping recording...")
                            stop_event.set()
                            break
                else:  # Unix/Linux/Mac
                    import sys, tty, termios
                    fd = sys.stdin.fileno()
                    old_settings = termios.tcgetattr(fd)
                    try:
                        tty.setraw(sys.stdin.fileno())
                        ch = sys.stdin.read(1)
                        if ch == ' ':
                            print("\nSpacebar pressed - stopping recording...")
                            stop_event.set()
                            break
                    finally:
                        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            except:
                # Fallback to Enter key if spacebar detection fails
                await loop.run_in_executor(None, input, "Press ENTER to stop recording...\n")
                stop_event.set()
                break

    async def receive_signal(client, stop_event):
        try:
            await client.start_notify(CHARACTERISTIC_UUID, notification_handler)
            await stop_event.wait()
        except Exception as e:
            print(f"Error during communication: {e}")
        finally:
            if client.is_connected:
                await client.stop_notify(CHARACTERISTIC_UUID)

    async def save_csv(filename):
        header = ["relative_time_ms", "force", "accX", "accY"]
        relative_data = []
        for i, row in enumerate(recorded_data):
            relative_time = i * 5  # assume 5ms per sample
            relative_data.append([relative_time, *row[1:]])  # drop timestamp

        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(relative_data)
        print(f"Saved {len(recorded_data)} rows into {filename}")

    # --- Scan and Connect ---
    target_fragment = "HIP_EXO_V2"
    print("Scanning for Bluetooth devices...")
    devices = await BleakScanner.discover()
    addr = None
    for d in devices:
        if d.name and target_fragment in d.name:
            addr = d.address
            print(f"Found target: {d.name} [{addr}]")
            break
    if not addr:
        print("...no matching device. Exiting.")
        return

    async with BleakClient(addr, timeout=10.0) as client:
        if not client.is_connected:
            print("Failed to connect.")
            return
        print("Connected!")

        stop_event = asyncio.Event()
        await asyncio.gather(
            receive_signal(client, stop_event),
            wait_for_spacebar(stop_event)
        )

    await save_csv(output_filename)

async def continuous_data_collection():
    """
    Continuous data collection mode - allows multiple trials without restarting.
    """
    print("=== Continuous BLE Data Collection Mode ===")
    print("This mode allows you to collect multiple trials without restarting the script.")
    print("Press 'q' to quit, or just press Enter to continue to next trial.")
    
    # Get participant name once at the start
    participant_name = input("Enter participant name: ").strip()
    if participant_name.lower() == 'q':
        return
    
    # Track trial number for this participant
    trial_number = 1
    all_geometries = ["10.5", "18.2", "14"]
    used_geometries = []
    
    while True:
        print(f"\n--- {participant_name} Trial {trial_number} ---")
        
        # Get available geometries (not yet used)
        available_geometries = [g for g in all_geometries if g not in used_geometries]
        
        if len(available_geometries) == 0:
            print("All geometries have been tested!")
            break
        
        # Get geometry for this trial
        while True:
            print(f"Available geometries: {', '.join(available_geometries)}")
            geometry = input("Enter geometry being tested: ").strip()
            if geometry in available_geometries:
                used_geometries.append(geometry)
                break
            else:
                print(f"Invalid geometry. Please choose from: {', '.join(available_geometries)}")
        
        # Generate filename with proper naming convention
        filename = f"{participant_name}{trial_number}_data.csv"
        print(f"Will save to: {filename}")
        
        # Confirm before starting
        confirm = input("Press Enter to start recording, or 's' to skip this trial: ").strip()
        if confirm.lower() == 's':
            continue
        
        try:
            await record_trial_data(filename)
            print(f"Trial completed! Data saved to {filename}")
            
            # Ask if user wants to continue
            if trial_number >= 3:
                print(f"\n=== All 3 trials completed for {participant_name}! ===")
                print(f"Geometries tested: {', '.join(used_geometries)}")
                print(f"\n=== Next Steps ===")
                print(f"To process all trials and create visualizations, run:")
                print(f"   python process_trials.py {participant_name}")
                print(f"\nThis will:")
                print(f"- Process all 3 trials")
                print(f"- Collect your rankings")
                print(f"- Create comparison plots")
                print(f"- Save analysis results")
                break
            else:
                continue_trial = input("Press Enter for next trial, or 'q' to quit: ").strip()
                if continue_trial.lower() == 'q':
                    break
                trial_number += 1
                
        except Exception as e:
            print(f"Error during trial: {e}")
            retry = input("Press Enter to retry, or 'q' to quit: ").strip()
            if retry.lower() == 'q':
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BLE data recorder for force + accX + accY.")
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="The path to save the recorded CSV data. If not specified, will auto-number."
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Enable continuous data collection mode for multiple trials"
    )
    args = parser.parse_args()

    try:
        if args.continuous:
            asyncio.run(continuous_data_collection())
        else:
            # Auto-generate filename if not provided
            if args.filename is None:
                args.filename = get_next_filename()
                print(f"Will save to: {args.filename}")

            asyncio.run(record_trial_data(args.filename))
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
