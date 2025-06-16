import asyncio
from bleak import BleakScanner, BleakClient
import struct
import csv
from datetime import datetime
import argparse

# UUID for the characteristic to which we write and read data
CHARACTERISTIC_UUID = '0000ffe1-0000-1000-8000-00805f9b34fb'

async def record_trial_data(output_filename):
    """
    Scans for a BLE device, connects, records data until Enter is pressed,
    and saves the data to a CSV file. This function is designed to be
    imported and called by other parts of the application.
    """
    buffer = bytearray()
    recorded_data = []
    data_labels = ["force", "roll", 'accX']

    # --- Packet Structure ---
    HEADER_MARKER = 0xAA
    FOOTER_MARKER = 0xBB
    FLOAT_SIZE = 4
    NUM_FLOATS = len(data_labels)
    PAYLOAD_SIZE = NUM_FLOATS * FLOAT_SIZE
    PACKET_SIZE = 1 + PAYLOAD_SIZE + 1

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

    async def wait_for_enter(stop_event):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, input, "Press ENTER to stop recording and save CSV...\n")
        stop_event.set()

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
        header = ["acc_x_time", "force", "roll", "acc_x_data"]
        
        # Convert timestamps to relative time with 5ms increments
        relative_data = []
        for i, row in enumerate(recorded_data):
            relative_time = i * 5  # 5ms increments
            relative_data.append([relative_time, *row[1:]])  # Keep all data except timestamp

        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(relative_data)
        print(f"Saved {len(recorded_data)} rows into {filename}")

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
            wait_for_enter(stop_event)
        )
    
    await save_csv(output_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BLE data recorder for crutch experiments.")
    parser.add_argument(
        "--filename",
        type=str,
        default="recorded_data.csv",
        help="The path to save the recorded CSV data."
    )
    args = parser.parse_args()

    try:
        asyncio.run(record_trial_data(args.filename))
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")