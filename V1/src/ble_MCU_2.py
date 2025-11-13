import asyncio
from bleak import BleakScanner, BleakClient
import struct
import csv
from datetime import datetime
import argparse
from typing import Optional

# UUID for the characteristic to which we write and read data
CHARACTERISTIC_UUID = '0000ffe1-0000-1000-8000-00805f9b34fb'

async def record_trial_data(
    output_filename: str,
    device_name_prefix: str = "HIP_EXO",
    device_address: Optional[str] = None,
) -> None:
    """
    Scans for a BLE device, connects, records data until Enter is pressed,
    and saves the data to a CSV file. This function is designed to be
    imported and called by other parts of the application.
    
    Args:
        output_filename: Path to save the recorded CSV data.
        device_name_prefix: Prefix to match device name (default: "HIP_EXO").
                           Matches V2 frontend behavior.
        device_address: Optional BLE device address to connect directly.
                       If provided, skips scanning.
    """
    buffer = bytearray()
    recorded_data = []
    data_labels = ["force", "accX", 'accY']

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
        header = ["relative_time_ms", "force", "accX", "accY"]
        
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

    # If address is provided, use it directly
    if device_address:
        addr = device_address
        print(f"Connecting directly to device: {addr}")
    else:
        # Scan for devices matching the prefix (same as V2 frontend)
        print(f"Scanning for Bluetooth devices with prefix '{device_name_prefix}'...")
        devices = await BleakScanner.discover()
        addr = None
        
        # Find device matching the prefix
        matching_devices = []
        for d in devices:
            if d.name and d.name.startswith(device_name_prefix):
                matching_devices.append((d.name, d.address))
        
        if len(matching_devices) == 0:
            print(f"❌ No device found with prefix '{device_name_prefix}'.")
            print("\nDiscovered devices (showing named devices only):")
            named_devices = [(d.name, d.address) for d in devices if d.name]
            if named_devices:
                for name, address in named_devices[:10]:  # Show first 10
                    print(f"   - {name} [{address}]")
                if len(named_devices) > 10:
                    print(f"   ... and {len(named_devices) - 10} more")
            else:
                print("   (No named devices found)")
            print(f"\n💡 Tip: Use --device-address to connect directly, or run scan_ble_devices.py to see all devices")
            return
        elif len(matching_devices) == 1:
            name, addr = matching_devices[0]
            print(f"✅ Found target: {name} [{addr}]")
        else:
            # Multiple matches - use the first one, but warn user
            name, addr = matching_devices[0]
            print(f"⚠️  Found {len(matching_devices)} matching device(s), using first:")
            for i, (n, a) in enumerate(matching_devices, 1):
                marker = "→" if i == 1 else " "
                print(f"   {marker} {i}. {n} [{a}]")
            print(f"\n✅ Connecting to: {name} [{addr}]")

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
    parser = argparse.ArgumentParser(
        description="BLE data recorder for crutch experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default device name prefix (HIP_EXO)
  python ble_MCU_2.py --filename trial_1.csv

  # Connect to a specific device by address
  python ble_MCU_2.py --device-address AA:BB:CC:DD:EE:FF

  # Use a different device name prefix
  python ble_MCU_2.py --device-prefix MY_DEVICE

  # Scan for all devices first (use scan_ble_devices.py)
  python scan_ble_devices.py
        """
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="recorded_data.csv",
        help="The path to save the recorded CSV data (default: recorded_data.csv)."
    )
    parser.add_argument(
        "--device-prefix",
        type=str,
        default="HIP_EXO",
        help="Device name prefix to search for (default: HIP_EXO, matches V2 frontend)."
    )
    parser.add_argument(
        "--device-address",
        type=str,
        default=None,
        help="BLE device address to connect directly (skips scanning). Format: XX:XX:XX:XX:XX:XX"
    )
    args = parser.parse_args()

    try:
        asyncio.run(record_trial_data(
            args.filename,
            device_name_prefix=args.device_prefix,
            device_address=args.device_address
        ))
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()