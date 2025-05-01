import asyncio
from bleak import BleakScanner, BleakClient
import struct

# UUID for the characteristic to which we write and read data
CHARACTERISTIC_UUID = '0000ffe1-0000-1000-8000-00805f9b34fb'

# Number of float values to send
NUM_FLOATS_TO_SEND = 8  # Change this to any number you want

async def run():

    target_device_name_fragment = "HIP_EXO_V2"

    print("Scanning for Bluetooth devices...")
    devices = await BleakScanner.discover()

    target_device_address = None
    for device in devices:
        if device.name and target_device_name_fragment in device.name:
            print(f"Target device found: {device.name} at {device.address}")
            target_device_address = device.address
            break

    if not target_device_address:
        print("\nAvailable Bluetooth devices:")
        device_list = [device for device in devices if device.name]

        if not device_list:
            print("No named Bluetooth devices found. Please check your device.")
            return

        for i, device in enumerate(device_list):
            print(f"{i}: {device.name} ({device.address})")

        while True:
            user_input = input("\nEnter the number of the device you want to connect to (or type 'exit' to quit): ").strip()
            
            if user_input.lower() == "exit":
                print("Exiting program.")
                return
            
            try:
                choice = int(user_input)
                if 0 <= choice < len(device_list):
                    target_device_address = device_list[choice].address
                    break
                else:
                    print("Invalid selection. Try again.")
            except ValueError:
                print("Please enter a valid number or type 'exit' to quit.")

    if target_device_address:
        connected, client = await connect_to_device(target_device_address)
        if connected:            
            await asyncio.gather(
                receive_signal(client),
            )
    
            await disconnect_device(client)
        else:
            print("Failed to connect to the device.")

    else:
        print(f"No device containing '{target_device_name_fragment}' in name found.")

async def connect_to_device(address):
    print("Attempting to connect...")
    client = BleakClient(address, timeout=10.0)
    try:
        connected = await client.connect()
        if connected:
            print(f"Successfully connected to {address}")
            return True, client
        else:
            print(f"Failed to connect to {address}")
            return False, None
    except Exception as e:
        print(f"Failed to connect with error: {e}")
        return False, None

buffer = bytearray()
data_labels = ["force", "roll", 'accX']  # Just modify this list to change fields

HEADER_MARKER = 0xAA
FOOTER_MARKER = 0xBB
FLOAT_SIZE = 4
NUM_FLOATS = len(data_labels)
PAYLOAD_SIZE = NUM_FLOATS * FLOAT_SIZE
PACKET_SIZE = 1 + PAYLOAD_SIZE + 1  # Header + Payload + Footer

def notification_handler(sender, data):
    global buffer
    buffer += data

    while len(buffer) >= PACKET_SIZE:
        # Find the header byte
        start_index = buffer.find(bytes([HEADER_MARKER]))
        if start_index == -1 or start_index + PACKET_SIZE > len(buffer):
            break  # Incomplete packet or no header found

        # Validate footer
        footer_index = start_index + PACKET_SIZE - 1
        if buffer[footer_index] == FOOTER_MARKER:
            try:
                # Extract and unpack the payload
                payload_start = start_index + 1
                payload_end = payload_start + PAYLOAD_SIZE
                payload = buffer[payload_start:payload_end]

                floats = struct.unpack('<' + 'f' * NUM_FLOATS, payload)
                output = ', '.join([f"{label} : {value:.4f}" for label, value in zip(data_labels, floats)])
                print(output)

                # Remove the processed packet from the buffer
                buffer = buffer[start_index + PACKET_SIZE:]
            except struct.error as e:
                print(f"[Unpack Error] {e}")
                buffer = buffer[start_index + 1:]  # Skip one byte and retry
        else:
            buffer = buffer[start_index + 1:]  # Invalid footer, skip and retry

async def receive_signal(client):
    try:
        await client.start_notify(CHARACTERISTIC_UUID, notification_handler)
        await asyncio.Future()  # Keep the coroutine running to continue receiving notifications
    except asyncio.CancelledError:
        print("Stopping receive signal...")
    except Exception as e:
        print(f"Error during communication: {e}")
    finally:
        await client.stop_notify(CHARACTERISTIC_UUID)

async def disconnect_device(client):
    if client.is_connected:
        await client.disconnect()
        print("Disconnected.")

# Run asyncio event loop
loop = asyncio.get_event_loop()
loop.run_until_complete(run())
