#!/usr/bin/env python3
"""
BLE Device Scanner - Diagnostic tool to list all available Bluetooth Low Energy devices.

This script helps identify available BLE devices and their names/addresses,
which is useful for debugging connection issues with ble_MCU_2.py.
"""

import asyncio
from bleak import BleakScanner
from typing import List
import argparse


async def scan_devices(timeout: float = 10.0) -> List[dict]:
    """
    Scan for BLE devices and return their information.
    
    Args:
        timeout: Scanning timeout in seconds.
        
    Returns:
        List of dictionaries containing device information.
    """
    print(f"Scanning for BLE devices (timeout: {timeout}s)...")
    print("Make sure your device is powered on and advertising.\n")
    
    devices = await BleakScanner.discover(timeout=timeout)
    
    device_info = []
    for device in devices:
        info = {
            "name": device.name or "<Unknown>",
            "address": device.address,
            "rssi": getattr(device, "rssi", None),
            "metadata": device.metadata if hasattr(device, "metadata") else {},
        }
        device_info.append(info)
    
    return device_info


def print_device_list(devices: List[dict], show_all: bool = False) -> None:
    """
    Print formatted list of discovered devices.
    
    Args:
        devices: List of device information dictionaries.
        show_all: If True, show all devices. If False, filter out common system devices.
    """
    if not devices:
        print("❌ No BLE devices found.")
        print("\nTroubleshooting tips:")
        print("  1. Make sure Bluetooth is enabled on your computer")
        print("  2. Make sure your target device is powered on")
        print("  3. Make sure your target device is in advertising mode")
        print("  4. Try running with --timeout 20 to scan longer")
        return
    
    print(f"✅ Found {len(devices)} BLE device(s):\n")
    print("-" * 80)
    print(f"{'Name':<40} {'Address':<20} {'RSSI':<10}")
    print("-" * 80)
    
    for i, device in enumerate(devices, 1):
        name = device["name"]
        address = device["address"]
        rssi = device["rssi"] if device["rssi"] is not None else "N/A"
        
        # Highlight devices that might match HIP_EXO_V2
        if "HIP" in name.upper() or "EXO" in name.upper():
            marker = "🎯"
        else:
            marker = "  "
        
        print(f"{marker} {i}. {name:<38} {address:<20} {rssi}")
    
    print("-" * 80)
    
    # Check for potential matches
    potential_matches = [
        d for d in devices
        if d["name"] and ("HIP" in d["name"].upper() or "EXO" in d["name"].upper())
    ]
    
    if potential_matches:
        print(f"\n🎯 Found {len(potential_matches)} potential match(es):")
        for device in potential_matches:
            print(f"   - {device['name']} [{device['address']}]")
    else:
        print("\n⚠️  No devices found containing 'HIP' or 'EXO' in the name.")
        print("   The target device name should contain 'HIP_EXO_V2'")
        print("   Check if:")
        print("   - The device name is different than expected")
        print("   - The device needs to be configured/powered on")


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Scan for BLE devices to help debug connection issues."
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Scanning timeout in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show all devices including system devices",
    )
    args = parser.parse_args()
    
    try:
        devices = await scan_devices(timeout=args.timeout)
        print_device_list(devices, show_all=args.all)
    except KeyboardInterrupt:
        print("\n\nScan interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during scan: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())



