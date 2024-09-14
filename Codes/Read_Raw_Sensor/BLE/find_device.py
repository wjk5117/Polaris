# This code is used to find the address of the "Bluefruit52" MCU
# For example, the address of the "Bluefruit52" MCU is "D3:A8:5B:55:AF:C5"
# Run this code to find the address of the "Bluefruit52" MCU before running the read_sensor.py code
import asyncio
from bleak import discover

async def run():
    devices = await discover()
    for d in devices:
        # only find the device with the name "Bluefruit52"
        if d.name == "Bluefruit52":
            print(d)

loop = asyncio.get_event_loop()
loop.run_until_complete(run())
