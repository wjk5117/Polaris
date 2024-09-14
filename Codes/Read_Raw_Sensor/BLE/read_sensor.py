# We will use this code to read the raw sensor data from the BLE device
# Before running this code, please change the address to the address of your device get from find_device.py
# For example, the address of a "Bluefruit52" MCU is "D3:A8:5B:55:AF:C5"

import asyncio
import struct
import datetime
import atexit
import numpy as np
from bleak import BleakClient
import pandas as pd
import atexit


# Nordic NUS characteristic for RX, which should be writable`
UART_RX_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
# Nordic NUS characteristic for TX, which should be readable
UART_TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"

#
num = 3
sensors = np.zeros((num, 3))
result = []
name = ['Time Stamp']
for i in range(num):
    name.append("Sensor " + str(i+1))

@atexit.register
def clean():
    print("Output csv")
    test = pd.DataFrame(columns=name, data=result)
    test.to_csv("test_BLE.csv")
    print("Exited")


def notification_handler(sender, data):
    """Simple notification handler which prints the data received."""
    global sensors
    global result
    current = [datetime.datetime.now()]
    for i in range(num):
        sensors[i, 0] = struct.unpack('f', data[12 * i: 12 * i + 4])[0]
        sensors[i, 1] = struct.unpack('f', data[12 * i + 4: 12 * i + 8])[0]
        sensors[i, 2] = struct.unpack('f', data[12 * i + 8: 12 * i + 12])[0]
        print("Sensor " + str(i+1)+": " +
              str(sensors[i, 0]) + ", " + str(sensors[i, 1]) + ", " + str(sensors[i, 2]))
        current.append(
            "("+str(sensors[i, 0]) + ", " + str(sensors[i, 1]) + ", " + str(sensors[i, 2])+")")
    print("############")
    result.append(current)


async def run(address, loop):
    async with BleakClient(address, loop=loop) as client:
        # wait for BLE client to be connected
        x = await client.is_connected()
        print("Connected: {0}".format(x))
        print("Press Enter to quit...")
        # wait for data to be sent from client
        await client.start_notify(UART_TX_UUID, notification_handler)
        while True:
            await asyncio.sleep(0.01)

# Change the address to the address of your device get from find_device.py
address = ("D3:A8:5B:55:AF:C5")
loop = asyncio.get_event_loop()
loop.run_until_complete(run(address, loop))

