import serial
import struct

COM = 'COM7'
num = 9
data = bytearray(4 * (3 * num+1))

arduino = serial.Serial(port=COM, baudrate=115200, timeout=None)

arduino.flushInput()
result = []
name = ['Time Stamp', 'Sensor 1', 'Sensor 2', 'Sensor 3', 'Sensor 4', 'Sensor 5', 'Sensor 6', 'Sensor 7', 'Sensor 8', 'Sensor 9']

import datetime
import pandas as pd
# ctrl + c to stop the program
try:
    while True:
        arduino.readinto(data)
        # print(data)
        current = [datetime.datetime.now()]
        for i in range(num):
            x, = struct.unpack('f', data[(i * 12):(4 + i * 12)])
            y, = struct.unpack('f', data[(i * 12 + 4):(8 + i * 12)])
            z, = struct.unpack('f', data[(i * 12 + 8):(12 + i * 12)])
            # 保留四位小数
            x = round(x, 4)
            y = round(y, 4)
            z = round(z, 4)
            current.append(
                "("+str(x) + ", " + str(y) + ", " + str(z)+")")
            print("Sensor " + str(i+1) + ": ", x, y, z)
        result.append(current)
        
except KeyboardInterrupt:
    print("Output csv")
    test = pd.DataFrame(columns=name, data=result)
    test.to_csv("test_wired.csv")
    print("Exited")