# We use this code to read the raw sensor data from the wired sensor.
# The sensor data will be saved in a csv file named "test_wired.csv".
# ctrl + c to stop the program

# import the necessary packages
# pip install pyserial
import serial
import struct

#  COM port, please change it to your own COM port
COM = 'COM6'
# number of sensors, please change it to the number of sensors you are using
num = 9
data = bytearray(4 * (3 * num))

arduino = serial.Serial(port=COM, baudrate=115200, timeout=None)

arduino.flushInput()
result = []
name = ['Time Stamp']
for i in range(num):
    name.append("Sensor " + str(i+1))

import datetime
import pandas as pd
# ctrl + c to stop the program
try:
    while True:
        arduino.flushInput()
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

# ctrl + c to stop the program 
except KeyboardInterrupt:
    print("Output csv")
    test = pd.DataFrame(columns=name, data=result)
    test.to_csv("test_wired.csv")
    print("Exited")