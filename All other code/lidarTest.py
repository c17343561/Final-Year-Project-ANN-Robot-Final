#!/usr/bin/env python

#To find what usb port name is enter ls -l /dev | grep ttyUSB

import time
import serial

ser = serial.Serial(
        port='/dev/ttyUSB0',
        baudrate = 115200,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=1
)
while 1:
	x = ser.readline()
	print(x)
