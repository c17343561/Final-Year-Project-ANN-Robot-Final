from rplidar import RPLidar # Rplidar library
import time
import logging
import matplotlib.pyplot as plt
import numpy as np
import math
from math import cos, sin, radians, pi
from collections import deque

plt.close()

lidar = RPLidar(port='/dev/ttyAMA0') # Initialising an instance of a RPLidar object & calling it lidar

lidar.__init__('/dev/ttyUSB0',115200,2,None) # Initialising the object properties
'''
Important part the __init__(port,baudrate,timeout,logger) function initilises parameters for the wired connection with the lidar,
this is what I had a lot of trouble with as the timeout is set up to 1 second as default... I found that
it must be above 1 preferably 3 seconds.

For Raspbian this works:

port = '/dev/ttyUSB0'
baudrate = 115200
timeout = 3
logger = None
'''

lidar.connect() # Initiates connection with the rplidar

lidar.set_pwm(550) # Setting the scanrate at 5.5 Hz 
info = lidar.get_info() # Getting the lidar's info
print(info) # Printing the info obtained
health = lidar.get_health() # Getting the lidar's health
print(health) # Printing the health obtained


def get_data():
    lidar = RPLidar('/dev/ttyUSB0', baudrate=115200)
    for scan in lidar.iter_scans(max_buf_meas=500):
        break
    lidar.stop()
    return scan

for i in range(1000000):
    if(i%7==0):
        x = []
        y = []
    print(i)
    current_data=get_data()
    for point in current_data:
        if point[0]>=10:
            x.append(point[2]*np.sin(point[1]))
            y.append(point[2]*np.cos(point[1]))
    plt.clf()
    plt.scatter(x, y)
    plt.pause(.1)
plt.show()
