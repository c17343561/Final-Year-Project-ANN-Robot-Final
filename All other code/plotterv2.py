from rplidar import RPLidar # Rplidar library
import time
import logging
import matplotlib.pyplot as plt
import numpy as np
import math
from math import cos, sin, radians, pi
from collections import deque

plt.close()

lidar = RPLidar(port='/dev/ttyUSB0') # Initialising an instance of a RPLidar object & calling it lidar

lidar.__init__('/dev/ttyUSB0',115200,3,None) # Initialising the object properties
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


data = [] # Initialising a data array where all the scan data will be stored
averageTime = [] # Average time array 
averageScans = [] # Average scans number array

for i, scan in enumerate(lidar.iter_scans()): # For loop for scanning continuesly (Until it reaches the if statement to brake when it reached a certain number of scans)
    startTime = time.time() # Time tag
    print('%d: Got %d measurments' % (i, len(scan))) # Printing how many measurments does one scan have
    elapsed = (time.time() - startTime) # Get time it took to gather one scan of multiple measurements
    print(scan)
    print('Time it took to get %d measurments is: %d us' % (len(scan),elapsed*1000000)) # Printing how long one scan took in microseconds
    data.append(scan) # Appending all scans into data array
    averageTime.append(elapsed*1000000) # Appending all time intervals for each scan
    averageScans.append(len(scan)) # Appending all number of measurements
    if i == 2: # If got 10 scans stop
        break # Break out of the loop
averageTime = sum(averageTime)/len(averageTime) # Getting average time
averageScans = sum(averageScans)/len(averageScans) # Getting average number of measurements

averageSampleRate = averageScans/(averageTime/1000000) # Getting average sample rate
print("***************************************************")
print("The Average Sample Rate is: %d Samples/second" % (averageSampleRate/3)) # Printing average sample rate
print("***************************************************")

angle = []
distance = []
lidar.stop_motor() # Stop motor
for k in range(len(data)-1): # Iterate through the data obtained
    for j in range(len(data[0])):
        if data[k][j][0] >= 10:
            angle.append(data[k][j][1]*((np.pi/180)))
            distance.append(data[k][j][2]) # Plot it on a polar 360 plot
arr_angles = np.array(angle)
arr_distances = np.array(distance)

ox = np.sin(arr_angles) * arr_distances
oy = np.cos(arr_angles) * arr_distances
plt.figure(figsize=(6,10))
plt.plot([oy, np.zeros(np.size(oy))], [ox, np.zeros(np.size(oy))], "ro-") # lines from 0,0 to the
plt.axis("equal")
bottom, top = plt.ylim()  # return the current ylim
plt.ylim((top, bottom)) # rescale y axis, to match the grid orientation
plt.grid(True)
plt.show()
