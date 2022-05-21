from rplidar import RPLidar # Rplidar library
import time
import logging
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

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
DMAX = 4000
IMIN = 0
IMAX = 50

lidar.connect() # Initiates connection with the rplidar

lidar.set_pwm(550) # Setting the scanrate at 5.5 Hz 
info = lidar.get_info() # Getting the lidar's info
print(info) # Printing the info obtained
health = lidar.get_health() # Getting the lidar's health
print(health) # Printing the health obtained


data = [] # Initialising a data array where all the scan data will be stored
averageTime = [] # Average time array 
averageScans = [] # Average scans number array
angle = []
distance = []



fig = plt.figure()
ax = fig.add_subplot(111,projection='polar')

line = ax.scatter(angle,distance,cmap='hsv',alpha=0.75, s = 1)
def animate(i):
    for k in range(len(scan)):
        angle.append(scan[k][1]*(np.pi/180))
        distance.append(scan[k][2])
        plt.cla()
        ax.scatter(angle,distance,cmap='hsv',alpha=0.75, s = 1)
    return line
        
        
for i, scan in enumerate(lidar.iter_scans()): # For loop for scanning continuesly (Until it reaches the if statement to brake when it reached a certain number of scans)
    startTime = time.time() # Time tag
    print('%d: Got %d measurments' % (i, len(scan))) # Printing how many measurments does one scan have
    elapsed = (time.time() - startTime) # Get time it took to gather one scan of multiple measurements
    print('Time it took to get %d measurments is: %d us' % (len(scan),elapsed*1000000)) # Printing how long one scan took in microseconds
    data.append(scan) # Appending all scans into data array
    averageTime.append(elapsed*1000000) # Appending all time intervals for each scan
    averageScans.append(len(scan)) # Appending all number of measurements
    
    ani = FuncAnimation(plt.gc  f(),animate,50)
    plt.show()


    if i >= 10: # If got 10 scans stop
        break # Break out of the loop

    
lidar.stop_motor()


averageTime = sum(averageTime)/len(averageTime) # Getting average time
averageScans = sum(averageScans)/len(averageScans) # Getting average number of measurements

averageSampleRate = averageScans/(averageTime/1000000) # Getting average sample rate
print("***************************************************")
print("The Average Sample Rate is: %d Samples/second" % (averageSampleRate/3)) # Printing average sample rate
print("***************************************************")


    


