from rplidar import RPLidar # Rplidar library
import time
import logging
import matplotlib.pyplot as plt
import numpy as np

lidar = RPLidar(port='/dev/ttyUSB0') # Initialising an instance of a RPLidar object & calling it lidar

lidar.__init__('/dev/ttyUSB0',115200,3,None) # Initialising the object properties

lidar.connect() # Initiates connection with the rplidar

#lidar.set_pwm(1023) # Setting the scanrate at 5.5 Hz 
info = lidar.get_info() # Getting the lidar's info
print(info) # Printing the info obtained
health = lidar.get_health() # Getting the lidar's health
print(health) # Printing the health obtained
raw = []
data = []
def get_data():
    for i, scan in enumerate(lidar.iter_scan(max_buf_meas=500)): # For loop for scanning continuesly (Until it reaches the if statement to brake when it reached a certain number of scans)
        print(scan)
        break # Break out of the loop
    lidar.stop()
    return scan
lidar.stop_motor() # Stop motor