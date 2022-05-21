from rplidar import RPLidar # Rplidar library

lidar = RPLidar(port='/dev/ttyUSB0') # Initialising an instance of a RPLidar object & calling it lidar

lidar.__init__('/dev/ttyUSB0',115200,3,None) # Initialising the object properties


lidar.connect() # Initiates connection with the rplidar

lidar.set_pwm(550) # Setting the scanrate at 5.5 Hz 
info = lidar.get_info() # Getting the lidar's info
print(info) # Printing the info obtained
health = lidar.get_health() # Getting the lidar's health
print(health) # Printing the health obtained


for i, scan in enumerate(lidar.iter_measurments(max_buf_meas=500)): # For loop for scanning continuesly (Until it reaches the if statement to brake when it reached a certain number of scans)
    print('%d: Got %d measurments' % (i, len(scan))) # Printing how many measurments does one scan have
    print(scan)
    if i == 360: # If got 10 scans stop
        lidar.stop_motor() # Stop motor
        break # Break out of the loop

