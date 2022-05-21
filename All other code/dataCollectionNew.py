from rplidar import RPLidar # Rplidar library
import csv
import datetime
import os
import numpy as np
import statistics as stat

numberOfScans = 0
numberOfMeasurements = 0
multiple = 10

filePath = '/home/pi/FYP/lidar/dataSet.csv';

if os.path.exists(filePath):
    os.remove(filePath)
else:
    print("Can't delete the file as it doesn't exist")


lidar = RPLidar(port='/dev/ttyUSB1') # Initialising an instance of a RPLidar object & calling it lidar

lidar.__init__('/dev/ttyUSB1',115200,3,None) # Initialising the object properties

lidar.connect() # Initiates connection with the rplidar
lidar.set_pwm(550) # Setting the scanrate at 5.5 Hz 
info = lidar.get_info() # Getting the lidar's info
print(info) # Printing the info obtained
health = lidar.get_health() # Getting the lidar's health
print(health) # Printing the health obtained

for i, scan in enumerate(lidar.iter_measurments(max_buf_meas=5000)): # For loop for scanning continuesly (Until it reaches the if statement to brake when it reached a certain number of scans)
    print('%d: Got %d measurments' % (i, len(scan))) # Printing how many measurments does one scan have
    #print(scan)
    numberOfMeasurements += len(scan)
    angle = []
    distance = []
    print(scan)
    
    
    if i >= 300:
        lidar.stop_motor() # Stop motor
        break # Break out of the loop
'''
        counter = 1
        newAngles = []
        newDistances = []
        roundedAngle = []
        for save in range(len(scan)):
            angle.append(scan[save][1])
            distance.append(scan[save][2])
        roundedAngle = [multiple * round(element/multiple) for element in angle]
        
        for j, k in enumerate(roundedAngle):
            if j < len(roundedAngle)-1 and roundedAngle[j+1] == k:
                counter += 1
            else:
                newDistances.append(np.round(np.average(distance[j-(counter-1):j+1]),2))
                newAngles.append(roundedAngle[j])
                counter = 1
        for ele in range(len(newAngles)):
            with open(filePath, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',',escapechar=' ', quoting=csv.QUOTE_NONE)
                csvwriter.writerow([newAngles[ele]] + [newDistances[ele]] + [''] + [datetime.datetime.now()])
        print("Written at: ", datetime.datetime.now())
        numberOfScans += 1
        
        with open(filePath, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',',escapechar=' ', quoting=csv.QUOTE_NONE)
            csvwriter.writerow(['end'])
        
    if i == 21: # If got 10 scans stop
        csvfile.close()
        lidar.stop_motor() # Stop motor
        break # Break out of the loop
        
print("Data collected: %d scans and %d measurements" % (numberOfScans, numberOfMeasurements))
'''