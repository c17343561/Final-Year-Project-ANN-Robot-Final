'''
# Code written by: Rafal Wolk
# Date of last update: 21/05/2022
# Description: Data collection code that enables to create your own dataset
# it collects and manages the data and writes it to an Excel file specified
# Purpose: Part of DT021A Year 4 Final Year Project
'''

from rplidar import RPLidar # Rplidar library
import csv # Excel module
import datetime # Date and time module
import os # Operating system module
import numpy as np # Numpy module
import statistics as stat # Statistics module 

numberOfScans = 0   # Initiating number of scans integer
numberOfMeasurements = 0 # Initiating number of measurements integer
multiple = 5 # Specifying the multiples that which the angles will go up in
fin = int(360/multiple) # Specifying the number of elements in the angle array


filePath = '/home/pi/FYP/lidar/dataSet.csv'; # Path and name of the excel file the results will be sotred in

if os.path.exists(filePath): # If the file already exists delete
    os.remove(filePath) # Delete the file
else: # Else carry on
    print("Can't delete the file as it doesn't exist")


lidar = RPLidar(port='/dev/ttyUSB0') # Initialising an instance of a RPLidar object & calling it lidar

lidar.__init__('/dev/ttyUSB0',115200,3,None) # Initialising the object properties

lidar.connect() # Initiates connection with the rplidar
lidar.set_pwm(550) # Setting the scanrate at 5.5 Hz 
info = lidar.get_info() # Getting the lidar's info
print(info) # Printing the info obtained
health = lidar.get_health() # Getting the lidar's health
print(health) # Printing the health obtained

def multiples(multi, f): # Creating an array at multiples of 5 in this case
    return [multi * i for i in range(1, f + 1)]

idealAngles = multiples(multiple, fin) # Calling the function to create an array at multiples of 5
idealAngles = [0] + idealAngles # Add a 0 at the start of the array

for i, scan in enumerate(lidar.iter_scans(max_buf_meas=5000)): # For loop for scanning continuesly (Until it reaches the if statement to brake when it reached a certain number of scans)
    print('%d: Got %d measurments' % (i, len(scan))) # Printing how many measurments does one scan have
    #print(scan)
    numberOfMeasurements += len(scan) # Add 1 to the count of scans
    angle = [] # Initialise the Angle list
    distance = [] # Innitialise the distance list

    if i%2 == 0 and i != 0: # Collect every second scan and ignor the very first scan
        counter = 1 # Set counter to 1
        newAngles = [] # New angles list
        newDistances = [] # New distances list
        finalDistances = [0] * len(idealAngles) # Creating a finalDistances list full of zeros
        roundedAngle = [] # Creating rounded angle list
        for save in range(len(scan)): # Iterate through measurements in the scan
            angle.append(scan[save][1]) # Extract all the angles and put them into this list
            distance.append(scan[save][2]) # Extract all the distances and put them into this list
        roundedAngle = [multiple * round(element/multiple) for element in angle] # Creting a list of ideal rounded angles that rounds up all angles to its nearest mutiple value in this case 5
        for j, k in enumerate(roundedAngle): # Iterate though rounded angle
            if j < len(roundedAngle)-1 and roundedAngle[j+1] == k: # If the next element is the same multiple value 
                counter += 1 # Add 1 to the counter
            else: # Else average out all the angles and distances at these angles
                newDistances.append(np.round(np.average(distance[j-(counter-1):j+1]),2)) # Average distance at that range
                newAngles.append(roundedAngle[j]) # Put the ideal angle in
                counter = 1 # Reset the counter to 1
        if newAngles[-1] == newAngles[0]: # If first and last element are the same average them
            newDistances[-1] = round((newDistances[0] + newDistances[-1])/2,2) # Avriging the two elements
            newAngles.pop(0) # Get rid of the first element 
            newDistances.pop(0) # Get rid of the first element 
            
        min_index = newAngles.index(min(newAngles)) # Find the index at which 0 degrees occurs
        
        for s in range(min_index): # Make the list go from 0 degrees to 360 degrees
            newAngles.append(newAngles[0]) # Put the new average at the end of the list
            newDistances.append(newDistances[0]) # Put the new average at the end of the list
            newAngles.pop(0) # Delete the first element
            newDistances.pop(0) # Delete the first element
        
        for z in range(0,len(idealAngles)): # Create the finalDistances list
            if idealAngles[z] in newAngles:
                finalDistances[z] = newDistances[newAngles.index(idealAngles[z])] # Extract all finalDistances
        
        for ele in range(len(idealAngles)): # Write to the excel file the results
            with open(filePath, 'a', newline='') as csvfile: # Open the file
                csvwriter = csv.writer(csvfile, delimiter=',',escapechar=' ', quoting=csv.QUOTE_NONE) # Setting up the writer
                csvwriter.writerow([idealAngles[ele]] + [finalDistances[ele]] + [''] + [datetime.datetime.now()]) # Writing the Angles, Distances and timestampt to each row
        print("Written at: ", datetime.datetime.now()) # Print to terminal
        print(finalDistances) # Print to terminal
        numberOfScans += 1 # Increment the number of scans
############# Stop motor and close file after 21 scans
    '''
    if i == 11: # If got 10 scans stop
        csvfile.close()
        lidar.stop_motor() # Stop motor
        break # Break out of the loop
    '''
print("Data collected: %d scans and %d measurements" % (numberOfScans, numberOfMeasurements)) # Print result
