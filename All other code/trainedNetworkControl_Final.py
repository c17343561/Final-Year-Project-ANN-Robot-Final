'''
# Code written by: Rafal Wolk
# Date of last update: 21/05/2022
# Description: This code is used to control the robot on a trained neural network
# this is a multithreaded process. One thread is used for data collection and
# management to prepare it in the same way that neural network was trained on.
# The second thread loads in the trained neural network and takes the managed data 
# and classifies it. That classification is then passed over to another thread which
# drives the motors according to the classification of the environment. This enables
# the neural network to drive the robot in a maze environment.
# Purpose: Part of DT021A Year 4 Final Year Project
'''
import tensorflow as tf # Tensorflow Neural Network
from tensorflow import keras # Keras runs on top of tf deep learning framework, has simple API's makes user interaction simple
import matplotlib.pyplot as plt # Plot library
import numpy as np # Mathematical computing
import os # Operating system module
import sys # System module
import pygame # pygame module
import RPi.GPIO as GPIO # GPIO pins module
import time # Time module
from rplidar import RPLidar # Rplidar library
from threading import Thread # Thread module 
import logging # Logging data module
import queue # Queues for data sharing module

finalDistances = [0]*73 # Creating an array of zeros for finalDistances
predictLabel = [4] # Creating a preadic label array

BUF_SIZE = 10 # Buffer size for each of the queues is 10 entries
q = queue.Queue(BUF_SIZE) # Initialise Queue 1
q2 = queue.Queue(BUF_SIZE) # Initialise Queue 2


class DataControl(object): # Thread 2 Data collection and management
    
    global finalDistances # Making finalDistance a global variable

    def control(self): # Control function
        while True: # Continuous while loop
##################################################################################
# Initialising all necessary variables and starting the connection with the lidar
##################################################################################
            multiple = 5 # Multiples of 5 degrees 
            fin = int(360/multiple) # Number of elements in angle array
            idealAngles = [multiple * i for i in range(1, fin + 1)] # Creating ideal angle between 0 to 360 in multiples of 5 degrees
            idealAngles = [0] + idealAngles # Add a zero at the start of the array

            print("****************************************************************")
            print('Starting the Lidar....') # Print to terminal
            print("****************************************************************")
            lidar = RPLidar(port='/dev/ttyUSB0') # Initialising an instance of a RPLidar object & calling it lidar

            lidar.__init__('/dev/ttyUSB0',115200,3,None) # Initialising the object properties

            lidar.connect() # Initiates connection with the rplidar
            lidar.set_pwm(550) # Setting the scan rate at 5.5 Hz 
            info = lidar.get_info() # Getting the lidar's info
            print(info) # Printing the info obtained
            health = lidar.get_health() # Getting the lidar's health
            print(health) # Printing the health obtained
            
            numberOfMeasurements = 0
##################################################################################
# Reading the data from the lidar
##################################################################################
            for i, scan in enumerate(lidar.iter_scans(max_buf_meas=2000)): # For loop for scanning continuously (Until it reaches the if statement to brake when it reached a certain number of scans)
                #print('%d: Got %d measurments' % (i, len(scan))) # Printing how many measurments does one scan have
                #print(scan)
                numberOfMeasurements += len(scan) # Counting a number of measurements obtained in each scan
                angle = [] # Creating an angle array
                distance = [] # Creating a distance array
                finalDistances = [0] * len(idealAngles) # Resetting the finalDistances array to all zeros
                finalDistancesANN = [] # Creating another distances array
##################################################################################
# Managing the data every 4th scan and not very first scan. This prepares the distances
# list to be ready to put into the trained neural network
##################################################################################
                if i%4 == 0 and i != 0: # Skip if first scan, and process the data ever 4th scan
                    #print(scan)
                    counter = 1 # Reseat the counter
                    newAngles = [] # Reset the new angles
                    newDistances = [] # Reset new distances
                    roundedAngle = [] # Reset rounded angles
##################################################################################
# Read angles & distances from the scans
##################################################################################
                    for save in range(len(scan)): # From the entire scan save the angles and instances in their own arrays
                        angle.append(scan[save][1]) # Put the angles from the scan measurements int this array
                        distance.append(scan[save][2]) # Put the distances from the scan measurements to this array 
                    roundedAngle = [multiple * round(element/multiple) for element in angle] # This rounds to the nearest 5 multiple value all the angles 
##################################################################################
# Round the angles to the nearest multiple in this case 5 degrees
# Average out the distances that have the same angle multiples
##################################################################################
                    for j, k in enumerate(roundedAngle): # Iterate through rounded angle
                        if j < len(roundedAngle)-1 and roundedAngle[j+1] == k: # If the neighbouring elements are the same and it is not the end of the array
                            counter += 1 # Count up
                        else: # Else average out corresponding distances at those angles
                            newDistances.append(np.round(np.average(distance[j-(counter-1):j+1]),2)) # Average a counter range of distances 
                            newAngles.append(roundedAngle[j]) # Put new angles in 
                            counter = 1 # Reset the counter
##################################################################################
# Structuring the data so the angles start from 0 degrees and end with 360 degrees
##################################################################################
                    if newAngles[-1] == newAngles[0]: # If there are two or more of the same element remove the duplicates
                        newDistances[-1] = round((newDistances[0] + newDistances[-1])/2,2) # Round the same elements that share the same angle multiple 
                        newAngles.pop(0) # Pop the first element from the array
                        newDistances.pop(0) # Pop the first element from the array
                        
                    min_index = newAngles.index(min(newAngles)) # Find minimum index at  which number 0 occurs
                    for s in range(min_index): # Make the array go from 0 -> 360
                        newAngles.append(newAngles[0]) # Put the first element at the end of the array
                        newDistances.append(newDistances[0]) # Put the first element at the end of the array
                        newAngles.pop(0) # Pop the first element from the array
                        newDistances.pop(0) # Pop the first element from the array
##################################################################################
# Validating that the angles have been structured correctly
# Extracting the ready finalDistances list
##################################################################################
                    for z in range(0,len(idealAngles)): # Checking if the new angles  generated match the ideal angles which is the structure that the neural network accepts
                        if idealAngles[z] in newAngles:
                            finalDistances[z] = newDistances[newAngles.index(idealAngles[z])] # Extracting the finalDistances
##################################################################################
# Putting the ready finalDistances list into Queue 1
##################################################################################
                    q.put(finalDistances) # Putting the array into the Queue 1


class ReadingData(object): # Thread 1 classifying the collected data
    global finalDistances # Creating a global variable finalDistances
    global predictLabel # Creating a global variable predictLabel
    def read(self):
        print("****************************************************************")
        print('Loading & Initialising the Neural Network....') # Print to terminal
        print("****************************************************************")
##################################################################################
# Initiating all necessary variable for classification
# Loading in the test data & labels to test later if the trained neural network 
# functions properly
##################################################################################
        testData = np.loadtxt('testData.csv', delimiter=',') # Load test data to make sure later the network is imported correctly
        testLabels = np.loadtxt('testLabels.csv', delimiter=',') # Load in the test labels to classify them
        testLabels = testLabels.astype(int) # Change the type of the labels to integers
        testData = testData.tolist() # Convert the test data to a list
        testLabels = testLabels.tolist() # Convert the test labels to a list


        checkpointPath = "training_2/cp-{epoch:04d}.ckpt" # Load in the path of a file that contains the trained neural network weight and biases
        checkpointDir = os.path.dirname(checkpointPath) # Specify the directory of the file
        os.listdir(checkpointDir)
        latest = tf.train.latest_checkpoint(checkpointDir) # Load in the latest weight and biases checkpoint from the training process
##################################################################################
# Initialising an empty skeleton structure for the neural network 
##################################################################################
        trainedNet = keras.Sequential() # Initialise a sequential neural network structure

        trainedNet.add(keras.layers.Dense(len(testData[0]),input_shape=(len(testData[0]), )))
        trainedNet.add(keras.layers.Dense(len(testData[0])*4, activation="relu")) # Dense ReLU Layer (hidden layer)
        trainedNet.add(keras.layers.Dense(len(testData[0])*2, activation="relu")) # Dense ReLU Layer (hidden layer)
        trainedNet.add(keras.layers.Dense(4, activation="softmax")) # Dense softmax layer (output layer) prepares the data for probability distribution

        #trainedNet.build()

        opt = keras.optimizers.Adam(learning_rate=0.01)
        trainedNet.compile(loss="sparse_categorical_crossentropy", # Cross entropy to calculate losses
                         optimizer=opt, # Load in the options described
                         metrics=["accuracy"]) # Compile the Neural Net with the desired loss function, optimizer, and metrics
##################################################################################
# Loading in the trained network weight and biases into the compiled empty skeleton
# structure
##################################################################################
        # Loads the weights
        trainedNet.load_weights(latest) # Load in the weights and biases into an empty skeleton of the neural network

        trainedNet.summary() # Gives summerised information on the created Neural Network
##################################################################################
# Test the trained network if it was imported correctly
##################################################################################
        loss, acc = trainedNet.evaluate(testData, testLabels, verbose=2) # Test if the network loaded in correctly
            
        print("****************************************************************")
        print("The Evaluation Accuracy on unseen data is: {:5.2f}%".format(100 * acc))
        print("****************************************************************")
##################################################################################
# Continuously classify data received from Queue 2
##################################################################################
        while True: # Continuous while loop
            finalDistances = q.get() # Get what is in Queue 1
            
            finalDistancesANN = [] # Initialise/reset the temporary list
            finalDistancesANN.append(finalDistances) # Put finalDistances into this temporary list
            prediction = trainedNet.predict(finalDistancesANN) # Classify the data on a trained network
            #print(finalDistancesANN)
            prediction = prediction.round() # Round the prediction value
            #print(prediction)
            predictLabel = np.argmax(trainedNet.predict(finalDistancesANN), axis=-1) # Predict the label of the data obtained
            #print(predictLabel)
            q2.put(predictLabel) # Put the result into Queue 2

class MotorControl(object): # Thread 3 controlling the motors on the result
    
    global predictLabel # initialise the predictLabel global variable
    def motors(self):
        print("****************************************************************")
        print("Initialising the GPIO pins and the stepper motors")
        print("****************************************************************")
##################################################################################
# Initialise all necessary variables for controlling motors
##################################################################################

        GPIO.setmode(GPIO.BCM) # Setting the mode of the pins
        left_control_pins = [24,25,7,8] # Pin numbers the left motor is connected to
        right_control_pins = [27,22,6,5] # Pin numbers that the right motor is connected to
        number_of_phases = 4 # Motors have 4 phases
        time_between_steps = 0.0006 # Timeout between steps in seconds
        step_angle = 5.625 # Angle of each step in degrees
        steps_revolution = int((360/step_angle)*number_of_phases) # Number of steps it takes for one full revolution
        RPM = steps_revolution/time_between_steps # revolutions per second
        for left_pin in left_control_pins: # Set the pins for the left motor
                #print("Setting up Left Motor GPIO pins")
                GPIO.setup(left_pin,GPIO.OUT) # Set the pins as outputs
                GPIO.output(left_pin, False)
        for right_pin in right_control_pins: # Set the pins for the right motor
                #print("Setting up Right Motor GPIO pins")
                GPIO.setup(right_pin,GPIO.OUT) # Set the pins as outputs
                GPIO.output(right_pin, False)
        # Half step sequence clockwise
        halfstep_seq = [
            [1,0,0,1],
            [0,0,0,1],
            [0,0,1,1],
            [0,0,1,0],
            [0,1,1,0],
            [0,1,0,0],
            [1,1,0,0],
            [1,0,0,0]]
        # Half step sequence anti clockwise
        halfstepAnti_seq = [
            [1,0,0,0],
            [1,1,0,0],
            [0,1,0,0],
            [0,1,1,0],
            [0,0,1,0],
            [0,0,1,1],
            [0,0,0,1],
            [1,0,0,1]]
        # Stop all motors
        stop_seq = [
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0]]

        reverse = [3,2,1,0] # Reverse sequence
        #Half Step Motor configuration 1 Full revolution
        timeout = 15 # Timeout between reading Queue 2


##################################################################################
# Continuously drive the motors depending on the result of the classification form the
# neural network prediction
##################################################################################

        while True: # Continuous while loop
            predictLabels = int(q2.get()) # Get the prediction from the Queue 2
            if predictLabels == 1: # If prediction is to go left
                print("Going Left") # Print to terminal
                for indx in range(timeout): # Drive the motors until timeout runs out
                    for halfstep in range(8): # Go through the half step sequence
                        for pin in range(4): # Write to pins
                            GPIO.output(left_control_pins[pin], halfstep_seq[halfstep][pin])
                            GPIO.output(right_control_pins[pin], halfstep_seq[halfstep][pin])
                            time.sleep(time_between_steps) # Timeout between steps
                
        #-----------------------------------------------------------------------------
            #Half Step Motor configuration 1 Full revolution

            if predictLabels == 2: # If prediction is to go right
                print("Going Right")
                for indx in range(timeout): # Drive the motors until timeout runs out
                    for halfstep in range(8): # Go through the half step sequence
                        for pin in range(4): # Write to pins
                            GPIO.output(left_control_pins[pin], halfstepAnti_seq[halfstep][pin])
                            GPIO.output(right_control_pins[pin], halfstepAnti_seq[halfstep][pin])
                            time.sleep(time_between_steps) # Timeout between steps

        #-----------------------------------------------------------------------------
            #Half Step Motor configuration 1 Full revolution
            
            if predictLabels == 0: # If prediction is to go forward
                print("Going Straight")
                for indx in range(timeout): # Drive the motors until timeout runs out
                    for halfstep in range(8): # Go through the half step sequence
                        for pin in range(4): # Write to pins
                            GPIO.output(left_control_pins[reverse[pin]], halfstepAnti_seq[halfstep][pin])
                            GPIO.output(right_control_pins[pin], halfstepAnti_seq[halfstep][pin])
                            time.sleep(time_between_steps) # Timeout between steps
        #-----------------------------------------------------------------------------

            #Half Step Motor configuration 1 Full revolution
            if predictLabels == 3: # If prediction is to go backward
                print("Going Back")
                for indx in range(timeout): # Drive the motors until timeout runs out
                    for halfstep in range(8): # Go through the half step sequence
                        for pin in range(4): # Write to pins
                            GPIO.output(left_control_pins[pin], halfstepAnti_seq[halfstep][pin])
                            GPIO.output(right_control_pins[reverse[pin]], halfstepAnti_seq[halfstep][pin])
                            time.sleep(time_between_steps) # Timeout between steps
                        

            if predictLabels == 4: # If prediction is to stop motors
                print("Stopped")
                for indx in range(timeout): # Drive the motors until timeout runs out
                    for halfstep in range(8): # Go through the half step sequence
                        for pin in range(4): # Write to pins
                            GPIO.output(left_control_pins[pin], stop_seq[halfstep][pin])
                            GPIO.output(right_control_pins[reverse[pin]], stop_seq[halfstep][pin])
                            time.sleep(time_between_steps) # Timeout between steps
            
#-----------------------------------------------------------------------------
##################################################################################
# Main function initialise and start up all threads
##################################################################################
def main(): # Main
 
    print("Creating Reading Data Thread 1 ") # Print to terminal
    r = ReadingData() # Create an instance of a class
    readingDataThread = Thread(target = r.read) # Create a Thread 1
    readingDataThread.setDaemon(True) # Set the Daemon which allows the program to exit
    readingDataThread.start() # Start the thread
    
    time.sleep(5) # Let Thread 1 initialise and load everything
    
    print("Creating Data Control Thread 2 ") # Print to terminal
    d = DataControl() # Create an instance of a class
    dataThread = Thread(target = d.control) # Create a Thread 2
    dataThread.setDaemon(True) # Set the Daemon which allows the program to exit
    dataThread.start() # Start the thread
    
    time.sleep(5) # Let Thread 2 initialise and load everything

    print("Creating Motor Control Thread 3 ") # Print to terminal
    m = MotorControl()# Create an instance of a class
    controllingMotorsThread = Thread(target = m.motors) # Create a Thread 3
    controllingMotorsThread.setDaemon(True) # Set the Daemon which allows the program to exit
    controllingMotorsThread.start() # Start the thread
    
    time.sleep(5) # Let Thread 3 initialise and load everything

if __name__ == '__main__':
    main()
