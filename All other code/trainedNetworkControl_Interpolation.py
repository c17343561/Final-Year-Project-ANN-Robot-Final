import tensorflow as tf # Tensorflow Neural Network
from tensorflow import keras # Keras runs on top of tf deep learning framework, has simple API's makes user interaction simple
import matplotlib.pyplot as plt # Plot library
import numpy as np # Mathematical computing
import os
import sys
import pygame
import RPi.GPIO as GPIO
import time
from rplidar import RPLidar # Rplidar library
from threading import Thread
import logging
import queue
import pandas as pd

finalDistances = [0]*73
predictLabel = [4]

BUF_SIZE = 10
q = queue.Queue(BUF_SIZE)
q2 = queue.Queue(BUF_SIZE)


class DataControl(object):
    
    global finalDistances

    def control(self):
        while True:
            
            multiple = 5
            fin = int(360/multiple)
            idealAngles = [multiple * i for i in range(1, fin + 1)]
            idealAngles = [0] + idealAngles

            print("****************************************************************")
            print('Starting the Lidar....')
            print("****************************************************************")
            lidar = RPLidar(port='/dev/ttyUSB0') # Initialising an instance of a RPLidar object & calling it lidar

            lidar.__init__('/dev/ttyUSB0',115200,3,None) # Initialising the object properties

            lidar.connect() # Initiates connection with the rplidar
            lidar.set_pwm(550) # Setting the scanrate at 5.5 Hz 
            info = lidar.get_info() # Getting the lidar's info
            print(info) # Printing the info obtained
            health = lidar.get_health() # Getting the lidar's health
            print(health) # Printing the health obtained
            
            numberOfMeasurements = 0
            for i, scan in enumerate(lidar.iter_scans(max_buf_meas=2000)): # For loop for scanning continuesly (Until it reaches the if statement to brake when it reached a certain number of scans)
                #print('%d: Got %d measurments' % (i, len(scan))) # Printing how many measurments does one scan have
                #print(scan)
                numberOfMeasurements += len(scan)
                angle = []
                distance = []
                finalDistances = []
                roughDistances = [0] * len(idealAngles)
                finalDistancesANN = []
                if i%5 == 0 and i != 0:
                    #print(scan)
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
                    if newAngles[-1] == newAngles[0]:
                        newDistances[-1] = round((newDistances[0] + newDistances[-1])/2,2)
                        newAngles.pop(0)
                        newDistances.pop(0)
                        
                    min_index = newAngles.index(min(newAngles))
                    
                    for s in range(min_index):
                        newAngles.append(newAngles[0])
                        newDistances.append(newDistances[0])
                        newAngles.pop(0)
                        newDistances.pop(0)
                    
                    for z in range(0,len(idealAngles)):
                        if idealAngles[z] in newAngles:
                            roughDistances[z] = newDistances[newAngles.index(idealAngles[z])]
                    
                    interpol = pd.Series(roughDistances)
                    interpol.replace(0,np.NaN, inplace=True)
                    x = interpol.interpolate(method = 'linear', limit_direction = 'both')
                    x = list(np.around(np.array(x),2))
                    finalDistances.append(x)
                    q.put(finalDistances)


class ReadingData(object):
    global finalDistances
    global predictLabel
    def read(self):
        print("****************************************************************")
        print('Loading & Initialising the Neural Network....')
        print("****************************************************************")
        testData = np.loadtxt('testData.csv', delimiter=',')
        testLabels = np.loadtxt('testLabels.csv', delimiter=',')
        testLabels = testLabels.astype(int)
        testData = testData.tolist()
        testLabels = testLabels.tolist()


        checkpointPath = "training_Interpolated/cp-{epoch:04d}.ckpt"
        checkpointDir = os.path.dirname(checkpointPath)

        os.listdir(checkpointDir)
        latest = tf.train.latest_checkpoint(checkpointDir) # Latest checpoint

        trainedNet = keras.Sequential()

        trainedNet.add(keras.layers.Dense(len(testData[0]),input_shape=(len(testData[0]), )))
        trainedNet.add(keras.layers.Dense(len(testData[0])*4, activation="relu")) # Dense ReLU Layer (hidden layer)
        trainedNet.add(keras.layers.Dense(len(testData[0])*2, activation="relu")) # Dense ReLU Layer (hidden layer)
        trainedNet.add(keras.layers.Dense(4, activation="softmax")) # Dense softmax layer (output layer) prepares the data for probability distribution

        #trainedNet.build()

        opt = keras.optimizers.Adam(learning_rate=0.01)
        trainedNet.compile(loss="sparse_categorical_crossentropy",
                         optimizer=opt,
                         metrics=["accuracy"]) # Compile the Neural Net with the desired loss function, optimizer, and metrics

        # Loads the weights
        trainedNet.load_weights(latest)

        trainedNet.summary() # Gives summerised information on the created Neural Network

        loss, acc = trainedNet.evaluate(testData, testLabels, verbose=2)
            
        print("****************************************************************")
        print("The Evaluation Accuracy on unseen data is: {:5.2f}%".format(100 * acc))
        print("****************************************************************")
        while True:
            finalDistances = q.get()
            
            finalDistancesANN = []
            finalDistancesANN.append(finalDistances)
            prediction = trainedNet.predict(finalDistancesANN)
            #print(finalDistancesANN)
            prediction = prediction.round()
            #print(prediction)
            predictLabel = np.argmax(trainedNet.predict(finalDistancesANN), axis=-1)
            #print(predictLabel)
            q2.put(predictLabel)

class MotorControl(object):
    
    global predictLabel
    def motors(self):
        lidarCtrl = 0
        print("****************************************************************")
        print("Initialising the GPIO pins and the stepper motors")
        print("****************************************************************")
        GPIO.setmode(GPIO.BCM)
        left_control_pins = [24,25,7,8]
        right_control_pins = [27,22,6,5]
        number_of_phases = 4
        time_between_steps = 0.0006 #seconds
        step_angle = 5.625 #degrees
        steps_revolution = int((360/step_angle)*number_of_phases)
        RPM = steps_revolution/time_between_steps # revolutions per second
        for left_pin in left_control_pins:
                #print("Setting up Left Motor GPIO pins")
                GPIO.setup(left_pin,GPIO.OUT)
                GPIO.output(left_pin, False)
        for right_pin in right_control_pins:
                #print("Setting up Right Motor GPIO pins")
                GPIO.setup(right_pin,GPIO.OUT)
                GPIO.output(right_pin, False)
            
        halfstep_seq = [
            [1,0,0,1],
            [0,0,0,1],
            [0,0,1,1],
            [0,0,1,0],
            [0,1,1,0],
            [0,1,0,0],
            [1,1,0,0],
            [1,0,0,0]]

        halfstepAnti_seq = [
            [1,0,0,0],
            [1,1,0,0],
            [0,1,0,0],
            [0,1,1,0],
            [0,0,1,0],
            [0,0,1,1],
            [0,0,0,1],
            [1,0,0,1]]
        
        stop_seq = [
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0]]

        reverse = [3,2,1,0]
        #Half Step Motor configuration 1 Full revolution
    
        timeout = 15
        while True:
            predictLabels = int(q2.get())
            if predictLabels == 1:
                print("Going Left")
                for indx in range(timeout):
                    for halfstep in range(8):
                        for pin in range(4):
                            GPIO.output(left_control_pins[pin], halfstep_seq[halfstep][pin])
                            GPIO.output(right_control_pins[pin], halfstep_seq[halfstep][pin])
                            time.sleep(time_between_steps)
                
        #-----------------------------------------------------------------------------
            #Half Step Motor configuration 1 Full revolution

            if predictLabels == 2:
                print("Going Right")
                for indx in range(timeout):
                    for halfstep in range(8):
                        for pin in range(4):
                            GPIO.output(left_control_pins[pin], halfstepAnti_seq[halfstep][pin])
                            GPIO.output(right_control_pins[pin], halfstepAnti_seq[halfstep][pin])
                            time.sleep(time_between_steps)

        #-----------------------------------------------------------------------------
            #Half Step Motor configuration 1 Full revolution
            
            if predictLabels == 0:
                print("Going Straight")
                for indx in range(timeout):
                    for halfstep in range(8):
                        for pin in range(4):
                            GPIO.output(left_control_pins[reverse[pin]], halfstepAnti_seq[halfstep][pin])
                            GPIO.output(right_control_pins[pin], halfstepAnti_seq[halfstep][pin])
                            time.sleep(time_between_steps)
        #-----------------------------------------------------------------------------

            #Half Step Motor configuration 1 Full revolution
            if predictLabels == 3:
                print("Going Back")
                for indx in range(timeout):
                    for halfstep in range(8):
                        for pin in range(4):
                            GPIO.output(left_control_pins[pin], halfstepAnti_seq[halfstep][pin])
                            GPIO.output(right_control_pins[reverse[pin]], halfstepAnti_seq[halfstep][pin])
                            time.sleep(time_between_steps)
                        

            if predictLabels == 4:
                print("Stopped")
                for indx in range(timeout):
                    for halfstep in range(8):
                        for pin in range(4):
                            GPIO.output(left_control_pins[pin], stop_seq[halfstep][pin])
                            GPIO.output(right_control_pins[reverse[pin]], stop_seq[halfstep][pin])
                            time.sleep(time_between_steps)
            
#-----------------------------------------------------------------------------

def main():
 
    print("Creating Reading Data Thread 1 ")
    r = ReadingData()
    readingDataThread = Thread(target = r.read)
    readingDataThread.setDaemon(True)
    readingDataThread.start()
    
    time.sleep(5)
    
    print("Creating Data Control Thread 2 ")
    d = DataControl()
    dataThread = Thread(target = d.control)
    dataThread.setDaemon(True)
    dataThread.start()
    
    time.sleep(5)

    print("Creating Motor Control Thread 3 ")
    m = MotorControl()
    controllingMotorsThread = Thread(target = m.motors)
    controllingMotorsThread.setDaemon(True)
    controllingMotorsThread.start()
    
    time.sleep(5)

if __name__ == '__main__':
    main()


