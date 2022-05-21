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
 
predictLabel = [0]

def getKey(keyName):
    ans = False
    for eve in pygame.event.get():pass
    keyInput = pygame.key.get_pressed()
    myKey = getattr(pygame,'K_{}'.format(keyName))
    if keyInput [myKey]:
        ans = True
    pygame.display.update()
    
    return ans

def MotorControl():
    
    global predictLabel
    
    lidarCtrl = 0

    GPIO.setmode(GPIO.BCM)
    left_control_pins = [24,25,7,8]
    right_control_pins = [27,22,6,5]
    number_of_phases = 4
    time_between_steps = 0.0006 #seconds
    step_angle = 5.625 #degrees
    steps_revolution = int((360/step_angle)*number_of_phases)
    RPM = steps_revolution/time_between_steps # revolutions per second
    for left_pin in left_control_pins:
            print("Setting up Left Motor GPIO pins")
            GPIO.setup(left_pin,GPIO.OUT)
            GPIO.output(left_pin, False)
    for right_pin in right_control_pins:
            print("Setting up Right Motor GPIO pins")
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
    [1,0,0,0],
    ]

    #Half Step Motor configuration 1 Full revolution

    while int(predictLabel[0]) == 1:
        for halfstep in range(8):
            for pin in range(4):
                GPIO.output(left_control_pins[pin], halfstep_seq[halfstep][pin])
                GPIO.output(right_control_pins[pin], halfstep_seq[halfstep][pin])
                time.sleep(time_between_steps)
#-----------------------------------------------------------------------------
    halfstepAnti_seq = [
      [1,0,0,0],
      [1,1,0,0],
      [0,1,0,0],
      [0,1,1,0],
      [0,0,1,0],
      [0,0,1,1],
      [0,0,0,1],
      [1,0,0,1]
    ]

    #Half Step Motor configuration 1 Full revolution

    while int(predictLabel[0]) == 2:
        for halfstep in range(8):
            for pin in range(4):
                GPIO.output(left_control_pins[pin], halfstepAnti_seq[halfstep][pin])
                GPIO.output(right_control_pins[pin], halfstepAnti_seq[halfstep][pin])
                time.sleep(time_between_steps)
#-----------------------------------------------------------------------------

    halfstepAnti_seq = [
      [1,0,0,0],
      [1,1,0,0],
      [0,1,0,0],
      [0,1,1,0],
      [0,0,1,0],
      [0,0,1,1],
      [0,0,0,1],
      [1,0,0,1]
    ]

    #Half Step Motor configuration 1 Full revolution
    reverse = [3,2,1,0]
    while int(predictLabel[0]) == 0:
        for halfstep in range(8):
            for pin in range(4):
                GPIO.output(left_control_pins[reverse[pin]], halfstepAnti_seq[halfstep][pin])
                GPIO.output(right_control_pins[pin], halfstepAnti_seq[halfstep][pin])
                time.sleep(time_between_steps)
#-----------------------------------------------------------------------------
    halfstepAnti_seq = [
      [1,0,0,0],
      [1,1,0,0],
      [0,1,0,0],
      [0,1,1,0],
      [0,0,1,0],
      [0,0,1,1],
      [0,0,0,1],
      [1,0,0,1]
    ]

    #Half Step Motor configuration 1 Full revolution
    reverse = [3,2,1,0]
    while int(predictLabel[0]) == 3:
        for halfstep in range(8):
            for pin in range(4):
                GPIO.output(left_control_pins[pin], halfstepAnti_seq[halfstep][pin])
                GPIO.output(right_control_pins[reverse[pin]], halfstepAnti_seq[halfstep][pin])
                time.sleep(time_between_steps)
                
    stop_seq = [
      [0,0,0,0],
      [0,0,0,0],
      [0,0,0,0],
      [0,0,0,0],
      [0,0,0,0],
      [0,0,0,0],
      [0,0,0,0],
      [0,0,0,0]
    ]

    while int(predictLabel[0]) == 4:
        for halfstep in range(8):
            for pin in range(4):
                GPIO.output(left_control_pins[pin], stop_seq[halfstep][pin])
                GPIO.output(right_control_pins[reverse[pin]], stop_seq[halfstep][pin])
                time.sleep(time_between_steps)
#-----------------------------------------------------------------------------
def DataControl():
    
    
    
    def init():
        pygame.init()
        os.environ["DISPLAY"] = ":0"
        pygame.display.init()
        win = pygame.display.set_mode((100,100))

    def multiples(multi, f):
        return [multi * i for i in range(1, f + 1)]

    init()
    while True:
        
        multiple = 5
        fin = int(360/multiple)
        idealAngles = multiples(multiple, fin)
        idealAngles = [0] + idealAngles
        
        
        if getKey('l'):
            print("****************************************************************")
            print('Loading & Initialising the Neural Network....')
            print("****************************************************************")
            testData = np.loadtxt('testData.csv', delimiter=',')
            testLabels = np.loadtxt('testLabels.csv', delimiter=',')
            testLabels = testLabels.astype(int)
            testData = testData.tolist()
            testLabels = testLabels.tolist()


            checkpointPath = "training_2/cp-{epoch:04d}.ckpt"
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
    #####################################################################################
        if getKey('t'):
            global predictLabel
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
            for i, scan in enumerate(lidar.iter_scans(max_buf_meas=10000)): # For loop for scanning continuesly (Until it reaches the if statement to brake when it reached a certain number of scans)
                print('%d: Got %d measurments' % (i, len(scan))) # Printing how many measurments does one scan have
                #print(scan)
                numberOfMeasurements += len(scan)
                angle = []
                distance = []
                finalDistances = [0] * len(idealAngles)
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
                            finalDistances[z] = newDistances[newAngles.index(idealAngles[z])]
                    finalDistancesANN.append(finalDistances)
                    prediction = trainedNet.predict(finalDistancesANN)
                    prediction = prediction.round()
                    #print(prediction)
                    predictLabel = np.argmax(trainedNet.predict(finalDistancesANN), axis=-1)
                    print(predictLabel)
                '''
                if int(predictLabel[0]) == 0:
                    forward_motors(left_control_pins,right_control_pins)
                    
                elif int(predictLabel[0]) == 1:
                    left_motors(left_control_pins,right_control_pins)
                    
                elif int(predictLabel[0]) == 2:
                    right_motors(left_control_pins,right_control_pins)
                    
                elif int(predictLabel[0]) == 3:
                    reverse_motors(left_control_pins,right_control_pins)
                ''' 
    #################################################################################              
                if getKey('y'):
                    print("****************************************************************")
                    print("LiDAR is stopped....")
                    print("****************************************************************")
                    lidar.stop_motor() # Stop motor
                    lidar.stop()
                    break
    #################################################################################
        if getKey('q'):
            print("****************************************************************")
            print('Quitting....')
            print("****************************************************************")
            pygame.quit()
            exit()

dataThread = Thread(target = DataControl())
dataThread.start()

motorThread = Thread(target = MotorControl())
motorThread.start()


