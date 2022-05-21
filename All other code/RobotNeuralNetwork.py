'''
# Code written by: Rafal Wolk
# Date of last update: 21/05/2022
# Description: This code reads the labelled dataset in the excel file and manages 
# the data to train a neural network. The trained neural network’s weights and biases 
# are then saved in a folder to be later used for classification. This code trains the
# network on non-interpolated data so some distances have a value of 0 mm
# Purpose: Part of DT021A Year 4 Final Year Project
'''
import csv
import tensorflow as tf # Tensorflow Neural Network
from tensorflow import keras # Keras runs on top of tf deep learning framework, has simple API's makes user interaction simple
import numpy as np # Mathematical computing
import random
import os

def multiples(multi, f): # Multiples function it can create an array between any number in any multiples (e.g. 0 - 360 in multiples of 5)
    return [multi * i for i in range(1, f + 1)] # Calculations in order to achieve the functionality

############################################################################
# Checkpoints patch is set up, that is where the checkpoins of the neural network's
# weights and biases will be saved as the neural net trains.
############################################################################
checkpointPath = "training_3TEST/cp-{epoch:04d}.ckpt"  # Patch to which the checkpoints will be saved and their structure
checkpointDir = os.path.dirname(checkpointPath) # Saving the directory name 

batchSize = 64 # Number of training examples utilised in one iteration. The higher the batch size, the more memory space you'll need

angles = [] # Initialising an empty list for angles
distances = [] # Initialising an empty list for distances
labels = [] # Initialising an empty list for labels
finalDistances = [] # Initialising an empty list for finalDistances

#dataSet = open('dataSetFinal.csv') # Single run in the maze
dataSet = open('dataSetFinal2_CombinedBack.csv', 'r') # Multiple runs in the maze
################# Reading the data
with dataSet as File:  # Iterate through the dataset file
    reader = csv.reader(File) # Using csv.reader to iterate through the rows of the excel spreadsheet
    for row in reader:
        angles.append(int(row[0])) # Read all angles from the spreadsheet and store them in this list
        distances.append(float(row[1])) # Read all distances from the spreadsheet and store them in this list

        if row[2] != '': # Read labels until they end
            labels.append(int(row[2])) # Read labels and store them in a list
            
multiple = angles[1] - angles[0] # Getting multiple automatically by getting the difference wbetween tow adjacent elements
lidarResolution = 360 # Resolution of the lidar is 360 degrees
fin = int(lidarResolution/multiple) # Number of elements 
scanSize = int((lidarResolution/multiple) + 1) # Getting the size of the scan list

################# Retrieving final distances for 0 -> 360
for i in range(int(len(distances)/scanSize)):
    finalDistances.append(distances[scanSize*i:scanSize*(i+1)]) # Put all distances into one finalDistances list

angleResolution = multiples(multiple, fin) # Creating a list of 0 to 360 in multiples of 5 in this case
angleResolution = [0] + angleResolution # Add a zero at the start

randomNumber = 42 # Weights and biases
trainRatio = 80 # 80% of the data is split into training
validationRatio = 10 # 10% of the data is split into validation
testRatio = 10 # 10% of the data is split into testing

################## Creating empty lists for the training, validation and testing data and correcponding labels
trainData = []
trainLabels = []
validationData = []
validationLabels = []
testData = []
testLabels = []
################## These lists will be used to store indexes of the elements of each of the labels
forwardIndx = []
leftIndx = []
rightIndx = []
backwardIndx = []
################# Counting the amount of forward, left, right and backward data there is
for k in range(len(labels)): # Iterate through the labels
    if labels[k] == 0: # If forward
        forwardIndx.append(k) # Put the index of that label into forward list
    elif labels[k] == 1: # If left
        leftIndx.append(k) # Put the index of that label into left list
    elif labels[k] == 2: # If right
        rightIndx.append(k) # Put the index of that label into right list
    elif labels[k] == 4: # If backward
        backwardIndx.append(k) # Put the index of that label into backward list
################# Balancing the data to right turns as it is the shortest
# Randomises the indexes, in order for the robot be trained in different areas of the maze every time
# So that the network is not overfitted to only forward classification for example
random.shuffle(forwardIndx)
random.shuffle(leftIndx)
random.shuffle(rightIndx)
shortestArray = min(forwardIndx,leftIndx,rightIndx, key=len) # Find which of the actions is the smallest

balancingLength = len(shortestArray) # What is the size of the smallest action
# Balancing part
forwardIndx = forwardIndx[0:balancingLength] # Making forward action be the same length as the smallest action
leftIndx = leftIndx[0:balancingLength] # Making left action be the same length as the smallest action
rightIndx = rightIndx[0:balancingLength] # Making right action be the same length as the smallest action
################# Extracting the balanced data for training dataset
extractLeftIndx = random.sample(leftIndx,k=round(len(leftIndx)*(trainRatio/100))) # Extract random left indexes
extractRightIndx = random.sample(rightIndx,k=round(len(rightIndx)*(trainRatio/100))) # Extract random right indexes
extractForwardIndx = random.sample(forwardIndx,k=round(len(forwardIndx)*(trainRatio/100))) # Extract random forward indexes

################# Getting rid of the indexes from the original list so they are not used again
for i in extractLeftIndx[:]:
    if i in leftIndx:
        leftIndx.remove(i)
for i in extractRightIndx[:]:
    if i in rightIndx:
        rightIndx.remove(i)
for i in extractForwardIndx[:]:
    if i in forwardIndx:
        forwardIndx.remove(i)
################# Extracting the balanced data for testing dataset
extractLeftIndx_test = random.sample(leftIndx,k=round(len(leftIndx)*(50/100))) # Extract random left indexes
extractRightIndx_test = random.sample(rightIndx,k=round(len(rightIndx)*(50/100))) # Extract random right indexes
extractForwardIndx_test = random.sample(forwardIndx,k=round(len(forwardIndx)*(50/100))) # Extract random forward indexes

################# Getting rid of the indexes from the original list so they are not used again
for i in extractLeftIndx_test[:]:
    if i in leftIndx:
        leftIndx.remove(i)
for i in extractRightIndx_test[:]:
    if i in rightIndx:
        rightIndx.remove(i)
for i in extractForwardIndx_test[:]:
    if i in forwardIndx:
        forwardIndx.remove(i)
################# The left over data is for validation dataset and then the labels and data is inserted into their own data structures
################# This is where the train, validation and test datasets are finalised using balanced indexes along with their corresponding labels
# Train, validation & test datasets & labels of left actions
for z in range(len(extractLeftIndx)):
    trainData.append(finalDistances[extractLeftIndx[z]])
    trainLabels.append(labels[extractLeftIndx[z]])
    if z < len(leftIndx):
        validationData.append(finalDistances[leftIndx[z]])
        validationLabels.append(labels[leftIndx[z]])
    if z < len(extractLeftIndx_test):
        testData.append(finalDistances[extractLeftIndx_test[z]])
        testLabels.append(labels[extractLeftIndx_test[z]])
# Train, validation & test datasets & labels of right actions     
for z in range(len(extractRightIndx)):
    trainData.append(finalDistances[extractRightIndx[z]])
    trainLabels.append(labels[extractRightIndx[z]])
    if z < len(rightIndx):
        validationData.append(finalDistances[rightIndx[z]])
        validationLabels.append(labels[rightIndx[z]])
    if z < len(extractRightIndx_test):
        testData.append(finalDistances[extractRightIndx_test[z]])
        testLabels.append(labels[extractRightIndx_test[z]])
# Train, validation & test datasets & labels of forward actions
for z in range(len(extractForwardIndx)):
    trainData.append(finalDistances[extractForwardIndx[z]])
    trainLabels.append(labels[extractForwardIndx[z]])
    if z < len(forwardIndx):
        validationData.append(finalDistances[forwardIndx[z]])
        validationLabels.append(labels[forwardIndx[z]])
    if z < len(extractForwardIndx_test):
        testData.append(finalDistances[extractForwardIndx_test[z]])
        testLabels.append(labels[extractForwardIndx_test[z]])
'''
##################################################################
# This code was used for randomly shuffling two arrays at once and the elements
# in each array would still correspond with each other after shuffle
# It zips up the two arrays together and shuffles them in the same order
# It proved itself not valuable for this project but it is there if needed
##################################################################

mixTrain = list(zip(trainData, trainLabels))
random.shuffle(mixTrain)
trainData, trainLabels = zip(*mixTrain)

mixValidation = list(zip(validationData, validationLabels))
random.shuffle(mixValidation)
validationData, validationLabels = zip(*mixValidation)

mixTest = list(zip(testData, testLabels))
random.shuffle(mixTest)
testData, testLabels = zip(*mixTest)

np.savetxt('testData.csv', testData, delimiter=',')
np.savetxt('testLabels.csv', testLabels, delimiter=',')
'''
np.random.seed(randomNumber) # 42 as a seed number is the answer to the great question of "life, the universe and everything" is indeed 42
tf.random.set_seed(randomNumber) # Set random weight and biases so the network produces different result every time

robotNet = keras.Sequential() # Initialising a sequential machine lerning model which accepts and outpust sequences of data

robotNet.add(keras.layers.Dense(len(trainData[0]),input_shape=(len(trainData[0]), ))) # Setting the input layer to 73 neurons
robotNet.add(keras.layers.Dense(len(trainData[0])*4, activation="relu")) # Dense ReLU Layer (hidden layer)
robotNet.add(keras.layers.Dense(len(trainData[0])*2, activation="relu")) # Dense ReLU Layer (hidden layer)
robotNet.add(keras.layers.Dense(4, activation="softmax")) # Dense softmax layer (output layer) prepares the data for probability distribution

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpointPath, # Enabling the checkpoints which same the weight and biases every epoch
    verbose = 1, # Show the training process
    save_weights_only = True, # Setting soe the weights are only saved during checkpoints
    save_freq = int(batchSize/4)) # Frequency of checkpoints
#--------------------------------------------------------------
print("------------------------------------------------------------")
print("************************************************************")
print("-----------------NEURAL NETWORK INFORMATION-----------------")
print("************************************************************")
print("------------------------------------------------------------")

robotNet.build() # Build the network put all the layers together
robotNet.summary() # Gives summerised information on the created Neural Network

#--------------------- Compiling the Neural Network ------------------------
opt = keras.optimizers.Adam(learning_rate = 0.01) # Uses stochastic gradient descent method
robotNet.compile(loss="sparse_categorical_crossentropy", # Calculate loss using crossentropy algorithm
                 optimizer = opt, # Load in the potimiser options
                 metrics=["accuracy"]) # Compile the Neural Net with the desired loss function, optimizer, and metrics

# Sparse_categorical_crossentropy is always used with multiclass classification
# Sgd means stochastic gradient descent, uses a single training example per epoch
# Accuracy is the accuracy of the training
#--------------------------------------------------------------

robotNet.save_weights(checkpointPath.format(epoch = 0)) # Save weights in the directory
# Initialise training process
robotNetHistory = robotNet.fit(trainData, # Load in the training data
                               trainLabels, # Load in the training labels
                               epochs = 30, # Number of runs over the entire datasets through the neural network
                               batch_size = batchSize, # Size of a batch
                               callbacks = [cp_callback], # Saving of weights is called 
                               validation_data = (validationData,validationLabels), # Load in the validation data & labels
                               verbose = 1) # Show no training process

# .fit trains the network
# Number of epochs represent how many times the dataset is passed through the network
# Validation accuracyis displayed using the validation data
#--------------------------------------------------------------
loss, acc = robotNet.evaluate(testData, testLabels, verbose = 2) # Classify the test dataset & return loss and accuracy
labelProbability = robotNet.predict(testData)*100 # Manually classifies a set number of images by ffeding them into the trained netowrk
labelProbabilityDecimal = labelProbability.round() # Probabilities are rounded up to give us a clear answer instead of percentage
predictLabel = np.argmax(robotNet.predict(testData), axis=-1) # Predicts the label of a data
percentage_prob = labelProbability.round(decimals = 1) # Turns the probability to a percentage

for i in range(5): # Displaying first 5 results
    print("------------------------------------------------------------")
    print("***************** Element Number: %d *****************" % (i))
    print("------------------------------------------------------------")
    print("Lables:                           Forward, Left, Right, Backward")
    print("------------------------------------------------------------")
    print("Probabilities for this action is: ", percentage_prob[i])
    print("------------------------------------------------------------")
    print("Decision for this action is: " ,predictLabel[i])
    print("------------------------------------------------------------")
    print("It should be: " ,testLabels[i])
    
performance = (np.sum(predictLabel == testLabels) / len(testLabels))*100 # Calulcuating performance
performance = performance.round(decimals = 2) # Runding it to two decimal places
print("****************************************************************")
print("The Accuracy of this Neural Network on unseen data is: " ,performance,"%")
print("The Evaluation Accuracy on unseen data is: {:5.2f}%".format(100 * acc))
print("****************************************************************")

#robotNet.save("trainedRobotNet")
#os.listdir(checkpointDir) # Find the checpoint directory
#latest = tf.train.latest_checkpoint(checkpointDir) # find latest checpoint
