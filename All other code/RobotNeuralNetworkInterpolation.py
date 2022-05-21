import csv
import tensorflow as tf # Tensorflow Neural Network
from tensorflow import keras # Keras runs on top of tf deep learning framework, has simple API's makes user interaction simple
import matplotlib.pyplot as plt # Plot library
import numpy as np # Mathematical computing
import random
import os
import pandas as pd

def multiples(multi, f):
    return [multi * i for i in range(1, f + 1)]

checkpointPath = "training_Interpolated_test/cp-{epoch:04d}.ckpt"
checkpointDir = os.path.dirname(checkpointPath)

batchSize = 64

angles = []
distances = []
labels = []
finalDistances = []

#dataSet = open('dataSetFinal.csv')
dataSet = open('dataSetFinal2_CombinedBack.csv', 'r')
################# Reading the data
with dataSet as File:  
    reader = csv.reader(File)
    for row in reader:
        angles.append(int(row[0]))
        distances.append(float(row[1]))
        if row[2] != '':
            labels.append(int(row[2]))
            
multiple = angles[1] - angles[0]
lidarResolution = 360
fin = int(lidarResolution/multiple)
scanSize = int((lidarResolution/multiple) + 1)

roughDistances = []
################# Retrieving final distances for 0 -> 360
for i in range(int(len(distances)/scanSize)):
    roughDistances.append(distances[scanSize*i:scanSize*(i+1)])

for j in roughDistances:
    interpol = pd.Series(j)
    interpol.replace(0,np.NaN, inplace=True)
    x = interpol.interpolate(method = 'linear', limit_direction = 'both')
    #x = interp1d(interpol, kind='cubic')
    x = list(np.around(np.array(x),2))
    finalDistances.append(x)
'''
z = pd.isna(finalDistances)
countNaN = 0
NaNindx = []
for d in range(len(z)):
    if any(z[d]) == True:
        NaNindx.append(d)
        countNaN += 1
print(countNaN)
'''
angleResolution = multiples(multiple, fin)
angleResolution = [0] + angleResolution

randomNumber = 42 # Weights and biases
trainRatio = 80
validationRatio = 10
testRatio = 10

trainData = []
trainLabels = []
validationData = []
validationLabels = []
testData = []
testLabels = []
    
forwardIndx = []
leftIndx = []
rightIndx = []
backwardIndx = []
################# Counting the amount of forward, left, right and backward data there is
for k in range(len(labels)):
    if labels[k] == 0:
        forwardIndx.append(k)
    elif labels[k] == 1:
        leftIndx.append(k)
    elif labels[k] == 2:
        rightIndx.append(k)
    elif labels[k] == 4:
        backwardIndx.append(k)
################# Balancing the data to right turns as it is the shortest 
random.shuffle(forwardIndx)
random.shuffle(leftIndx)
random.shuffle(rightIndx)
shortestArray = min(forwardIndx,leftIndx,rightIndx, key=len)

balancingLength = len(shortestArray)

forwardIndx = forwardIndx[0:balancingLength]
leftIndx = leftIndx[0:balancingLength]
rightIndx = rightIndx[0:balancingLength]
################# Extracting the balanced data for training dataset
extractLeftIndx = random.sample(leftIndx,k=round(len(leftIndx)*(trainRatio/100)))
extractRightIndx = random.sample(rightIndx,k=round(len(rightIndx)*(trainRatio/100)))
extractForwardIndx = random.sample(forwardIndx,k=round(len(forwardIndx)*(trainRatio/100)))

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
extractLeftIndx_test = random.sample(leftIndx,k=round(len(leftIndx)*(50/100)))
extractRightIndx_test = random.sample(rightIndx,k=round(len(rightIndx)*(50/100)))
extractForwardIndx_test = random.sample(forwardIndx,k=round(len(forwardIndx)*(50/100)))

for i in extractLeftIndx_test[:]:
        if i in leftIndx:
            leftIndx.remove(i)
for i in extractRightIndx_test[:]:
        if i in rightIndx:
            rightIndx.remove(i)
for i in extractForwardIndx_test[:]:
        if i in forwardIndx:
            forwardIndx.remove(i)
################# The rest is for validation dataset and then the labels and data is inserted into their own data structures

for z in range(len(extractLeftIndx)):
    trainData.append(finalDistances[extractLeftIndx[z]])
    trainLabels.append(labels[extractLeftIndx[z]])
    if z < len(leftIndx):
        validationData.append(finalDistances[leftIndx[z]])
        validationLabels.append(labels[leftIndx[z]])
    if z < len(extractLeftIndx_test):
        testData.append(finalDistances[extractLeftIndx_test[z]])
        testLabels.append(labels[extractLeftIndx_test[z]])
          
for z in range(len(extractRightIndx)):
    trainData.append(finalDistances[extractRightIndx[z]])
    trainLabels.append(labels[extractRightIndx[z]])
    if z < len(rightIndx):
        validationData.append(finalDistances[rightIndx[z]])
        validationLabels.append(labels[rightIndx[z]])
    if z < len(extractRightIndx_test):
        testData.append(finalDistances[extractRightIndx_test[z]])
        testLabels.append(labels[extractRightIndx_test[z]])
        
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
mixTrain = list(zip(trainData, trainLabels))
random.shuffle(mixTrain)
trainData, trainLabels = zip(*mixTrain)

mixValidation = list(zip(validationData, validationLabels))
random.shuffle(mixValidation)
validationData, validationLabels = zip(*mixValidation)

mixTest = list(zip(testData, testLabels))
random.shuffle(mixTest)
testData, testLabels = zip(*mixTest)
'''
np.savetxt('testData.csv', testData, delimiter=',')
np.savetxt('testLabels.csv', testLabels, delimiter=',')

np.random.seed(randomNumber)
tf.random.set_seed(randomNumber)

robotNet = keras.Sequential()

robotNet.add(keras.layers.Dense(len(trainData[0]),input_shape=(len(trainData[0]), )))
robotNet.add(keras.layers.Dense(len(trainData[0])*4, activation="relu")) # Dense ReLU Layer (hidden layer)
robotNet.add(keras.layers.Dense(len(trainData[0])*2, activation="relu")) # Dense ReLU Layer (hidden layer)
robotNet.add(keras.layers.Dense(4, activation="softmax")) # Dense softmax layer (output layer) prepares the data for probability distribution

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpointPath, 
    verbose=1, 
    save_weights_only=True,
    save_freq=int(batchSize/4))
#--------------------------------------------------------------
print("------------------------------------------------------------")
print("************************************************************")
print("-----------------NEURAL NETWORK INFORMATION-----------------")
print("************************************************************")
print("------------------------------------------------------------")

robotNet.build()
robotNet.summary() # Gives summerised information on the created Neural Network

#--------------------- Compiling the Neural Network ------------------------
opt = keras.optimizers.Adam(learning_rate=0.01)
robotNet.compile(loss="sparse_categorical_crossentropy",
                 optimizer=opt,
                 metrics=["accuracy"]) # Compile the Neural Net with the desired loss function, optimizer, and metrics

# Sparse_categorical_crossentropy is always used with multiclass classification
# Sgd means stochastic gradient descent, uses a single training example per epoch
# Accuracy is the accuracy of the training
#--------------------------------------------------------------

robotNet.save_weights(checkpointPath.format(epoch=0))

robotNetHistory = robotNet.fit(trainData,
                               trainLabels,
                               epochs=30,
                               batch_size=batchSize,
                               callbacks=[cp_callback],
                               validation_data = (validationData,validationLabels),
                               verbose=0)

# .fit trains the network
# Number of epochs represent how many times the dataset is passed through the network
# Validation accuracyis displayed using the validation data
#--------------------------------------------------------------
loss, acc = robotNet.evaluate(testData, testLabels, verbose=2)
labelProbability = robotNet.predict(testData)*100 # Manually classifies a set number of images by ffeding them into the trained netowrk
labelProbabilityDecimal = labelProbability.round() # Probabilities are rounded up to give us a clear answer instead of percentage
predictLabel = np.argmax(robotNet.predict(testData), axis=-1)
percentage_prob = labelProbability.round(decimals = 1)

for i in range(5): # Displaying results
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
    
performance = (np.sum(predictLabel == testLabels) / len(testLabels))*100
performance = performance.round(decimals=2)
print("****************************************************************")
print("The Accuracy of this Neural Network on unseen data is: " ,performance,"%")
print("The Evaluation Accuracy on unseen data is: {:5.2f}%".format(100 * acc))
print("****************************************************************")

#robotNet.save("trainedRobotNet")
os.listdir(checkpointDir)
latest = tf.train.latest_checkpoint(checkpointDir) # Latest checpoint
'''
trainedNet = keras.Sequential()

trainedNet.add(keras.layers.Dense(len(trainData[0]),input_shape=(len(trainData[0]), )))
trainedNet.add(keras.layers.Dense(len(trainData[0])*4, activation="relu")) # Dense ReLU Layer (hidden layer)
trainedNet.add(keras.layers.Dense(len(trainData[0])*2, activation="relu")) # Dense ReLU Layer (hidden layer)
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
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

'''
