import tensorflow as tf # Tensorflow Neural Network
from tensorflow import keras # Keras runs on top of tf deep learning framework, has simple API's makes user interaction simple
import matplotlib.pyplot as plt # Plot library
import numpy as np # Mathematical computing
import warnings
warnings.filterwarnings("ignore")

plt.close()

RANDOM_NUMBER = 42 # Weights and biases
IMAGE_RESOLUTION = 28 # Images are 28 x 28
TRAIN_RATIO = 5000 # Train dataset split
VALIDATION_RATIO = 5000 # Validation dataset split

#--------------------- Data Management ------------------------
mnist = keras.datasets.mnist # Loading in the mnist dataset
(trainImages_full, trainLabels_full),(testImages, testLabels) = mnist.load_data() # Extracting the training and test data

trainImages_norm = trainImages_full/255. # Data normalisation from 0->1 to 0->255
testImages_norm = testImages/255.

validationImages, trainImages = trainImages_norm[:VALIDATION_RATIO], trainImages_norm[TRAIN_RATIO:] # Extracting the validation data 5000 samples from the training data
validationLabels, trainLabels = trainLabels_full[:VALIDATION_RATIO], trainLabels_full[TRAIN_RATIO:]

# 55000 for training
# 10000 for testing
# 5000 for validation

# x_ variables are images
# y_ variables are lables
# Lables: 0 1 2 3 4 5 6 7 8 9

testImages = testImages_norm # Both are normalised data
#--------------------------------------------------------------

#--------------------- Setting up the Neural Network ------------------------
np.random.seed(RANDOM_NUMBER) # Initialising random weights and biases
tf.random.set_seed(RANDOM_NUMBER) # Very widely used value is 42 in the machine learning world

imgNet = keras.models.Sequential() # Sequential stack Neural Network

imgNet.add(keras.layers.Flatten(input_shape=[IMAGE_RESOLUTION,IMAGE_RESOLUTION])) # 1 Flatten layer as (input layer) fingle array prepares data for the following layers
imgNet.add(keras.layers.Dense(300, activation="relu")) # 2 Dense ReLU Layers (hidden layers)
imgNet.add(keras.layers.Dense(100, activation="relu"))
imgNet.add(keras.layers.Dense(10, activation="softmax")) # Dense softmax layer (output layer) prepares the data for probability distribution
#--------------------------------------------------------------
print("------------------------------------------------------------")
print("************************************************************")
print("-----------------NEURAL NETWORK INFORMATION-----------------")
print("************************************************************")
print("------------------------------------------------------------")

imgNet.summary() # Gives summerised information on the created Neural Network

#--------------------- Compiling the Neural Network ------------------------
imgNet.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"]) # Compile the Neural Net with the desired loss function, optimizer, and metrics

# Sparse_categorical_crossentropy is always used with multiclass classification
# Sgd means stochastic gradient descent, uses a single training example per epoch
# Accuracy is the accuracy of the training
#--------------------------------------------------------------

#--------------------- Training the Neural Network ------------------------
imgNet_history = imgNet.fit(trainImages,trainLabels,epochs=3,validation_data=(validationImages,validationLabels))

# .fit trains the network
# Number of epochs represent how many times the dataset is passed through the network
# Validation accuracyis displayed using the validation data
#--------------------------------------------------------------

#--------------------- Testing the Neural Network ------------------------
imgNet.evaluate(testImages,testLabels) # Classifies the test data on the trained network
imagesSample = testImages[:5] # Take 5 sample images from the evaluation
labelProbability = imgNet.predict(imagesSample)*100 # Manually classifies a set number of images by ffeding them into the trained netowrk
labelProbabilityDecimal = labelProbability.round() # Probabilities are rounded up to give us a clear answer instead of percentage
predictLabel = np.argmax(imgNet.predict(imagesSample), axis=-1)

fig = plt.figure(figsize=(25, 25)) # Setting up figure for plotting the images

percentage_prob = labelProbability.round(decimals = 1)

for i in range(5): # Displaying results
    print("------------------------------------------------------------")
    print("***************** Element Number: %d *****************" % (i))
    print("------------------------------------------------------------")
    print("Lables: 0 1 2 3 4 5 6 7 8 9")
    print("------------------------------------------------------------")
    print("Probabilities for this image: ", percentage_prob[i])
    print("------------------------------------------------------------")
    print("Rounded Probability for this image is: " ,labelProbabilityDecimal[i])
    print("------------------------------------------------------------")
    print("Decision for this image is: " ,predictLabel[i])
    print("------------------------------------------------------------")
    print("It should be: " ,testLabels[i])
    
for x in range(4): # Plotting the images
    fig.add_subplot(2,2,x+1) # Printing the first 4 images from that array onto a plot
    plt.imshow(testImages[x])
plt.show()
#--------------------------------------------------------------