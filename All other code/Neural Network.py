import os
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Dense, Flatten, Softmax, Conv2D, Dropout, MaxPooling2D

print("Preparing Dataset...")
# Download dataset from internet (if not present)
mnist = tf.keras.datasets.mnist.load_data()
(x_train, y_train), (x_test, y_test) = mnist

# CONSTANTS
HEIGHT, WIDTH = x_train[0].shape
NCLASSES = tf.size(tf.unique(y_train).y)
BUFFER_SIZE = 5000
BATCH_SIZE = 100
NUM_EPOCHS = 10
STEPS_PER_EPOCH = 100


def scale(image, label):
    """scale pixel value from 0~255 to 0~1"""
    image = tf.cast(image, tf.float32)
    image /= 255
    image = tf.expand_dims(image, -1)
    return image, label

def load_dataset(training=True):
    """Loads MNIST dataset into a tf.data.Dataset"""
    (x_train, y_train), (x_test, y_test) = mnist
    x = x_train if training else x_test
    y = y_train if training else y_test
    # One-hot encode the classes
    y = tf.keras.utils.to_categorical(y, NCLASSES)
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(scale).batch(BATCH_SIZE)
    if training:
        dataset = dataset.shuffle(BUFFER_SIZE).repeat()
    return dataset

print("Building Model...")
# configure CNN 
model = Sequential([
        Conv2D(64, kernel_size=3,
                activation='relu', input_shape=(WIDTH, HEIGHT, 1)),
        MaxPooling2D(2),
        Conv2D(32, kernel_size=3,
                activation='relu'),
        MaxPooling2D(2),
        Flatten(),
        Dense(400, activation='relu'),
        Dense(100, activation='relu'),
        Dropout(.25),
        Dense(10),
        Softmax()
    ])

model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

print("Loading Data...")
# Load train and validation datasets    
train_data = load_dataset()
validation_data = load_dataset(training=False)

# save model and tensorboard data
OUTDIR = "mnist_digits/"
checkpoint_callback = ModelCheckpoint(
    OUTDIR, save_weights_only=True, verbose=1)
tensorboard_callback = TensorBoard(log_dir=OUTDIR)

# Train
print("Training...")
t1 = time.perf_counter()
history = model.fit(
    train_data, 
    validation_data=validation_data,
    epochs=NUM_EPOCHS, 
    steps_per_epoch=STEPS_PER_EPOCH,
    verbose=2,
    callbacks=[checkpoint_callback, tensorboard_callback]
)
t2 = time.perf_counter()
print("training took: {:4.4f} secs.".format(t2 - t1))