from __future__ import absolute_import, division, print_function

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
from keras.utils import np_utils

# MNIST dataset parameters.
num_classes = 6 # total classes (0-9 digits).
num_features = 1000 # data features (img shape: 28*28).

# Training parameters.
learning_rate = 0.01
epoch = 100
batch_size = 32

# Network parameters.
n_hidden_1 = 64 # 1st layer number of neurons.
n_hidden_2 = 128 # 2nd layer number of neurons.


data        = np.load('/tmp/171.npz', allow_pickle=True)
dataset  = data['a']
ylabel   = data['b']



encoder = LabelEncoder()
encoder.fit(ylabel)
encoded_Y = encoder.transform(ylabel)

# convert integers to dummy variables (i.e. one hot encoded)

ylabel = np_utils.to_categorical(encoded_Y)
ylabel = np.float32(ylabel)


dataset = pd.DataFrame(dataset)

scaler = StandardScaler()
print(scaler.fit(dataset))

dataset = scaler.transform(dataset)
dataset = np.float32(dataset)



dataset,ylabel= shuffle(dataset,ylabel)

import time
start_time = time.time()
X_train, X_test, Y_train, Y_test = train_test_split(dataset, ylabel, test_size=0.30, random_state=42)

X_valid, X_test, Y_valid, Y_test = train_test_split(X_test, Y_test, test_size=0.50, random_state=42)
display_step       = (len(X_train)/ batch_size)
training_steps = int(display_step) * epoch
train_data      = tf.data.Dataset.from_tensor_slices((X_train,Y_train))
train_data      = train_data.repeat().batch(batch_size).prefetch(1)

random_normal   = tf.initializers.RandomNormal()
weight = {
            'h1' : tf.Variable(random_normal([num_features,n_hidden_1])),
            'h2' : tf.Variable(random_normal([n_hidden_1,n_hidden_2])),
            'out': tf.Variable(random_normal([n_hidden_2,num_classes]))
         }

biase  = {
            'b1' :tf.Variable(tf.zeros([n_hidden_1])),
            'b2' :tf.Variable(tf.zeros([n_hidden_2])),
            'out':tf.Variable(tf.zeros([num_classes]))
        }



# initilize layer with computation
def NeuralNetwork(x):

    layerh1 = tf.add(tf.matmul(x,weight['h1']),biase['b1'])
    layerh1 = tf.nn.relu(layerh1)
    layerh2 = tf.add(tf.matmul(layerh1, weight['h2']), biase['b2'])
    layerh2 = tf.nn.relu(layerh2)
    out_layer = tf.add(tf.matmul(layerh2, weight['out']), biase['out'])
    out_layer = tf.nn.softmax(out_layer)

    return out_layer
def crossentropy(ypred,ytrue):
    ypred = tf.clip_by_value(ypred, 1e-9, 1.)
    return tf.reduce_mean(-tf.reduce_sum(ytrue * tf.math.log(ypred)))
optimizer  = tf.optimizers.Adam(learning_rate)

def run_optimization(x_batch,y_batch):

    with tf.GradientTape() as tap:
        pred = NeuralNetwork(x_batch)
        loss = crossentropy(pred,y_batch)

    learnvariable = list(weight.values()) + list(biase.values())
    grad          = tap.gradient(loss,learnvariable)
    optimizer.apply_gradients(zip(grad,learnvariable))
0


def accuracy(pred,ytrue):
    ytrue = np.array(ytrue)
    pred  = np.array(pred)
    pred = tf.argmax(pred,1)
    prediction = [ytrue[j][pred[j]] for j in range(len(pred))]
    return tf.reduce_mean(tf.cast(prediction, tf.float32), axis=-1)

for step,(x_batch,y_batch) in enumerate(train_data.take(int(training_steps)),1):
    run_optimization(x_batch,y_batch)
    if step%int(display_step) == 0:
        pred     = NeuralNetwork(x_batch)
        loss     = crossentropy(pred,y_batch)
        acc      = accuracy(pred,y_batch)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))

print("--- %s seconds ---" % (time.time() - start_time))
