from __future__ import absolute_import, division, print_function

import pandas as pd
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Import TensorFlow v2.
import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np

import random
#tf = tf.compat.v1
no_classes     = 34
timesteps      = 12
training_steps = 816 * 20
batch_size     = 32
data           = np.load('/tmp/124.npz')
dataset        = data['a']
ylabel         = data['b']
feature_num    = len(dataset[1][:])
dataset        = pd.DataFrame(dataset)
scaler         = StandardScaler()
print(scaler.fit(dataset))

dataset = scaler.transform(dataset)
dataset = np.float64(dataset)

encoder = LabelEncoder()
encoder.fit(ylabel)
encoded_Y = encoder.transform(ylabel)


# convert integers to dummy variables (i.e. one hot encoded)


ylabel = np_utils.to_categorical(encoded_Y)
ylabel = np.int8(ylabel)
import time
start_time = time.time()
##########################
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

dataset,ylabel= shuffle(dataset,ylabel)

X_train, X_test, Y_train, Y_test = train_test_split(dataset, ylabel, test_size=0.20, random_state=42)

X_valid, X_test, Y_valid, Y_test = train_test_split(X_test, Y_test, test_size=0.50, random_state=42)

'''
training_size = 2176
testsize = 272
valsize  = 272
xtrain   = X_train.reshape(training_size,timesteps,feature_num)
ytrain   = Y_train.reshape(training_size,timesteps,no_classes)
xtest    = X_test.reshape(testsize,timesteps,feature_num)
ytest    = Y_test.reshape(testsize,timesteps,no_classes)
xval     = X_valid.reshape(valsize,timesteps,feature_num)
yval     = Y_valid.reshape(valsize,timesteps,no_classes)
epoch    = np.int(training_size/batch_size)
'''
X_train  = X_train.reshape(X_train.shape[0], 501, 1).astype('float32')
X_valid  = X_valid.reshape(X_valid.shape[0], 501, 1).astype('float32')
testX    = X_test.reshape(X_test.shape[0], 501, 1).astype('float32')

epoch    = 816 #np.int(/batch_size)

# LSTM
############################

train_data = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
train_data = train_data.repeat().shuffle(500).batch(batch_size).prefetch(1)

### to do

############################

# Create LSTM Model.
class OneDCNN(Model):
    # Set layers.
    def __init__(self):
        super(OneDCNN, self).__init__()
        # Define a Masking Layer with -1 as mask.
        # Define a LSTM layer to be applied over the Masking layer.
        # Dynamic computation will automatically be performed to ignore -1 values.
        self.cnn    = layers.Conv1D(filters=8, kernel_size=64,strides=12, input_shape=(501, 1))
        self.flater = layers.Flatten()
        #self.lstm = layers.LSTM(units=128,return_sequences=True)
        # Output fully connected layer (2 classes: linear or random seq).
        self.out = layers.Dense(34,activation='softmax')

    # Set forward pass.
    def call(self, x, is_training=False):
        # A RNN Layer expects a 3-dim input (batch_size, seq_len, num_features).
        #x = tf.re
        # GHUFRshape(x, shape=[-1, 12, 1])
        # Apply Masking layer.
      #  x = self.masking(x)
        # Apply LSTM layer.
        x = self.cnn(x)
        x_flat = self.flater(x)
        # Apply output layer.
        x = self.out(x_flat)

        return x

# Build LSTM model.
cnn1d = OneDCNN()


# Cross-Entropy Loss.
# Note that this will apply 'softmax' to the logits.


def cross_entropy_loss(x, y):
    # Convert labels to int 64 for tf cross-entropy function.
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)

    ypred = tf.clip_by_value(x, 1e-9, 1.)

    return tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(ypred)))

# Accuracy metric.
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    '''
    y_pred = np.moveaxis(y_pred, 1, 0)
    y_true = np.moveaxis(y_true, 1, 0)

    correct_pred = []
    for ind in range(timesteps):
        correct_pred.append(
            tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred[ind], 1), tf.argmax(y_true[ind], 1)), tf.float32)).numpy())
    '''

    ytrue = np.array(y_true)
    pred = np.array(y_pred)
    pred = tf.argmax(pred, 1)
    prediction = [ytrue[j][pred[j]] for j in range(len(pred))]
    return tf.reduce_mean(tf.cast(prediction, tf.float32), axis=-1)

# Adam optimizer.
optimizer = tf.optimizers.Adam(0.001)

# Optimization process.
def run_optimization(x, y):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        # Forward pass.
        pred = cnn1d(x, is_training=True)
        # Compute loss.
        loss = cross_entropy_loss(pred, y)

    # Variables to update, i.e. trainable variables.
    trainable_variables = cnn1d.trainable_variables

    # Compute gradients.
    gradients = g.gradient(loss, trainable_variables)

    # Update weights following gradients.
    optimizer.apply_gradients(zip(gradients, trainable_variables))

display_step = 1
pred=0

graph_training = []
graph_val      = []
counter = 0
# Run training for the given number of steps.
for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    # Run the optimization to update W and b values.
    run_optimization(batch_x, batch_y)

    if step%epoch == 0 or step == 1:
        counter = counter + 1
        pred = cnn1d(batch_x, is_training=True)
        loss = cross_entropy_loss(pred, batch_y)
        acc = accuracy(pred, batch_y)
        ### for validation
        val_pred = cnn1d(X_valid, is_training=False)
        val_loss = cross_entropy_loss(val_pred, Y_valid)
        val_acc  = accuracy(val_pred, Y_valid)

        graph_training.append(acc.numpy())
        graph_val.append(val_acc.numpy())
        print("epoch: %i, Training loss: %f, accuracy: %f"% (counter, loss, np.max(acc)))
        print("epoch: %i, Validation loss: %f, accuracy: %f"%   (counter,val_loss, np.max(val_acc)))

test_pred = cnn1d(testX, is_training=False)
test_loss = cross_entropy_loss(test_pred, Y_test)
test_acc  = accuracy(test_pred, Y_test)
print("Test loss: %f, accuracy: %f" % (test_loss, np.max(test_acc)))



# for graph Generation



plt.plot(range(1,counter+1),np.array(graph_training))
plt.locator_params(integer=True)
plt.plot(range(1,counter+1),np.array(graph_val))
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','valid'], loc='upper left')
plt.show()
