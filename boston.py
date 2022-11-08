#!/usr/bin/env python
# coding: utf-8

# 1. Run BOSTON
# 2. Notice that the mini-batch size is set to 1. Experiment with different mini-batch sizes. What do you observe? Can you account for your observation?
# 2. Run a series of experiments to find the best model. (Hint: look back at the previous labs.)
# 3. Retrain the best model on all the training data and evaluate on the test data

# In[4]:


# BOSTON

from tensorflow.keras.datasets import boston_housing
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# feature normalize
mean = train_data.mean(axis = 0)
train_data -= mean # shift
std = train_data.std(axis = 0)
train_data /= std # rescale

test_data -= mean
test_data /= std

from tensorflow.keras import models
from tensorflow.keras import layers

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, 
                           activation = 'relu', 
                           input_shape = (train_data.shape[1], )))
    model.add(layers.Dense(64, activation = 'relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer = 'rmsprop', loss = 'mse', metrics = ['mae'])
    
    return model


# In[5]:


import tensorflow as tf

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        c = ['\b|', '\b/', '\b-', '\b\\'] 
        print(c[epoch % 4], end='')
    def on_epoch_end(self, epoch, logs=None):
        print('\b', end='')


# In[6]:


#%%time
import numpy as np, tensorflow as tf

# K-fold validation
K = 4
num_val_samples = len(train_data) // K
num_epochs = 500
all_mae_histories = []

for i in range(K):
    print('processing fold', i)
    
    print('\t validation data')
    # Prepare the validation data: data from partition i
    a, b = i * num_val_samples, (i + 1) * num_val_samples
    val_data = train_data[a : b]
    val_targets = train_targets[a : b]
    
    print('\t training data')
    # Prepare the training data: data from all other partitions
    partial_train_data = np.concatenate([train_data[:a], train_data[b:]], axis=0)
    partial_train_targets = np.concatenate([train_targets[:a], train_targets[b:]], axis=0)

    print('\t model')    
    # Build the Keras model (already compiled)
    model = build_model()
    
    print('\t history')
    # Train the model (in silent mode, verbose=0)
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0, 
                        callbacks=[CustomCallback()])

    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)

