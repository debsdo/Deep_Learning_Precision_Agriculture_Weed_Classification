#XCEPTION MODEL

#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function


# In[2]:

import numpy as np
import scipy as sp
import pandas as pd
import h5py
import tqdm
from numpy.random import rand

#sklearn imports for metrics
from sklearn import preprocessing
from sklearn.metrics import auc, precision_recall_curve, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

#Matplotlib imports for graphs
import matplotlib.pyplot as plt

# In[3]:

import tensorflow as tf
import keras

# Models
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

# Layers
from keras.layers import Dense, Activation, Flatten, Dropout, GlobalMaxPooling2D,BatchNormalization
from keras import backend as K

# Other
from keras import optimizers
from keras import losses
from keras.optimizers import SGD, Adam, Adagrad
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.models import load_model


# In[4]:

train_data_dir = 'Deb_rice_soil/train'
validation_data_dir = 'Deb_rice_soil/test'
nb_train_samples = 3882
nb_validation_samples = 600
epochs = 50
batch_size = 16
img_width, img_height = 224, 224

# In[13]:


train_datagen = ImageDataGenerator(
    rescale = 1. / 255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale =1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size = (img_height, img_width),
    batch_size = batch_size,
    class_mode = 'categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size = (img_height, img_width),
    batch_size = batch_size,
    class_mode = 'categorical')

# In[6]:


from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input

base_model = Xception(weights='imagenet', include_top=False)
base_model.summary()


# In[8]:


for layer in base_model.layers:
    layer.trainable = False

# In[9]:


x = base_model.output
x = GlobalMaxPooling2D()(x)
x = Dense(256, activation = "relu")(x)
x = Dropout(0.5)(x)
output = Dense(3, activation = "sigmoid")(x)
new_model = Model(inputs = base_model.input, outputs = output)
new_model.summary()

# In[10]:



new_model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])


# In[14]:

filepath = '/checkpt/checkpt_best_xcep-{epoch:02d}-{val_acc:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor = 'val_acc', verbose = 1, save_best_only = True, mode = 'max')

# In[15]:

cb_early = EarlyStopping(monitor = 'val_acc', mode='auto', verbose = 1, patience = 2, baseline = 0.985, restore_best_weights = True)
 
# In[16]:


history = new_model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples,
    callbacks = [checkpoint, cb_early])

new_model.save('debxcep.h5')

# In[17]:

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()




