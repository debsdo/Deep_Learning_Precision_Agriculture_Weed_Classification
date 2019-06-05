#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function


# In[ ]:


#Pandas and Numpy for data structures and utils
import numpy as np
import scipy as sp
import pandas as pd
from numpy.random import rand

#sklearn imports for metrics
from sklearn import preprocessing
from sklearn.metrics import auc, precision_recall_curve, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

#Matplotlib imports for graphs
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[2]:


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
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.models import load_model


# In[ ]:


import os

os.chdir('')
train_data_DIR = ''
test_data_DIR = ''

#Load and Prepare Data Set
BATCH_SIZE = 16
EPOCHS = 50
steps_per_epoch = 100
NUM_CLASSES = 9
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
nb_train_samples = 2700
nb_validation_samples = 270


# In[4]:


from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

vgg16_model = keras.applications.vgg16.VGG16()
vgg16_model.summary()


# In[5]:


model = Sequential()

for layer in vgg16_model.layers[:-1]:
    model.add(layer)


# In[6]:


model.summary()


# In[7]:


for layer in model.layers:
    layer.trainable = False


# In[9]:


model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(9, activation = "sigmoid")) #as there are 9 classes
model.summary()


# In[20]:


#import pydot
#import graphviz
#pydot.Dot.create(pydot.Dot())
#from IPython.display import SVG
#from keras.utils.vis_utils import model_to_dot

#SVG(model_to_dot(vgg16_model).create(prog='dot', format='svg'))


# In[ ]:


opt = Adam(lr=0.001)

model.compile(optimizer = opt, loss = "categorical_crossentropy", metrics = ["accuracy"])


# In[ ]:


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
    class_mode = 'binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size = (img_height, img_width),
    batch_size = batch_size,
    class_mode = 'binary')


# In[ ]:


model.fit_generator( train_generator, samples_per_epoch=nb_train_samples, epochs=epochs,
                    validation_data=validation_generator,
                    nb_val_samples=nb_validation_samples)


# In[ ]:


history = vgg16_model.fit(x, y, validation_split=0.25, epochs=50, batch_size=16, verbose=1)

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


# In[ ]:


vgg16_model.predict(x, batch_size=None, verbose=0, steps=None, callbacks=None)

