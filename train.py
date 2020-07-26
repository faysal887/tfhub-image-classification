#!/usr/bin/env python
# coding: utf-8

# !pip install -q -U tf-hub-nightly
# !pip install -q tfds-nightly

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageOps
import tensorflow as tf
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import itertools
import tensorflow_hub as hub
from keras.preprocessing import image
from datetime import datetime
import pandas as pd
import time


print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")

n_epochs=int(sys.argv[1])

data_dir='./assets/'



module_selection = ("mobilenet_v2_100_224", 224) 
handle_base, pixels = module_selection
MODULE_HANDLE ="https://tfhub.dev/google/imagenet/{}/feature_vector/4".format(handle_base)
IMAGE_SIZE = (pixels, pixels)
print("Using {} with input size {}".format(MODULE_HANDLE, IMAGE_SIZE))

BATCH_SIZE = 32 



datagen_kwargs = dict(rescale=1./255, validation_split=.20)
dataflow_kwargs = dict(target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
                   interpolation="bilinear")

valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    **datagen_kwargs)
valid_generator = valid_datagen.flow_from_directory(
    data_dir, subset="validation", shuffle=False, **dataflow_kwargs)

do_data_augmentation = False 
if do_data_augmentation:
  train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
      rotation_range=40,
      horizontal_flip=True,
      width_shift_range=0.2, height_shift_range=0.2,
      shear_range=0.2, zoom_range=0.2,
      **datagen_kwargs)
else:
  train_datagen = valid_datagen
train_generator = train_datagen.flow_from_directory(
    data_dir, subset="training", shuffle=True, **dataflow_kwargs)



do_fine_tuning = False 




print("Building model with", MODULE_HANDLE)
model = tf.keras.Sequential([
    # Explicitly define the input shape so the model can be properly
    # loaded by the TFLiteConverter
    tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
    hub.KerasLayer(MODULE_HANDLE, trainable=do_fine_tuning),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(train_generator.num_classes,
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001))
])
model.build((None,)+IMAGE_SIZE+(3,))
model.summary()




model.compile(
  optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9), 
  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
  metrics=['accuracy'])





steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = valid_generator.samples // valid_generator.batch_size
hist = model.fit(
    train_generator,
    epochs=n_epochs, steps_per_epoch=steps_per_epoch,
    validation_data=valid_generator,
    validation_steps=validation_steps).history

model_dir='models'
os.makedirs(model_dir,exist_ok=True)
# dt=datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
# dt=str(time.time()).replace('.','')


hist['is_default']=False

try:
    df=pd.read_csv(f'{model_dir}/models_hist.csv').reset_index(drop=True)
    hist['model_name']='model'+'_'+str(df.index.values.max()+1)
    models_hist=pd.concat([df, pd.DataFrame(hist)]).reset_index(drop=True)

except:
    hist['model_name']='model_0'
    models_hist=pd.DataFrame(hist).reset_index(drop=True)
    
    
model.save(f'{model_dir}/{hist['model_name']}')
models_hist.to_csv(f'{model_dir}/models_hist.csv', index=False)


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

