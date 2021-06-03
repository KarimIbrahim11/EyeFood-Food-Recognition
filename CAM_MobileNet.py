# -*- coding: utf-8 -*-
"""06_grad_cam.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13Hg_JMz_3BTzRTpNwhHM5Pz68MN6yndY

<a href="https://colab.research.google.com/drive/13Hg_JMz_3BTzRTpNwhHM5Pz68MN6yndY?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

By [Ibrahim Sobh](https://www.linkedin.com/in/ibrahim-sobh-phd-8681757/)

## CAM 
**Class Activation Map (CAM)** visualization techniques produce heatmaps of 2D class activation over input images, showing how important each location is for  considered class. 

## Grad-CAM
In the paper [Grad-CAM: Why did you say that? Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391), the visualization is conducted by taking the output feature map of a convolution layer (given an input image), and then weighing every channel (feature map) by the gradient of the output class wrt the feature map.

![Grad-Cam](https://www.mathworks.com/help/examples/nnet/win64/GradCAMRevealsTheWhyBehindDeepLearningDecisionsExample_02.png)
"""

from keras.applications.vgg16 import VGG16
import tensorflow as tf
# from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt
import cv2

# load the model
from keras.models import Model, load_model

tf.compat.v1.disable_eager_execution()
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.applications import InceptionV3, VGG16, ResNet50, MobileNetV2, NASNetMobile
from tensorflow.keras.applications import NASNetLarge, InceptionResNetV2, DenseNet121
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
import numpy as np

from matplotlib import pyplot as plt

# Predicted values
food_list = []
fileReader = open('../food-101/meta/labels.txt', 'r')
for line in fileReader.readlines():
    food_list.append(line.rstrip())
fileReader.close()

K.clear_session()

# food_list = ['falafel', 'pizza', 'omelette', 'hamburger', 'sushi']

base_model = MobileNetV2(weights='imagenet', include_top=False)
model_name = 'MobileNetV2'

# """### Add new top layers to the selected model"""

n_classes = 101

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)

predictions = Dense(n_classes,
                    kernel_regularizer=regularizers.l2(0.005),
                    activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model_best = load_model("weights-improvement-41-0.82.hdf5", compile=False)
model = model_best

# image load and resize
# img_path = '1_129.jpg'
# img_path = 'dogo.jpg'
img_path = 'twodishes.jpg'
img = image.load_img(img_path, target_size=(299, 299))
plt.imshow(img)

# image preprocess
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# predict (just fo sainty)
preds = model.predict(x)
# print(decode_predictions(preds, top=3)[0])
cls_id = np.argmax(preds[0])
print(cls_id)  # the output index

output_point = model.output[:, cls_id]

# Select the last conv layer

feature_maps = model.get_layer('Conv_1')
# feature_maps = model.layers[2]

# grads of the output wrt the conv layer
grads = K.gradients(output_point, feature_maps.output)[0]

# grads has the same shape as the selected conv layer
# the weight of each feature map is simple the mean of the grads
mean_grads = K.mean(grads, axis=(0, 1, 2))

# the function
iterate = K.function([model.input], [mean_grads, feature_maps.output[0]])

mean_grads_value, feature_maps_value = iterate([x])

# sample feature map number 200
feature_map = feature_maps_value[:, :, 200].copy()
feature_map -= feature_map.mean()
feature_map /= feature_map.std()
feature_map *= 64
feature_map += 128
feature_map = np.clip(feature_map, 0, 255).astype('uint8')
plt.imshow(feature_map)

feature_map.min(), feature_map.max(), feature_map.mean()

# mean of all feature maps (without weighting)
plt.imshow(np.mean(feature_maps_value, axis=-1))

# weight each feature map by its mean gradinet
for i in range(len(feature_maps_value)):
    feature_maps_value[:, :, i] *= mean_grads_value[i]

# The heatmap the mean of all weighted maps
heatmap = np.mean(feature_maps_value, axis=-1)

heatmap.min(), heatmap.max(), heatmap.mean()

heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
print(heatmap.min(), heatmap.max(), heatmap.mean())
plt.matshow(heatmap)

# overlay 
# img = cv2.imread(img_path)
img = image.load_img(img_path)
# heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = cv2.resize(heatmap, (np.array(img).shape[1], np.array(img).shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_RAINBOW)
#heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
print(heatmap.min(), heatmap.max(), heatmap.mean())
plt.imshow(heatmap)
plt.axis('off')
plt.title('heatmap')
plt.show()

img = image.load_img(img_path)
plt.imshow(img)
plt.show()
superimposed_img = heatmap * 400 + img
plt.imshow(superimposed_img)
plt.axis('off')
plt.title('superimposed heatmap')
plt.show()
