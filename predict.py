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
fileReader = open('food-101/meta/labels.txt', 'r')
for line in fileReader.readlines():
    food_list.append(line.rstrip())
fileReader.close()


def predict_class(model, images, show=True):
    for img in images:
        img = image.load_img(img, target_size=(299, 299))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img /= 255.

        pred = model.predict(img)
        index = np.argmax(pred)
        food_list.sort()
        pred_value = food_list[index]
        if show:
            plt.imshow(img[0])
            plt.axis('off')
            plt.title(pred_value)
            plt.show()


K.clear_session()

# food_list = ['falafel', 'pizza', 'omelette', 'hamburger', 'sushi']

base_model = MobileNetV2(weights='imagenet', include_top=False)
model_name = 'MobileNetV2'
# epoch_num = 70

"""### Add new top layers to the selected model"""

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

images = []
images.append('sushii.jpg')
images.append('omlette.jpg')
predict_class(model_best, images, True)
