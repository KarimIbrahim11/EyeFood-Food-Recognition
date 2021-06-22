from tensorflow.keras.models import load_model
import tensorflow as tf
import tensorflow.keras.backend as K

from master.classification_utils import *

# Load Classification Model

# base_model = MobileNetV2(weights='imagenet', include_top=False)
# model_name = 'MobileNetV2'
# # top layers
# n_classes = 101
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(512, activation='relu')(x)
# x = Dropout(0.2)(x)
# predictions = Dense(n_classes,
#                     kernel_regularizer=regularizers.l2(0.005),
#                     activation='softmax')(x)
# model = Model(inputs=base_model.input, outputs=predictions)
# # model.summary()


# Class Labels
fileReader = open('../food-101/meta/labels.txt', 'r')
food_list = [line.rstrip() for line in fileReader.readlines()]
fileReader.close()

K.clear_session()

model_best = load_model("weights-improvement-41-0.82.hdf5", compile=False)

img_paths = ['sushii.jpg', 'tomato.jpeg', 'egg.jpeg', 'cucumber.jpeg']
images = []
with tf.device('/device:GPU:0'):
    for img_path in img_paths:
        images.append(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
    pred_labels = predict_class(model_best, images, food_list, True)
