"""### Import all necessary libraries"""
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.applications import InceptionV3, VGG16, ResNet50, MobileNetV2, NASNetMobile
from tensorflow.keras.applications import NASNetLarge, InceptionResNetV2, DenseNet121
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow import keras
import numpy as np
import random

from matplotlib import pyplot as plt

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import matplotlib.image as mpimg
import os

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
dataset_path = "D:/College/Semester 9/GP/Codes/Datasets/Custom Dataset"
def show_img(path, label):
    path = 'D:\\College\\Semester 9\\GP\\Codes\\Datasets\\Custom Dataset\\images\\' + path
    img = mpimg.imread(path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(label, fontdict={'family': 'serif',
                               'color': 'darkred',
                               'weight': 'normal',
                               'size': 10,
                               })
    plt.show()


def main():
    base_model = MobileNetV2(weights='imagenet', include_top=False)
    model_name = 'MobileNetV2'
    # epoch_num = 70

    """### Add new top layers to the selected model"""

    n_classes = 54

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)

    predictions = Dense(n_classes,
                        kernel_regularizer=regularizers.l2(0.005),
                        activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.summary()

    print("({}) model loaded.".format(model_name))

    img_width, img_height = 299, 299
    train_data_dir = dataset_path + '/train/'
    validation_data_dir = dataset_path + '/test/'
    batch_size = 32  # 64

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    nb_train_samples = train_generator.n
    nb_validation_samples = validation_generator.n
    n_classes = train_generator.num_classes

    """### Compile the model
    #### Compile the model with SGD optimazer, and use top 1 and top 5 accuracy metrics. Initialize two callbacks, one for checkpoints and one for the training logs
    """

    model.load_weights("D:/College/Semester 9/GP/Codes/master/classification "
                       "weights/54_weights/weights_2/weights-improvement-13-0.85.hdf5")
    print("Model weights loaded.")
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'top_k_categorical_accuracy'])

    # Predicted values
    str_labels = []
    fileReader = open('D:/College/Semester 9/GP/Codes/Datasets/Custom Dataset/meta/labels.txt', 'r')
    for line in fileReader.readlines():
        str_labels.append(line.rstrip())
    fileReader.close()

    y_pred = model.predict_generator(validation_generator, nb_validation_samples / batch_size)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = validation_generator.classes

    fnames = validation_generator.filenames
    errors = np.where(y_pred != y_true)[0]
    print("Miss-classified total:", len(errors))
    # is the predicted values
    # for i in errors:
    while True:
        random_index = random.randint(0, len(errors) - 1)
        i = errors[random_index]
        print("Path: ", fnames[i])
        label = "Predicted Label: " + str_labels[y_pred[i]] + ". True Label: " + str_labels[y_true[i]]
        show_img(fnames[i], label)


K.clear_session()

# Run using GPU
GPU = True

if GPU:
    with tf.device('/device:GPU:0'):
        main()
else:
    main()

'''
    
'''
