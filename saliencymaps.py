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
from tensorflow.keras.preprocessing import image
from tensorflow.keras import applications
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras import backend as K
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

from matplotlib import pyplot as plt

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import matplotlib.image as mpimg

# util function to convert a tensor into a valid image
def tensor2image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.2
    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)
    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def show_img(path, label):
    path = 'D:\\College\\Semester 9\\GP\\Codes\\food-101\\images\\' + path
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

    n_classes = 101

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
    train_data_dir = 'food-101/train/'
    validation_data_dir = 'food-101/test/'
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

    model.load_weights("weights-improvement-41-0.82.hdf5")
    print("Model weights loaded.")
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'top_k_categorical_accuracy'])

    # Predicted values
    str_labels = []
    fileReader = open('food-101/meta/labels.txt', 'r')
    for line in fileReader.readlines():
        str_labels.append(line.rstrip())
    fileReader.close()

    y_pred = model.predict_generator(validation_generator, nb_validation_samples / batch_size)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = validation_generator.classes

    # get the symbolic outputs of each "key" layer
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    for layer in model.layers:
        print(layer.name)


    fnames = validation_generator.filenames
    errors = np.where(y_pred != y_true)[0]
    print("Miss-classified total:", len(errors))

    while True:
        random_index = random.randint(0, len(errors) - 1)
        i = errors[random_index]
        print("Path: ", fnames[i])
        img = image.load_img('D:\\College\\Semester 9\\GP\\Codes\\food-101\\images\\' +fnames[i], target_size=(299, 299))
        x = image.img_to_array(img)
        print('image.img_to_array: ', x.shape, np.max(x), np.min(x))
        x = np.expand_dims(x, axis=0)
        print('expand_dims: ', x.shape, np.max(x), np.min(x))
        x = preprocess_input(x)
        print('preprocess_input: ', x.shape, np.max(x), np.min(x))

        preds = model.predict(x)
        # Get results into a list of tuples (class, description, probability)
        # print('Predicted:', decode_predictions(preds, top=3)[0])

        model_out = K.mean(layer_dict['Conv_1_bn'].output)
        # compute the gradient of the input picture wrt this loss
        grads = K.gradients(model_out, model.input)[0]

        # Normalize the gradient
        grads /= K.std(grads) + 1e-8

        # function: returns the loss and grads given the input picture
        model_predictor = K.function([model.input], [model_out, grads])
        model_outputs, grads_values = model_predictor([x])

        # get the grads that have the same shape  as the input image
        abs_grads_values = np.abs(grads_values)
        sm = tensor2image(abs_grads_values[0])
        print(sm.shape)

        # let's see the grads as an image
        gs = sm[:, :, 0] + sm[:, :, 1] + sm[:, :, 2]
        gs[gs < 150] = 0
        plt.figure(figsize=(6, 6))
        plt.imshow(gs, cmap='Blues_r')
        plt.colorbar()

        x1 = image.img_to_array(img).astype('uint8')
        concan = np.concatenate((x1, sm), axis=0)
        plt.figure(figsize=(12, 12))
        plt.imshow(concan)

        label = "Predicted Label: " + str_labels[y_pred[i]] + ". True Label: " + str_labels[y_true[i]]
        show_img(fnames[i], label)


K.clear_session()

# Run using GPU
GPU = False

if GPU:
    with tf.device('/device:GPU:0'):
        main()
else:
    main()

'''
    
'''
