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
import os
from tensorflow.keras.regularizers import l2
from tensorflow import keras
import numpy as np

from matplotlib import pyplot as plt

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
dataset_path = "D:/College/Semester 9/GP/Codes/Datasets/Custom Dataset"


def display_confusion_matrix(cmat, score, precision, recall, str_labels):
    plt.figure(figsize=(15, 15))
    ax = plt.gca()
    ax.matshow(cmat, cmap='Reds')
    ax.set_xticks(range(len(str_labels)))
    ax.set_xticklabels(str_labels, fontdict={'fontsize': 7})
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
    ax.set_yticks(range(len(str_labels)))
    ax.set_yticklabels(str_labels, fontdict={'fontsize': 7})
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    titlestring = ""
    if score is not None:
        titlestring += 'f1 = {:.3f} '.format(score)
    if precision is not None:
        titlestring += '\nprecision = {:.3f} '.format(precision)
    if recall is not None:
        titlestring += '\nrecall = {:.3f} '.format(recall)
    if len(titlestring) > 0:
        ax.text(101, 1, titlestring,
                fontdict={'fontsize': 18, 'horizontalalignment': 'right', 'verticalalignment': 'top',
                          'color': '#804040'})
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

    # Importing the dependancies
    from sklearn import metrics
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import numpy as np

    # Predicted values
    # !head food-101/meta/train.txt
    str_labels = []
    fileReader = open('D:/College/Semester 9/GP/Codes/Datasets/Custom Dataset/meta/labels.txt', 'r')
    for line in fileReader.readlines():
        str_labels.append(line.rstrip())
    fileReader.close()

    y_pred = model.predict_generator(validation_generator, nb_validation_samples / batch_size)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = validation_generator.classes

    score = f1_score(y_true, y_pred, labels=range(len(str_labels)), average='macro')
    precision = precision_score(y_true, y_pred, labels=range(len(str_labels)), average='macro')
    recall = recall_score(y_true, y_pred, labels=range(len(str_labels)), average='macro')

    cm = metrics.confusion_matrix(y_true, y_pred, labels=range(len(str_labels)), normalize='true')
    print(cm)
    # Write confusion matrix in a file
    mat = np.matrix(cm)
    with open('confusion_matrix_54.txt', 'wb') as f:
        for line in mat:
            np.savetxt(f, line, fmt='%.2f')
    # np.savetxt('confusion_matrix.txt', cm, fmt='%.2f')
    # print(confusion_matrix)
    print("Confusion Matrix Shape:", cm.shape)
    display_confusion_matrix(cm, score, precision, recall, str_labels)

    print("Classification Report using another method: ")
    # Printing the precision and recall, among other metrics
    print(metrics.classification_report(y_true, y_pred, labels=range(len(str_labels))))


K.clear_session()

with tf.device('/device:GPU:0'):
    main()

'''
    fnames = validation_generator.filenames  ## fnames is all the filenames/samples used in testing
    errors = np.where(y_pred != y_true)[0]  # # misclassifications done on the test data where y_pred
    # is the predicted values
    for i in errors:
        print(fnames[i])
'''
