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
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2
from tensorflow import keras
import numpy as np

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
dataset_path = "D:/College/Semester 9/GP/Codes/Datasets/Custom Dataset"


def main():
    # The following line imports the SimpleShallowNet, which is a shallow CNN
    # developed for the purposes of the this book chapter
    # from ipynb.fs.full.BCh_PureFoodNet import PureFoodNet
    K.clear_session()

    """### Choose the model
    #### Choose the model that you want to use by setting the value of the "use_the_model" variable from 1 to 8. We should highlight that models from 1 to 7, are popular pretrained networks with ImageNet dataset , which not include the top layers. The 8th model is a simple shallow CNN netword developed for the purposes of this book chapter and it is not pretrained.
    
    """

    use_the_model = 7
    model_name = ''

    if use_the_model == 1:
        base_model = InceptionV3(weights='imagenet', include_top=False)
        model_name = 'InceptionV3'
        epoch_num = 50

    elif use_the_model == 2:
        base_model = VGG16(weights='imagenet', include_top=False)
        model_name = 'VGG16'
        epoch_num = 70

    elif use_the_model == 3:
        base_model = ResNet50(weights='imagenet', include_top=False)
        model_name = 'ResNet50'
        epoch_num = 30

    elif use_the_model == 4:
        base_model = InceptionResNetV2(weights='imagenet', include_top=False)
        model_name = 'InceptionResNetV2'
        epoch_num = 50

    elif use_the_model == 5:
        base_model = NASNetMobile(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
        model_name = 'NASNetMobile'
        epoch_num = 50
    elif use_the_model == 6:
        base_model = NASNetLarge(input_shape=(331, 331, 3), weights='imagenet', include_top=False)
        model_name = 'NASNetLarge'
        epoch_num = 50

    elif use_the_model == 7:
        base_model = MobileNetV2(weights='imagenet', include_top=False)
        model_name = 'MobileNetV2'
        epoch_num = 35
        # epoch_num = 10

    elif use_the_model == 8:
        base_model = DenseNet121(weights='imagenet', include_top=False)
        model_name = 'DenseNet121'
        epoch_num = 50

    # elif use_the_model == 9:
    #     base_model = PureFoodNet.getModel(input_shape=train_generator.image_shape)
    #     model_name = 'PureFoodNet'
    #     epoch_num = 300

    print("({}) {} model loaded with {} epochs.".format(model_name, use_the_model, epoch_num))
    # D:\College\Semester 9\GP\Codes\master\classification weights\54_weights
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

    """### Add new top layers to the selected model"""

    # img_width, img_height = 299, 299
    # train_data_dir = 'food-101/train/'
    # validation_data_dir = 'food-101/test/'
    # batch_size = 32
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

    """### Loading training progress (weights)
        model.load_weights("weights-improvement-02-0.29.hdf5")
    """

    """### Compile the model #### Compile the model with SGD optimazer, and use top 1 and top 5 accuracy metrics. 
    Initialize two callbacks, one for checkpoints and one for the training logs """

    model.load_weights("D:/College/Semester 9/GP/Codes/master/classification "
                       "weights/54_weights/weights-improvement-34-0.84.hdf5")

    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'top_k_categorical_accuracy'])

    """### Saving Training Progress"""

    filepath = "D:/College/Semester 9/GP/Codes/master/classification weights/54_weights/weights_2/weights-improvement-{" \
               "epoch:02d}-{val_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, save_mode='max')
    callbacks_list = [checkpoint]

    """### Training session of the selected model"""

    hist = model.fit_generator(train_generator,
                               steps_per_epoch=nb_train_samples // batch_size,
                               validation_data=validation_generator,
                               validation_steps=nb_validation_samples // batch_size,
                               epochs=epoch_num,
                               verbose=1,
                               callbacks=callbacks_list  # [cp_callback]
                               )
    print(hist.history.keys())
    # summarize history for accuracy
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # # Importing the dependancies
    # from sklearn import metrics
    # from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    # import numpy as np
    #
    # # Predicted values
    # # !head food-101/meta/train.txt
    # str_labels = []
    # fileReader = open(dataset_path + '/meta/labels.txt', 'r')
    # for line in fileReader.readlines():
    #     str_labels.append(line.rstrip())
    # fileReader.close()
    #
    # y_pred = model.predict_generator(validation_generator, nb_validation_samples / batch_size)
    # y_pred = np.argmax(y_pred, axis=1)
    # y_true = validation_generator.classes
    # from matplotlib import pyplot as plt
    #
    # def display_confusion_matrix(cmat, score, precision, recall):
    #     plt.figure(figsize=(15, 15))
    #     ax = plt.gca()
    #     ax.matshow(cmat, cmap='Reds')
    #     ax.set_xticks(range(len(str_labels)))
    #     ax.set_xticklabels(str_labels, fontdict={'fontsize': 7})
    #     plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
    #     ax.set_yticks(range(len(str_labels)))
    #     ax.set_yticklabels(str_labels, fontdict={'fontsize': 7})
    #     plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    #     titlestring = ""
    #     if score is not None:
    #         titlestring += 'f1 = {:.3f} '.format(score)
    #     if precision is not None:
    #         titlestring += '\nprecision = {:.3f} '.format(precision)
    #     if recall is not None:
    #         titlestring += '\nrecall = {:.3f} '.format(recall)
    #     if len(titlestring) > 0:
    #         ax.text(101, 1, titlestring,
    #                 fontdict={'fontsize': 18, 'horizontalalignment': 'right', 'verticalalignment': 'top',
    #                           'color': '#804040'})
    #     plt.show()
    #
    # cm = metrics.confusion_matrix(y_true, y_pred, labels=range(len(str_labels)), normalize='true')
    # print("Confusion Matrix Shape:", cm.shape)
    # print(cm)
    #
    # from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    #
    # score = f1_score(y_true, y_pred, labels=range(len(str_labels)), average='macro')
    # precision = precision_score(y_true, y_pred, labels=range(len(str_labels)), average='macro')
    # recall = recall_score(y_true, y_pred, labels=range(len(str_labels)), average='macro')
    #
    # display_confusion_matrix(cm, score, precision, recall)
    #
    # print("Classification Report using another method: ")
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=str_labels)
    # disp.plot()
    #
    # # Printing the precision and recall, among other metrics
    # print(metrics.classification_report(y_true, y_pred, labels=range(len(str_labels))))

    """### Save the last trained model"""

    model.save('last_model_food101_' + str(model_name) + '_acc' + str(max(hist.history['acc'])) + '.hdf5')


with tf.device('/device:GPU:0'):
    main()

# loss = [4.3454, 2.8749, 2.1014, 1.7668, 1.6163, 1.4814, 1.3848, 1.3053, 1.2497, 1.1967, 1.1421, 1.1033, 1.0563, 1.0336,
#         1.0023, 0.9684, 0.9362, -1, 0.9253, 0.8782, 0.8717, 0.8474, 0.8213, 0.8041]
# accuracy = [0.0808, 0.4353, 0.5834, 0.6578, 0.6874, 0.7212, 0.7486, 0.7601, 0.7749, 0.7876, 0.7979, 0.8060, 0.8146,
#             0.8195, 0.8280, 0.8370, 0.8443, -1, 0.8449, 0.8536, 0.8544, 0.8593,  0.8665, 0.8702]
# top_k_categorical_accuracy = [0.2230, 0.7282, 0.8503, 0.8935, 0.9063, 0.9212, 0.9317, 0.9397, 0.9421, 0.9489, 0.9509,
#                               0.9545, 0.9599, 0.9609, 0.9629, 0.9655, 0.9670, -1, 0.9680, 0.9724, 0.9706, 0.9724,  0.9736, 0.9756 ]
# val_loss = [3.1793, 2.2187, 1.8041, 1.6019, 1.4940, 1.4012, 1.3375, 1.2821, 1.2184, 1.1840, 1.1667, 1.1375, 1.0987,
#             1.0773, 1.0553, 1.0361, 1.0284, -1, 0.9987, 0.9831, 0.9660, 0.9630, 0.9453, 0.9361]
# val_accuracy = [0.3753, 0.5882, 0.6650, 0.7000, 0.7198, 0.7398, 0.7510, 0.7646, 0.7801, 0.7841, 0.7870, 0.7920, 0.8029,
#                 0.8056, 0.8084, 0.8118, 0.8113, -1, 0.8201, 0.8214, 0.8237, 0.8243, 0.8248, 0.8285]
# val_top_k_categorical_accuracy = [0.6800, 0.8486, 0.8906, 0.9102, 0.9213, 0.9319, 0.9382, 0.9406, 0.9451, 0.9499,
#                                   0.9493, 0.9517, 0.9534, 0.9544, 0.9551, 0.9566, 0.9570, -1, 0.9587, 0.9590, 0.9598,
#                                   0.9588, 0.9596, 0.9601]
