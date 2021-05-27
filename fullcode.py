# -*- coding: utf-8 -*-
"""Last_Updated_FEB.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15HfW5aE6H1Cf-kPg-xwFatxRbFQS3QgP
"""

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

cd '/content/drive/MyDrive/GP_CNNs_Fine_Tuning_on_FOOD101'

"""### Import all necessary libraries"""

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.applications import InceptionV3,VGG16,ResNet50,MobileNetV2, NASNetMobile
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

# The following line imports the SimpleShallowNet, which is a shallow CNN
# developed for the purposes of the this book chapter
#from ipynb.fs.full.BCh_PureFoodNet import PureFoodNet
K.clear_session()

"""### Choose the model
#### Choose the model that you want to use by setting the value of the "use_the_model" variable from 1 to 8. We should highlight that models from 1 to 7, are popular pretrained networks with ImageNet dataset , which not include the top layers. The 8th model is a simple shallow CNN netword developed for the purposes of this book chapter and it is not pretrained.

"""

use_the_model = 7
model_name = ''

if use_the_model is 1:
    base_model = InceptionV3(weights='imagenet', include_top=False)
    model_name = 'InceptionV3'
    epoch_num = 50
    
elif use_the_model is 2: 
    base_model = VGG16(weights='imagenet', include_top=False)
    model_name = 'VGG16'
    epoch_num = 70
    
elif use_the_model is 3: 
    base_model = ResNet50(weights='imagenet', include_top=False)
    model_name = 'ResNet50'
    epoch_num = 30
    
elif use_the_model is 4: 
    base_model = InceptionResNetV2(weights='imagenet', include_top=False)
    model_name = 'InceptionResNetV2'
    epoch_num = 50
    
elif use_the_model is 5: 
    base_model = NASNetMobile(input_shape=(224,224,3), weights='imagenet', include_top=False)
    model_name = 'NASNetMobile'
    epoch_num = 50
elif use_the_model is 6: 
    base_model = NASNetLarge(input_shape=(331,331,3), weights='imagenet', include_top=False)
    model_name = 'NASNetLarge'
    epoch_num = 50
    
elif use_the_model is 7: 
    base_model = MobileNetV2(weights='imagenet', include_top=False)
    model_name = 'MobileNetV2'
    epoch_num = 70
    epoch_num = 66
    #epoch_num = 10
    
elif use_the_model is 8: 
    base_model = DenseNet121(weights='imagenet', include_top=False)
    model_name = 'DenseNet121'
    epoch_num = 50
    
elif use_the_model is 9: 
    base_model = PureFoodNet.getModel(input_shape=train_generator.image_shape)
    model_name = 'PureFoodNet'
    epoch_num = 300

print("({}) {} model loaded with {} epochs.".format(model_name,use_the_model, epoch_num))

"""### Prepare the training and the validation sets of the food101 dataset
#### Add a small image augmentation to the training set (shear_range, zoom_range, horizontal_flip)
"""

# Helper function to download data and extract
import os
def get_data_extract():
  if "food-101" in os.listdir():
    print("Dataset already downloaded and extracted")
    return True
  elif "food-101.tar.gz" in os.listdir():
    print("Dataset Downloaded but not extracted")
    print("Extracting data..")
    !tar xzvf food-101.tar.gz
    print("Extraction done!")
    return False
  else:
    print("Downloading the data...")
    !wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
    print("Dataset downloaded!")
    print("Extracting data..")
    !tar xzvf food-101.tar.gz
    print("Extraction done!")
    return False

# Download data and extract it to folder
DatasetSplit = get_data_extract()

# import os
# os.listdir('food-101/images')
# print("/////////////////////////////////////")
# os.listdir('food-101/meta/')

!head food-101/meta/train.txt

!head food-101/meta/classes.txt

print(DatasetSplit)

import collections
# Helper method to split dataset into train and test folders
from shutil import copy
def prepare_data(filepath, src,dest):
  classes_images = collections.defaultdict(list)
  with open(filepath, 'r') as txt:
      paths = [read.strip() for read in txt.readlines()]
      for p in paths:
        food = p.split('/')
        classes_images[food[0]].append(food[1] + '.jpg')

  for food in classes_images.keys():
    #print("\nCopying images into ",food)
    if not os.path.exists(os.path.join(dest,food)):
      os.makedirs(os.path.join(dest,food))
    for i in classes_images[food]:
      copy(os.path.join(src,food,i), os.path.join(dest,food,i))
  print("Copying Done!")

# Prepare train dataset by copying images from food-101/images to food-101/train using the file train.txt
if DatasetSplit:
  print("Train set already created")
else:
  print("Creating train data...")
  prepare_data('food-101/meta/train.txt', 'food-101/images', 'food-101/train')
  print("Success")

# Prepare test data by copying images from food-101/images to food-101/test using the file test.txt
if DatasetSplit:
  print("Test set already created")
else:
  print("Creating test data...")
  prepare_data('food-101/meta/test.txt', 'food-101/images', 'food-101/test')
  print("Success..")

# Check how many files are in the train folder
print("Total number of samples in train folder")
!find food-101/train -type d -or -type f -printf '.' | wc -c

# Check how many files are in the test folder
print("Total number of samples in test folder")
!find food-101/test -type d -or -type f -printf '.' | wc -c

img_width, img_height = 299, 299
train_data_dir = 'food-101/train/'
validation_data_dir = 'food-101/test/'
batch_size = 64

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
    shuffle = False)

nb_train_samples = train_generator.n
nb_validation_samples = validation_generator.n
n_classes = train_generator.num_classes

"""### Add new top layers to the selected model"""

#img_width, img_height = 299, 299
#train_data_dir = 'food-101/train/'
#validation_data_dir = 'food-101/test/'
#batch_size = 32
n_classes = 101

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512,activation='relu')(x)
x = Dropout(0.2)(x)

predictions = Dense(n_classes,
                    kernel_regularizer=regularizers.l2(0.005), 
                    activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.summary()

'''
import json
from keras.callbacks import LambdaCallback

json_log = open('loss_log.json', mode='wt', buffering=1)
json_logging_callback = LambdaCallback(
            on_epoch_end=lambda epoch, logs: json_log.write(
                json.dumps({'epoch': epoch, 
                            'loss': logs['loss'],
                            'weights': model.get_weights()}) + '\n'),
            on_train_end=lambda logs: json_log.close()
)
'''

"""### Loading training progress (weights)

"""

model.load_weights("weights-improvement-02-0.29.hdf5")

"""### Compile the model
#### Compile the model with SGD optimazer, and use top 1 and top 5 accuracy metrics. Initialize two callbacks, one for checkpoints and one for the training logs
"""

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), 
              loss='categorical_crossentropy', 
              metrics=['accuracy','top_k_categorical_accuracy'])


'''
checkpointer = ModelCheckpoint(filepath='best_model_food101_'+model_name+'.hdf5',
                               verbose=1,
                               save_best_only=True)
csv_logger = CSVLogger('hist_food101_'+model_name+'.log')
'''

"""### Saving Training Progress"""

filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, save_mode='max')
callbacks_list = [checkpoint]

"""### Training session of the selected model"""

'''
# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "MobileVnet2_food101/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
'''

'''
# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq=5*batch_size)

# Save the weights using the `checkpoint_path` format

model.save_weights(checkpoint_path.format(epoch=0))
'''

hist = model.fit_generator(train_generator,
                           steps_per_epoch = nb_train_samples // batch_size,
                           validation_data = validation_generator,
                           validation_steps = nb_validation_samples // batch_size,
                           epochs = epoch_num,
                           verbose = 1,
                           callbacks = callbacks_list # [cp_callback]
                          )

# Importing the dependancies
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
# Predicted values
#!head food-101/meta/train.txt
str_labels = []
fileReader = open('food-101/meta/labels.txt', 'r')
for line in fileReader.readlines():
    str_labels.append(line.rstrip())
fileReader.close()

y_pred = model.predict_generator(validation_generator, nb_validation_samples / batch_size)
y_pred = np.argmax(y_pred, axis=1)
y_true = validation_generator.classes

from matplotlib import pyplot as plt
def display_confusion_matrix(cmat, score, precision, recall):
    plt.figure(figsize=(15,15))
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
        ax.text(101, 1, titlestring, fontdict={'fontsize': 18, 'horizontalalignment':'right', 'verticalalignment':'top', 'color':'#804040'})
    plt.show()

cm = metrics.confusion_matrix(y_true, y_pred,labels=range(len(str_labels)),normalize='true')
print("Confusion Matrix Shape:",cm.shape)
print(cm)

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

score = f1_score(y_true, y_pred, labels=range(len(str_labels)), average='macro')
precision = precision_score(y_true, y_pred, labels=range(len(str_labels)), average='macro')
recall = recall_score(y_true, y_pred, labels=range(len(str_labels)), average='macro')

display_confusion_matrix(cm, score, precision, recall)

print("Classification Report using another method: ")
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=str_labels)
disp.plot()

# Printing the precision and recall, among other metrics
print(metrics.classification_report(y_true, y_pred, labels=range(len(str_labels))))

"""### Save the last trained model"""

model.save('last_model_food101_'+str(model_name)+'_acc'+str(max(hist.history['acc']))+'.hdf5')

"""0
1
2
3
4
5
6
7
8
9
10
11
12
13
14


"""