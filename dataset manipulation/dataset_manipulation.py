import os
def get_data_extract():
  if "food-101" in os.listdir():
    print("Dataset already downloaded and extracted")
    return True
  elif "food-101.tar.gz" in os.listdir():
    print("Dataset Downloaded but not extracted")
    print("Extracting data..")
    tar xzvf food-101.tar.gz
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

head food-101/meta/train.txt

head food-101/meta/classes.txt

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
