from os import listdir
from os import path
from os import mkdir
from os.path import isfile, join

dataset_path = "D:/College/Semester 9/GP/Codes/Datasets/Custom Dataset"

# Create meta folder
metapath = dataset_path + "/meta"
if not path.isdir(metapath):
    mkdir(metapath)

# Create classes.txt
f = open(metapath + '/classes.txt', 'w')
for directory in listdir(dataset_path + "/images"):
    f.write(str(directory) + "\n")
f.close()

# Create labels.txt
f = open(metapath + '/labels.txt', 'w')
for directory in listdir(dataset_path + "/images"):
    f.write(str(directory.capitalize().replace("_", " ")) + "\n")
f.close()

# Create train.txt
train = open(metapath + '/train.txt', 'w')
test = open(metapath + '/test.txt', 'w')

for directory in listdir(dataset_path + "/images"):
    count = 1
    for imagename in listdir(dataset_path + "/images/" + directory):
        if count <= 750:
            train.write(str(directory) + "/" + str(imagename.replace(".jpg", "")) + "\n")
        else:
            test.write(str(directory) + "/" + str(imagename.replace(".jpg", "")) + "\n")
        count = count + 1
train.close()
test.close()
# print(f.read())
# onlydirs = [d for d in listdir(dirpath)]
# mypath = "D:/College/Semester 9/GP/Codes/Datasets/food-101/images/apple_pie"
#
# onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
# print(onlydirs)
