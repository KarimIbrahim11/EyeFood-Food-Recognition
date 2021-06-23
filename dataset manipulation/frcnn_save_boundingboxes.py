import glob

import matplotlib
from matplotlib import patches
from tensorflow.keras.models import load_model
import tensorflow as tf
import tensorflow.keras.backend as K
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from master.detection_utils import *
from master.classification_utils import *

## Detection MODEL
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2  # 100
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
path = 'D:/College/Semester 9/GP/Codes/master/detection weights/fasterrcnn_uec12.pth'
if torch.cuda.is_available():
    model.load_state_dict(torch.load(path))
else:
    model.load_state_dict(torch.load(path, map_location=device))

model.eval()  # put the model in evaluation mode
##

## LOAD IMAGE
images_count=0
for path in glob.glob("D:/College/Semester 9/GP/Codes/Datasets/Custom Dataset/Foul/*.jpg"):
    # path = 'D:/College/Semester 9/GP/Codes/Datasets/Custom Dataset/Foul/*.jpg'
    img = cv2.imread(path)
    ##

    ## Pre-Process image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = torchvision.transforms.ToPILImage()(img)
    transform = torchvision.transforms.Compose([transforms.Resize((224, 224)),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.5, ], [0.5, ])])
    image = transform(image)
    ##

    ## Predict Detection Images
    with torch.no_grad():
        prediction = model([image.to(device)])[0]

    # print('MODEL OUTPUT\n')

    nms_prediction = apply_nms(prediction, iou_thresh=0.08)
    images, plates_positions = get_cropped_images(img, nms_prediction)
    print(len(images))
    ##

    ## Show Cropped Images
    # plot_img_bbox(img, nms_prediction)
    f = plt.figure()
    c = 0

    # Class Labels
    fileReader = open('D:/College/Semester 9/GP/Codes/Datasets/food-101/meta/labels.txt', 'r')
    food_list = [line.rstrip() for line in fileReader.readlines()]
    fileReader.close()

    K.clear_session()

    model_best = load_model(
        "D:/College/Semester 9/GP/Codes/master/classification weights/weights-improvement-41-0.82.hdf5", compile=False)

    # for i in images:
    #     f.add_subplot(1, len(images), c + 1)
    #     c += 1
    #     plt.imshow(i)
    # plt.show()

    # with tf.device('/device:CPU:0'):
    #     predicted_labels = predict_class(model_best, images, food_list, False)

    for l in range(len(images)):
        print("Ana hena")
        cv2.imwrite('D:/College/Semester 9/GP/Codes/Datasets/Custom Dataset/Foul/boundingboxes/FulMedames_' + str(images_count) +'_'+str(l)+ '.jpg',
                    cv2.cvtColor(images[l], cv2.COLOR_BGR2RGB))
    images_count=images_count+1
    # print(plates_positions, predicted_labels)
    print(plates_positions)
    ##
