import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from tqdm import tqdm
import xml.etree.ElementTree as ET
import PIL.Image


def loadPascal():
    images, targets = [], []
    boxes, labels, image_id, iscrowd = [], [], [], []
    directory = '/home/amir/Desktop/archive/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations/'
    for filename in os.listdir(directory):
        if filename.endswith(".xml"):
            try:
                tree = ET.parse(directory + filename)
                root = tree.getroot()
                key = os.path.splitext(filename)[0]

                imgfilename = os.path.join(
                    '/home/amir/Desktop/archive/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages', key + '.jpg')
                img = cv2.imread(imgfilename)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                y, x = img.shape[0], img.shape[1]
                x_scalar = 224 / x
                y_scalar = 224 / y

                info = {}
                for bndbox in root.iter('bndbox'):
                    info[key] = []

                    info[key].append(float(bndbox[0].text))
                    info[key].append(float(bndbox[1].text))
                    info[key].append(float(bndbox[2].text))
                    info[key].append(float(bndbox[3].text))

                    target = {}

                    x1 = int(info[key][0] * x_scalar)
                    y1 = int(info[key][1] * y_scalar)
                    x2 = int(info[key][2] * x_scalar)
                    y2 = int(info[key][3] * y_scalar)
                    poly = [x1, y1, x2, y2]
                    area = (poly[0] - poly[2]) * (poly[1] - poly[3])
                    poly = torch.tensor(poly)
                    poly = torch.unsqueeze(poly, 0)

                    target['boxes'] = poly
                    target['labels'] = torch.tensor([0])  # 0 means non food
                    target['image_id'] = torch.tensor([int(key)])
                    target['area'] = torch.tensor([area])
                    target['iscrowd'] = torch.tensor([0])

                    images.append(img)
                    targets.append(target)

            except FileNotFoundError:
                print('cannot find file ', filename)

    return images, targets


def LoadFoodData(folder_path, n_cls):
  targets = []
  images = []
  boxes, labels, image_id, iscrowd = [], [], [], []
  for i in tqdm(range(1, n_cls+1)):
    info = {}
    path = os.path.join(folder_path, str(i))
    file = open(path + '/bb_info.txt')
    txt = file.read()
    file.close()
    txt = txt.split('\n')
    # Making a dict of text file
    for j in txt[1:]:
      if len(j) > 0:
        temp = j.split(' ')
        info[temp[0]] = [int(x) for x in temp[1:]]
    # For loading images and targets
    for key in info:
      target = {}
      filename = os.path.join(path, key + '.jpg')
      img = cv2.imread(filename)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      y, x = img.shape[0], img.shape[1]

      x_scalar = 224/x
      y_scalar = 224/y
      x1 = int(info[key][0]*x_scalar)
      y1 = int(info[key][1]*y_scalar)
      x2 = int(info[key][2]*x_scalar)
      y2 = int(info[key][3]*y_scalar)
      poly = [x1, y1, x2, y2]
      area = (poly[0]-poly[2]) * (poly[1]-poly[3])
      poly = torch.tensor(poly)
      poly = torch.unsqueeze(poly, 0)

      target['boxes'] = poly
      target['labels'] = torch.tensor([1]) # replce i with 1
      target['image_id'] = torch.tensor([int(key)])
      target['area'] = torch.tensor([area])
      target['iscrowd'] = torch.tensor([0])

      images.append(img)
      targets.append(target)

  return images, targets


class FoodData(Dataset):
  def __init__(self, images, targets, transforms=None):
    self.images = images
    self.targets = targets
    self.transforms = transforms

  def __len__(self):
    return len(self.images)


  def __getitem__(self, idx):
    image = self.images[idx]
    target = self.targets[idx]
    image = torchvision.transforms.ToPILImage()(image)
    if self.transforms:
      image = self.transforms(image)
    return image, target

def collate(batch):
  return tuple(zip(*batch))


im, tar = loadPascal()

images, targets = LoadFoodData('/home/amir/Desktop/UECFOOD100', 100)

images = images + im
targets = targets + tar

train_images, test_images, train_targets, test_targets = train_test_split(images, targets, test_size = 0.2, random_state = 7)

transform = torchvision.transforms.Compose([transforms.Resize((224,224)),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5,], [0.5,])])
traindata = FoodData(train_images, train_targets, transform)
trainloader = DataLoader(traindata, batch_size=8, shuffle=True, collate_fn=collate, num_workers=0)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2 #100
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

device = torch.device('cpu')

model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=3e-4)

import warnings
warnings.filterwarnings('ignore')

epochs = 4
iteration = 100
itr = 0
loss = 0
for e in range(epochs):
    print('Epoch {}'.format(e + 1))
    for img, tar in trainloader:
        img = list(image.to(device) for image in img)
        tar = [{k: v.to(device) for k, v in t.items()} for t in tar]

        optimizer.zero_grad()
        loss_dict = model(img, tar)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        losses.backward()
        optimizer.step()

        loss += loss_value
        itr += 1
        if (itr % iteration == 0):
            print('Iteration:{}\tLoss:{}'.format(itr, (loss / iteration)))
            loss = 0

torch.save(model.state_dict(), 'fasterrcnn_foodtracker_pascal.pth')