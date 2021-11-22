import os
import time
import cv2
import csv
import torch
import einops
import ast
from datetime import datetime
import random
from random import shuffle
import pandas as pd
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader
from natsort import natsorted
import matplotlib.pyplot as plt
from scipy import ndimage
from SynClock import gen_clock
from tensorboardX import SummaryWriter
from kornia.geometry.transform import warp_perspective
from kornia.geometry.homography import find_homography_dlt
from kornia.augmentation import RandomPerspective

def get_iou(bb_det, gt):
  iou = []
  for bb_gt in gt:
    bb1  = {'x1':bb_det[0], 'x2':bb_det[0]+bb_det[2], 'y1':bb_det[1], 'y2':bb_det[1]+bb_det[3]}
    bb2  = {'x1':bb_gt[0], 'x2':bb_gt[0]+bb_gt[2], 'y1':bb_gt[1], 'y2':bb_gt[1]+bb_gt[3]}
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
      iou.append(0.)
      break

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou.append(intersection_area / float(bb1_area + bb2_area - intersection_area))
  return max(iou)

def crop_with_context(img, bbox, margin=0.2):
  x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]
  I = img
  H, W, _ = np.shape(I)
  h = y2-y1
  w = x2-x1
  X1 = int(max(0, x1-margin*w))
  X2 = int(min(W, x2+margin*h))
  Y1 = int(max(0, y1-margin*w))
  Y2 = int(min(H, y2+margin*h))
  hh = Y2-Y1
  ww = X2-X1
  XX1 = max(0, (X1+X2)//2 - max(hh, ww)//2)
  XX2 = min(W, (X1+X2)//2 + max(hh, ww)//2)
  YY1 = max(0, (Y1+Y2)//2 - max(hh, ww)//2)
  YY2 = min(H, (Y1+Y2)//2 + max(hh, ww)//2)
  I = I[YY1:YY2, XX1:XX2, :]
  return I

class ClockSyn(torch.utils.data.Dataset):
  def __init__(self, augment=True, use_homography=True, use_artefacts=True, size=80000):
    self.size = size
    self.use_homography = use_homography
    self.use_artefacts = use_artefacts
    self.augment = augment

  def __len__(self):
    return self.size

  def __getitem__(self, i):
    img, hour, minute, Minv = gen_clock(use_homography=self.use_homography, use_artefacts=self.use_artefacts)
    if self.augment:
      img = cv2.resize(img, (256, 256))

      r = img[:,:,0] * np.random.uniform(0.9, 1.1)
      g = img[:,:,1] * np.random.uniform(0.9, 1.1)
      b = img[:,:,2] * np.random.uniform(0.9, 1.1)
      
      rgb = [r,g,b]

      img = np.stack(rgb, 2)
      if random.random() < 0.5:
        k = np.random.randint(1,10)
        img = cv2.blur(img, (k,k))
      if random.random() < 0.5:
        H, W, _ = np.shape(img)
        img = img + 10 * np.random.uniform(-1.0, 1.0, (H, W, 3))
      if random.random() < 0.5:
        sz = np.random.randint(64, 224)
        img = cv2.resize(img, (sz, sz))
    img = np.clip(img, 0, 255)
    img = cv2.resize(img, (224, 224))/255.
    img = einops.rearrange(img, 'h w c -> c h w')
    if hour == 12: hour = 0
    if minute == 60: minute = 0
    return img, hour, float(minute), Minv

class ClockEval(torch.utils.data.Dataset):
  def __init__(self, dataset):
    assert dataset in ['coco', 'openimages', 'clockmovies']
    base_dir = '/path/to/dataset'
    if dataset == 'coco':
      anno_dir = '../data/coco_final.csv'
      data_dir = base_dir + 'COCO_train/trainval_images'
    if dataset == 'openimages':
      anno_dir = '../data/openimg_final.csv'
      data_dir = base_dir + 'OpenImages_train/data'
    if dataset == 'clockmovies':
      anno_dir =  '../data/movies_final.csv'
      data_dir = base_dir + 'ClockMovies/images'

    annos = pd.read_csv(anno_dir).to_dict()
    self.bbox = annos['bbox_det']
    self.dataset = dataset
    if dataset != 'clockmovies':
      self.bbox_gt = annos['bbox_gt']
    self.data_dir = data_dir
    self.file_name = annos['file_name']
    self.hour = annos['hour']
    self.minute = annos['minute']
    self.names = [x for x in natsorted(os.listdir(data_dir)) if ('.jpg' in x) or ('.png' in x)]
    if dataset == 'clockmovies':
      self.names.remove('Arizona Dream 2.19 p.m.png') #detector fails, count as failure

  def __len__(self):
    return len(self.names)

  def __getitem__(self, i):
    idx = list(self.file_name.keys())[list(self.file_name.values()).index(self.names[i])]
    img = cv2.imread(os.path.join(self.data_dir, self.file_name[idx]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bbox = ast.literal_eval(self.bbox[idx])
    img = crop_with_context(img, bbox)
    img = cv2.resize(img, (224, 224))
    img = einops.rearrange(img, 'h w c -> c h w') / 255.
    hour = self.hour[idx]
    if hour == 12: hour = 0
    minute = self.minute[idx]
    if self.dataset != 'clockmovies':
      iou = get_iou(bbox, ast.literal_eval(self.bbox_gt[idx]))
    else:
      iou = 0.
    iou50 = iou >= 0.5

    return img, hour, minute, iou50

class ClockTimelapse(torch.utils.data.Dataset):
  def __init__(self, anno_dir, augment=True):
    self.data_dir = '/path/to/cropped_data_cbnet'
    self.anno_dir = anno_dir
    with open(anno_dir, "r") as f:
      anno = f.readlines()
    self.names = []
    self.times = []
    for l in anno:
      name, time = l.split(',', 1)
      self.names.append(name)
      self.times.append(time)
    self.augment = augment
    print(str(len(self.names)) + ' valid videos')
  def __len__(self):
    return len(self.names) * 10

  def __getitem__(self, i):
    i = i % len(self.names)
    #each data is a video, not an image
    imgs = natsorted([x for x in os.listdir(os.path.join(self.data_dir, self.names[i])) if '.png' in x])
    
    num_imgs = len(imgs)
    idx = np.random.randint(0, num_imgs)
    img = cv2.imread(os.path.join(self.data_dir, self.names[i],imgs[idx]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if self.augment:
      img = cv2.resize(img, (256, 256))
      w = np.random.randint(224,256)
      h = np.random.randint(224,256)
      x = np.random.randint(0, 256-w)
      y = np.random.randint(0, 256-h)

      img = img[y:y+h, x:x+w, :]

      r = img[:,:,0] * np.random.uniform(0.9, 1.1)
      g = img[:,:,1] * np.random.uniform(0.9, 1.1)
      b = img[:,:,2] * np.random.uniform(0.9, 1.1)
      
      rgb = [r,g,b]
      random.shuffle(rgb)

      img = np.stack(rgb, 2)

      if random.random() < 0.5:
        H, W, _ = np.shape(img)
        img = img + 10 * np.random.uniform(-1.0, 1.0, (H, W, 3))

    img = np.clip(img, 0, 255)
    img = cv2.resize(img, (224, 224))/255.
    img = einops.rearrange(img, 'h w c -> c h w')
    img = torch.Tensor(img)
    aug = RandomPerspective(0.5, p=0.5)
    img = aug(img)[0]

    time = ast.literal_eval(self.times[i])[idx]
    hour = time // 60
    minute = time % 60
    if hour == 12: hour = 0
    if minute == 60: minute = 0
    return img, hour, minute

