import os
import time
import torch
import einops
import random
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
from tensorboardX import SummaryWriter
from datetime import datetime
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from data import *
from utils import warp, update_train_log, write_train_log, update_eval_log, write_eval_log

def main(args):
  bsz = 32
  lr = 1e-4
  verbose = args.verbose
  use_stn = not args.no_stn

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  dt_string = datetime.now().strftime("%m_%d_%H_%M")
  writer = SummaryWriter(logdir='../logs/{}-{}'.format(dt_string, verbose))

  # DATASET
  trn_dataset = ClockSyn(augment=True, use_homography=True, use_artefacts=True)
  timelapse_dataset = ClockTimelapse('../data/labels/{}.txt'.format(args.verbose), augment=True)
  coco_dataset = ClockEval('coco')
  openimg_dataset = ClockEval('openimages')
  movie_dataset = ClockEval('clockmovies')
  trn_loader = DataLoader(trn_dataset, batch_size=bsz, shuffle=True)
  timelapse_loader = DataLoader(timelapse_dataset, batch_size=bsz, shuffle=True, drop_last=True)
  coco_loader = DataLoader(coco_dataset, batch_size=1, shuffle=False)
  openimg_loader = DataLoader(openimg_dataset, batch_size=1, shuffle=False)
  movie_loader = DataLoader(movie_dataset, batch_size=1, shuffle=False)

  # MODEL
  model_stn = models.resnet50(pretrained=True)
  model_stn.fc = nn.Linear(2048, 8)
  model = models.resnet50(pretrained=True)
  model.fc = nn.Linear(2048, 720)
  resume_path = '../models/{}.pth'.format(args.verbose)
  stn_resume_path = '../models/{}_st.pth'.format(args.verbose)
  model.load_state_dict(torch.load(resume_path))
  model_stn.load_state_dict(torch.load(stn_resume_path))
  model_stn.to(device)
  model.to(device)

  #OPTIM
  optimizer = optim.Adam(list(model.parameters()) + list(model_stn.parameters()), lr=lr)
  cross_entropy = torch.nn.CrossEntropyLoss()

  for ep in range(40):
    print('Epoch {}'.format(ep))
    train_log = {'loss_cls': [], 'loss_reg': [], 'hour_acc': [], 'minute_acc': []}

    for i, trn_sample in enumerate(zip(trn_loader, timelapse_loader)):
      model.train()
      model_stn.train()
      optimizer.zero_grad()

      img, hour, minute, Minv = trn_sample[0]
      img2, hour2, minute2 = trn_sample[1]

      img = torch.cat([img, img2], 0)
      hour = torch.cat([hour, hour2], 0)
      minute = torch.cat([minute, minute2], 0)

      img = img.float().to(device)
      Minv = Minv.to(device)
      hour = hour.type(torch.long).to(device)
      minute = minute.type(torch.long).to(device)

      # PREDICT
      if use_stn:
        pred_st = model_stn(img)
        pred_st = torch.cat([pred_st,torch.ones(bsz*2,1).to(device)], 1)
        Minv_pred = torch.reshape(pred_st, (-1, 3, 3))
        img_ = warp(img, Minv_pred)
        if random.random() < 0.5:
          pred = model(img_)
        else:
          pred = model(img)
        loss_reg = torch.mean(torch.abs((Minv.reshape(bsz,9) - pred_st[:bsz])))
        loss_cls = cross_entropy(pred, hour * 60 + minute)
      else:
        pred = model(img)
        loss_cls = cross_entropy(pred, hour * 60 + minute)
        loss_reg = 0.

      # LOSS
      loss = 100 * loss_reg + loss_cls
      loss.backward()
      optimizer.step()

      # METRIC
      max_pred = torch.argsort(pred, dim=1, descending=True)  
      max_pred = max_pred[:,0]
      max_h = max_pred // 60
      max_m = max_pred % 60
      hour_acc = float(torch.sum(max_h == hour)) / (2*bsz)
      minute_acc = float(torch.sum(torch.abs(max_m - minute) <= 1)) / (2*bsz)

      update_train_log(train_log, loss_cls, loss_reg, hour_acc, minute_acc)
      if i == 0:
        writer.add_images('train', img, ep)
        if use_stn: writer.add_images('train_warped', img_, ep)
    write_train_log(writer, train_log, use_stn, ep)

    names = ['COCO', 'OpenImages', 'ClockMovies']
    for i, vloader in enumerate([coco_loader, openimg_loader, movie_loader]):
      eval_log = {'top_1': [],'top_2': [],'top_3': [],'top_1_hr': [], 'top_1_min': [], 'iou50': []}
      imgs = []
      imgs_warped = []
      for idx, val_sample in enumerate(vloader):
        with torch.no_grad():
          model.eval()
          model_stn.eval()
          
          img, hour, minute, iou50 = val_sample
          img = img.float().to(device)
          hr = hour.type(torch.long).to(device)
          mn = minute.type(torch.long).to(device)

          #MODEL
          if use_stn:
            pred_st = model_stn(img)
            pred_st = torch.cat([pred_st,torch.ones(1,1).to(device)], 1)
            Minv_pred = torch.reshape(pred_st, (-1, 3, 3))
            img_ = warp(img, Minv_pred)
            pred = model(img_)
          else:
            pred = model(img)  

          #top 3 predictions
          max_pred = torch.argsort(pred, dim=1, descending=True)
          max_pred = max_pred[0,:3]
          max_h = max_pred[0] // 60
          max_m = max_pred[0] % 60

          minute_err = torch.sum(torch.abs(max_m - mn))
          both_err = torch.abs(max_pred - (hr * 60 + mn))
          top_1 = float(both_err[0] <= 1) + float(both_err[0] == 719)
          top_2 = float(both_err[1] <= 1) + float(both_err[1] == 719)
          top_3 = float(both_err[2] <= 1) + float(both_err[2] == 719)
          top_1_hr = float(torch.sum(max_h == hr))
          top_1_min = float(minute_err <= 1) + float(minute_err == 59)

          update_eval_log(eval_log, top_1, top_2, top_3, top_1_hr, top_1_min, int(iou50))

          if idx < 64:
            imgs.append(img[0])
            if use_stn: imgs_warped.append(img_[0])        
      writer.add_images(names[i], torch.stack(imgs,0), ep)
      if use_stn: writer.add_images(names[i]+'_warped', torch.stack(imgs_warped,0), ep)
      write_eval_log(writer, eval_log, i, ep)

    torch.save(model.state_dict(), '../models/{}+.pth'.format(verbose))
    if use_stn: torch.save(model_stn.state_dict(), '../models/{}+_st.pth'.format(verbose))

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--no_stn', action='store_true')
    parser.add_argument('--verbose', type=str, default='base')
    
    args = parser.parse_args()
    main(args)
