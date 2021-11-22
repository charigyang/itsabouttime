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
from utils import warp, update_train_log, write_train_log, update_eval_log, write_eval_log, print_eval_log

def main(args):
  verbose = args.verbose
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  # DATASET
  coco_dataset = ClockEval('coco')
  openimg_dataset = ClockEval('openimages')
  movie_dataset = ClockEval('clockmovies')
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

  names = ['COCO', 'OpenImages', 'ClockMovies']
  for i, vloader in enumerate([coco_loader, openimg_loader, movie_loader]):
    eval_log = {'top_1': [],'top_2': [],'top_3': [],'top_1_hr': [], 'top_1_min': [], 'iou50': []}
    for idx, val_sample in enumerate(vloader):
      with torch.no_grad():
        model.eval()
        model_stn.eval()
        
        img, hour, minute, iou50 = val_sample
        img = img.float().to(device)
        hr = hour.type(torch.long).to(device)
        mn = minute.type(torch.long).to(device)

        #MODEL
        pred_st = model_stn(img)
        pred_st = torch.cat([pred_st,torch.ones(1,1).to(device)], 1)
        Minv_pred = torch.reshape(pred_st, (-1, 3, 3))
        img_ = warp(img, Minv_pred)
        pred = model(img_)


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
        img = einops.rearrange(img[0], 'c h w -> h w c').cpu().numpy()[:,:,::-1] * 255 
        img_ = einops.rearrange(img_[0], 'c h w -> h w c').cpu().numpy()[:,:,::-1] * 255

        #uncomment this to save image
        #os.makedirs('../viz/{}/{}'.format(verbose,names[i]), exist_ok=True)
        #if idx < 100:
        #cv2.imwrite('../viz/{}/{}/{}_{}_{}.png'.format(verbose,names[i],idx, int(max_pred[0]), int(hr*60+mn)), img)
        #cv2.imwrite('../viz/{}/{}/{}_w.png'.format(verbose,names[i],idx), img_)

    print_eval_log(eval_log, i)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--verbose', type=str, default='full+++')

    args = parser.parse_args()
    main(args)
