import os
import time
import torch
import einops
import random
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
from natsort import natsorted
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
  images = [x for x in natsorted(os.listdir(args.dir)) if ('.jpg' in x) or ('.png' in x)]

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

  for img_name in images:
    with torch.no_grad():
      model.eval()
      model_stn.eval()

      #MODEL
      img = cv2.imread(os.path.join(args.dir, img_name))
      img = cv2.resize(img, (224, 224))/255.
      img = einops.rearrange(img, 'h w c -> c h w')
      img = torch.Tensor(img)
      img = img.float().to(device)
      img = torch.unsqueeze(img, 0)

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

      print(img_name, max_h.cpu().numpy(), max_m.cpu().numpy())
      
      #img = einops.rearrange(img[0], 'c h w -> h w c').cpu().numpy()[:,:,::-1] * 255 
      #img_ = einops.rearrange(img_[0], 'c h w -> h w c').cpu().numpy()[:,:,::-1] * 255

      #uncomment this to save image
      #os.makedirs('../viz/{}/{}'.format(verbose,names[i]), exist_ok=True)
      #if idx < 100:
      #cv2.imwrite('../viz/{}/{}/{}_{}_{}.png'.format(verbose,names[i],idx, int(max_pred[0]), int(hr*60+mn)), img)
      #cv2.imwrite('../viz/{}/{}/{}_w.png'.format(verbose,names[i],idx), img_)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--verbose', type=str, default='full+++')
    parser.add_argument('--dir', type=str, default='../data/demo')

    args = parser.parse_args()
    main(args)
