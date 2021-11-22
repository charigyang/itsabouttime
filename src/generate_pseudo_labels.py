import os
import time
import torch
import einops
import random
import cv2
import numpy as np
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from utils import warp
from natsort import natsorted
from cyclic_ransac import RANSACRegressor
def main(args):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  # MODEL
  model_stn = models.resnet50(pretrained=True)
  model_stn.fc = nn.Linear(2048, 8)
  model = models.resnet50(pretrained=True)
  model.fc = nn.Linear(2048, 720)
  
  base_dir = '/path/to/cropped_data_cbnet/'
  resume_path = '../models/{}.pth'.format(args.verbose)
  stn_resume_path = '../models/{}_st.pth'.format(args.verbose)
  model.load_state_dict(torch.load(resume_path))
  model_stn.load_state_dict(torch.load(stn_resume_path))
  model.eval()
  model_stn.eval()
  
  model.to(device)
  model_stn.to(device)
  
  #model = nn.DataParallel(model.cuda())
  #model_stn = nn.DataParallel(model_stn.cuda())

  min_vid_length = 30
  min_range = 20
  score_threshold = 0.7
  ransac_threshold = 3

  os.makedirs('../data/labels', exist_ok=True)
  os.makedirs('../plots/pos', exist_ok=True)
  os.makedirs('../plots/neg', exist_ok=True)

  if not args.no_save:
    if os.path.isfile('../data/labels/{}.txt'.format(args.verbose)):
      os.remove('../data/labels/{}.txt'.format(args.verbose))

  vids = natsorted([x for x in os.listdir(base_dir) if '_' in x])
  keep = []
  for vid in vids:
    imgs = natsorted([x for x in os.listdir(base_dir+vid) if '.png' in x])
    ids = [int(x.strip('.png')) for x in imgs]
    if len(imgs) > min_vid_length:
      length = len(imgs)
      frame_gap = len(imgs) // 100 + 1
      imgs = imgs[::frame_gap]
      data = []
      for i in imgs:
        img = cv2.imread(base_dir+vid+'/'+i)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = einops.rearrange(img, 'h w c -> c h w') / 255.
        data.append(img)
      data = np.stack(data, 0)

      data = torch.Tensor(data).float().to(device)

      pred_st = model_stn(data)
      pred_st = torch.cat([pred_st,torch.ones(len(imgs),1).to(device)], 1)
      Minv_pred = torch.reshape(pred_st, (-1, 3, 3))
      data_ = warp(data, Minv_pred, sz=224)

      pred = model(data_)
      max_pred = torch.argsort(pred, dim=1, descending=True)  
      max_pred = max_pred[:,0].detach().cpu().numpy()


      X = np.array([int(x.strip('.png')) for x in imgs]).reshape(-1, 1)
      y = max_pred
      ransac = RANSACRegressor(residual_threshold=ransac_threshold, stop_probability=0.999)
      ransac.fit(X, y)
      inlier_mask = ransac.inlier_mask_
      outlier_mask = np.logical_not(inlier_mask)
      line_X = np.array(ids).reshape(-1, 1)
      line_y_ransac = ransac.predict(line_X)
      line_y_plot = ransac.predict(X)
      score = np.sum(inlier_mask) / len(imgs)

      if (score > score_threshold) and (np.max(line_y_ransac) - np.min(line_y_ransac) > min_range):
        valid = True
      else:
        valid = False

      if args.plot:
        plt.plot()
        plt.scatter(X[inlier_mask], y[inlier_mask], color="yellowgreen", marker=".", label="Inliers")
        plt.scatter(X[outlier_mask], y[outlier_mask], color="gold", marker=".", label="Outliers")
        plt.plot(line_X, line_y_ransac, color="cornflowerblue", linewidth=2,label="RANSAC regressor")
        folder = 'pos' if valid else 'neg'
        #plt.tight_layout(pad=0.15)
        plt.savefig('../plots/{}/{}.png'.format(folder,vid))
        plt.close()
        img_ = cv2.imread('../plots/{}/{}.png'.format(folder,vid))
        H, W, _ = np.shape(img_)
        margin = 20
        if valid:
          cv2.rectangle(img_, (0, 0), (W, H), (0, 255, 0), 20)
        else:
          cv2.rectangle(img_, (0, 0), (W, H), (0, 0, 255), 20)
        x = int(len(imgs) // 7)
        for i in [0, x, 2*x, 3*x, 4*x, 5*x, 6*x, len(imgs)-1]:
          img_i = cv2.imread(base_dir+vid+'/'+imgs[i])
          img_i = cv2.resize(img_i, (W, H))
          if abs(line_y_plot[i] - y[i]) % 720 <= 3:
            cv2.rectangle(img_i, (0, 0), (W, H), (0, 255, 0), 20)
          else:
            cv2.rectangle(img_i, (0, 0), (W, H), (0, 0, 255), 20)
          img_ = np.concatenate([img_, np.ones([H, 40 ,3])*255, img_i], 1)
        cv2.imwrite('../plots/{}/{}.png'.format(folder,vid), img_)

      if valid and not args.no_save:
        with open('../data/labels/{}.txt'.format(args.verbose), 'a') as f:
          f.write(vid + ',' + str(list(np.rint(line_y_ransac).astype(int))))
          f.write('\n')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--verbose', type=str, default='base')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--no_save', action='store_true')
    args = parser.parse_args()
    main(args)
