import torch
import torch.nn.functional as F
from kornia.geometry.transform import warp_perspective
import numpy as np

def warp(img, Minv_pred, sz=224):
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	s,t = sz/2., 1.
	Minv_pred = torch.Tensor([[s,0,t*s],[0,s,t*s],[0,0,1]]).to(device) @ Minv_pred @ torch.Tensor([[1/s,0,-t],[0,1/s,-t],[0,0,1]]).to(device)
	img_ = warp_perspective(img, Minv_pred, (sz, sz))
	return img_


def update_train_log(train_log, loss_cls, loss_reg, hour_acc, minute_acc):
	train_log['loss_cls'].append(loss_cls)
	train_log['loss_reg'].append(loss_reg)
	train_log['hour_acc'].append(hour_acc)
	train_log['minute_acc'].append(minute_acc)
	return

def write_train_log(writer, train_log, use_stn, ep):
	loss_main = train_log['loss_cls']
	loss_st = train_log['loss_reg']
	h_acc = train_log['hour_acc']
	m_acc = train_log['minute_acc']
	if use_stn: writer.add_scalar('train/loss_st', sum(loss_st)/len(loss_st), ep)
	writer.add_scalar('train/loss_main', sum(loss_main)/len(loss_main), ep)
	writer.add_scalar('train/h_acc', sum(h_acc)/len(h_acc), ep)
	writer.add_scalar('train/m_acc', sum(m_acc)/len(m_acc), ep)
	return


def update_eval_log(eval_log, top_1, top_2, top_3, top_1_hr, top_1_min, iou50):
	eval_log['top_1'].append(top_1)
	eval_log['top_2'].append(top_2)
	eval_log['top_3'].append(top_3)
	eval_log['top_1_hr'].append(top_1_hr)
	eval_log['top_1_min'].append(top_1_min)
	eval_log['iou50'].append(iou50)
	return

def write_eval_log(writer, eval_log, i, ep):
	names = ['COCO', 'OpenImages', 'ClockMovies']
	counts = [1911, 1317, 1244]
	b_acc = eval_log['top_1']
	b2_acc = eval_log['top_2']
	b3_acc = eval_log['top_3']
	h_acc = eval_log['top_1_hr']
	m_acc = eval_log['top_1_min']
	iou50s = eval_log['iou50']

	b2_agg = np.clip(np.array(b_acc) + np.array(b2_acc), 0, 1)
	b3_agg = np.clip(np.array(b_acc) + np.array(b2_acc) + np.array(b3_acc), 0, 1)

	writer.add_scalar('{}/top1_both_acc'.format(names[i]), sum(b_acc)/counts[i], ep)
	writer.add_scalar('{}/top1_h_acc'.format(names[i]), sum(h_acc)/counts[i], ep)
	writer.add_scalar('{}/top1_m_acc_pm1'.format(names[i]), sum(m_acc)/counts[i], ep)
	writer.add_scalar('{}/top2_both_acc'.format(names[i]), sum(b2_agg)/counts[i], ep)
	writer.add_scalar('{}/top3_both_acc'.format(names[i]), sum(b3_agg)/counts[i], ep)
	
	if i != 2: #not available for openimages
		writer.add_scalar('{}/AP50-top1_both_acc'.format(names[i]), np.sum(np.array(b_acc)*np.array(iou50s))/sum(iou50s), ep)
		writer.add_scalar('{}/AP50-top1_h_acc'.format(names[i]), np.sum(np.array(h_acc)*np.array(iou50s))/sum(iou50s), ep)
		writer.add_scalar('{}/AP50-top1_m_acc_pm1'.format(names[i]), np.sum(np.array(m_acc)*np.array(iou50s))/sum(iou50s), ep)
		writer.add_scalar('{}/AP50-top2_both_acc'.format(names[i]), np.sum(np.array(b2_agg)*np.array(iou50s))/sum(iou50s), ep)
		writer.add_scalar('{}/AP50-top3_both_acc'.format(names[i]), np.sum(np.array(b3_agg)*np.array(iou50s))/sum(iou50s), ep)

def print_eval_log(eval_log, i):
	names = ['COCO', 'OpenImages', 'ClockMovies']
	counts = [1911, 1317, 1244]
	b_acc = eval_log['top_1']
	b2_acc = eval_log['top_2']
	b3_acc = eval_log['top_3']
	h_acc = eval_log['top_1_hr']
	m_acc = eval_log['top_1_min']
	iou50s = eval_log['iou50']

	b2_agg = np.clip(np.array(b_acc) + np.array(b2_acc), 0, 1)
	b3_agg = np.clip(np.array(b_acc) + np.array(b2_acc) + np.array(b3_acc), 0, 1)

	print('{}/top1_both_acc'.format(names[i]), sum(b_acc)/counts[i])
	print('{}/top1_h_acc'.format(names[i]), sum(h_acc)/counts[i])
	print('{}/top1_m_acc_pm1'.format(names[i]), sum(m_acc)/counts[i])
	print('{}/top2_both_acc'.format(names[i]), sum(b2_agg)/counts[i])
	print('{}/top3_both_acc'.format(names[i]), sum(b3_agg)/counts[i])
	
	if i != 2: #not available for openimages
		print('{}/AP50-top1_both_acc'.format(names[i]), np.sum(np.array(b_acc)*np.array(iou50s))/sum(iou50s))
		print('{}/AP50-top1_h_acc'.format(names[i]), np.sum(np.array(h_acc)*np.array(iou50s))/sum(iou50s))
		print('{}/AP50-top1_m_acc_pm1'.format(names[i]), np.sum(np.array(m_acc)*np.array(iou50s))/sum(iou50s))
		print('{}/AP50-top2_both_acc'.format(names[i]), np.sum(np.array(b2_agg)*np.array(iou50s))/sum(iou50s))
		print('{}/AP50-top3_both_acc'.format(names[i]), np.sum(np.array(b3_agg)*np.array(iou50s))/sum(iou50s))
