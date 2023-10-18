#copyright Alex Spirin @ 2022
import math 
import numpy as np
import cv2 
from PIL import Image

def blended_roll(img_copy, shift, axis):
  if int(shift) == shift:
    return np.roll(img_copy, int(shift), axis=axis)

  max = math.ceil(shift)
  min = math.floor(shift)
  if min != 0 :
    img_min = np.roll(img_copy, min, axis=axis)
  else:
    img_min = img_copy
  img_max = np.roll(img_copy, max, axis=axis)
  blend = max-shift
  img_blend = img_min*blend + img_max*(1-blend)
  return img_blend

def move_cluster(img,i,res2, center, mode='blended_roll'):
  img_copy = img.copy()
  motion = center[i]
  mask = np.where(res2==motion, 1, 0)[...,0][...,None]
  y, x = motion
  if mode=='blended_roll':
    img_copy = blended_roll(img_copy, x, 0)
    img_copy = blended_roll(img_copy, y, 1)
  if mode=='int_roll':
    img_copy = np.roll(img_copy, int(x), axis=0)
    img_copy = np.roll(img_copy, int(y), axis=1)
  return img_copy, mask

def get_k(flow, K):
  Z = flow.reshape((-1,2))
  # convert to np.float32
  Z = np.float32(Z)
  # define criteria, number of clusters(K) and apply kmeans()
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
  ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
  # Now convert back into uint8, and make original image
  res = center[label.flatten()]
  res2 = res.reshape((flow.shape))
  return res2, center

def k_means_warp(flo, img, num_k):
  # flo = np.load(flo)
  img = np.array((img).convert('RGB'))
  num_k = 8

  # print(img.shape)
  res2, center = get_k(flo, num_k)
  center = sorted(list(center), key=lambda x: abs(x).mean())

  img = cv2.resize(img, (res2.shape[:-1][::-1]))
  img_out = np.ones_like(img)*255.

  for i in range(num_k):
    img_rolled, mask_i = move_cluster(img,i,res2,center)
    img_out = img_out*(1-mask_i) + img_rolled*(mask_i)

  # cv2_imshow(img_out)
  return Image.fromarray(img_out.astype('uint8'))
