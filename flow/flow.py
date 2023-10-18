# (c) Alex Spirin 2022-2023

from glob import glob 
from PIL import Image
import PIL
from flow.occlusionw import make_cc_map
import numpy as np
import torch
from torchvision.utils import flow_to_image as flow_to_image_torch

class flowDataset():
  def __init__(self, in_path, half=True, normalize=False, warp_interp=None, width_height=None, input_padder=None):
    frames = sorted(glob(in_path+'/*.*'));
    assert len(frames)>2, f'WARNING!\nCannot create flow maps: Found {len(frames)} frames extracted from your video input.\nPlease check your video path.'
    self.frames = frames
    self.normalize = normalize 
    self.warp_inter = warp_interp
    self.input_padder = input_padder
    self.width_height = width_height

  def __len__(self):
    return len(self.frames)-1

  def load_img(self, img, size):
    img = Image.open(img).convert('RGB').resize(size, self.warp_interp)
    return torch.from_numpy(np.array(img)).permute(2,0,1).float()[None,...]

  def __getitem__(self, i):
    frame1, frame2 = self.frames[i], self.frames[i+1]
    frame1 = self.load_img(frame1, self.width_height)
    frame2 = self.load_img(frame2, self.width_height)
    padder = self.input_padder(frame1.shape)
    frame1, frame2 = padder.pad(frame1, frame2)
    batch = torch.cat([frame1, frame2])
    if self.normalize:
      batch = 2 * (batch / 255.0) - 1.0
    return batch
  
def flow_batch(i, batch, pool, ds, flo_fwd_folder, check_consistency, 
               raft_model, use_jit_raft=False, flow_lq=True, flow_save_img_preview=False, 
               save_preview=False, num_flow_updates=12, use_legacy_cc=False, 
               missed_consistency_dilation=3, edge_consistency_width=11):
  with torch.cuda.amp.autocast():
          batch = batch[0]
          frame_1 = batch[0][None,...].cuda()
          frame_2 = batch[1][None,...].cuda()
          frame1 = ds.frames[i]
          frame1 = frame1.replace('\\','/')
          out_flow21_fn = f"{flo_fwd_folder}/{frame1.split('/')[-1]}"
          if flow_lq:   frame_1, frame_2 = frame_1, frame_2
          if use_jit_raft:
            _, flow21 = raft_model(frame_2, frame_1)
          else:
            flow21 = raft_model(frame_2, frame_1, num_flow_updates=num_flow_updates)[-1] #flow_bwd
          mag = (flow21[:,0:1,...]**2 + flow21[:,1:,...]**2).sqrt()
          mag_thresh = 0.5
          #zero out flow values for non-moving frames below threshold to avoid noisy flow/cc maps
          if mag.max()<mag_thresh:
            flow21_clamped = torch.where(mag<mag_thresh, 0, flow21)
          else:
            flow21_clamped = flow21
          flow21 = flow21[0].permute(1, 2, 0).detach().cpu().numpy()
          flow21_clamped = flow21_clamped[0].permute(1, 2, 0).detach().cpu().numpy()

          if flow_save_img_preview or i in range(0,len(ds),max(1, len(ds)//10)):
            pool.apply_async(save_preview, (flow21, out_flow21_fn+'.jpg') )
          pool.apply_async(np.save, (out_flow21_fn, flow21))
          if check_consistency:
            if use_jit_raft:
              _, flow12 = raft_model(frame_1, frame_2)
            else:
              flow12 = raft_model(frame_1, frame_2)[-1] #flow_fwd

            flow12 = flow12[0].permute(1, 2, 0).detach().cpu().numpy()
            if flow_save_img_preview:
              pool.apply_async(save_preview, (flow12, out_flow21_fn+'_12'+'.jpg'))
            if use_legacy_cc:
              pool.apply_async(np.save, (out_flow21_fn+'_12', flow12))
            else:
              joint_mask = make_cc_map(flow12, flow21_clamped, dilation=missed_consistency_dilation,
                                       edge_width=edge_consistency_width)
              joint_mask = PIL.Image.fromarray(joint_mask.astype('uint8'))
              cc_path = f"{flo_fwd_folder}/{frame1.split('/')[-1]}-21_cc.jpg"
              joint_mask.save(cc_path)

def save_preview(flow21, out_flow21_fn, flow_to_image=flow_to_image_torch):
  try:
    Image.fromarray(flow_to_image(flow21)).save(out_flow21_fn, quality=90)
  except:
    print('Error saving flow preview for frame ', out_flow21_fn)