# imports
import numpy as np
import argparse
import glob
import os
from functools import partial
import vispy
import scipy.misc as misc
from tqdm import tqdm
import yaml
import time
import sys
from Inpainting.mesh import write_ply, read_ply, output_3d_photo
from Inpainting.utils import get_MiDaS_samples, read_MiDaS_depth
import torch
import cv2
from skimage.transform import resize
import imageio
import copy
from Inpainting.networks import Inpaint_Color_Net, Inpaint_Depth_Net, Inpaint_Edge_Net
from Inpainting.MiDaS.run import run_depth
from Inpainting.MiDaS.monodepth_net import MonoDepthNet # model to compute depth
import Inpainting.MiDaS.MiDaS_utils as MiDaS_utils
from Inpainting.bilateral_filtering import sparse_bilateral_filtering

import yaml
import subprocess

def inpaint(file_name):
  subprocess.call(["sed -i 's/offscreen_rendering: True/offscreen_rendering: False/g' Inpainting/argument.yml"],shell=True)

  with open("Inpainting/argument.yml") as f:
    list_doc = yaml.load(f)

  list_doc['src_folder'] = 'Input'
  list_doc['depth_folder'] = 'Output'
  list_doc['require_midas'] = True

  list_doc['specific'] = file_name.split('.')[0]

  with open("Inpainting/argument.yml", "w") as f:
      yaml.dump(list_doc, f)

  # command line arguments
  config = yaml.load(open('Inpainting/argument.yml', 'r'))
  if config['offscreen_rendering'] is True:
      vispy.use(app='egl')

  # create some directories  
  os.makedirs(config['mesh_folder'], exist_ok=True)
  os.makedirs(config['video_folder'], exist_ok=True)
  os.makedirs(config['depth_folder'], exist_ok=True)
  sample_list = get_MiDaS_samples(config['src_folder'], config['depth_folder'], config, config['specific']) # dict of important stuffs
  normal_canvas, all_canvas = None, None

  # find device
  if isinstance(config["gpu_ids"], int) and (config["gpu_ids"] >= 0):
      device = config["gpu_ids"]
      os.environ["CUDA_VISIBLE_DEVICES"] = "0"
  else:
    device = "cpu"

  print(f"running on device {device}")

  # iterate over each image.
  for idx in tqdm(range(len(sample_list))):
      depth = None
      sample = sample_list[idx] # select image
      print("Current Source ==> ", sample['src_pair_name'])
      mesh_fi = os.path.join(config['mesh_folder'], sample['src_pair_name'] +'.ply')
      image = imageio.imread(sample['ref_img_fi'])

      print(f"Running depth extraction at {time.time()}")
      if config['require_midas'] is True:
          run_depth([sample['ref_img_fi']], config['src_folder'], config['depth_folder'], # compute depth 
                    config['MiDaS_model_ckpt'], MonoDepthNet, MiDaS_utils, target_w=1280)
      if 'npy' in config['depth_format']:
          config['output_h'], config['output_w'] = np.load(sample['depth_fi']).shape[:2]
      else:
          config['output_h'], config['output_w'] = imageio.imread(sample['depth_fi']).shape[:2]

      frac = config['longer_side_len'] / max(config['output_h'], config['output_w'])
      config['output_h'], config['output_w'] = int(config['output_h'] * frac), int(config['output_w'] * frac)
      config['original_h'], config['original_w'] = config['output_h'], config['output_w']
      if image.ndim == 2:
          image = image[..., None].repeat(3, -1)
      if np.sum(np.abs(image[..., 0] - image[..., 1])) == 0 and np.sum(np.abs(image[..., 1] - image[..., 2])) == 0:
          config['gray_image'] = True
      else:
          config['gray_image'] = False

      image = cv2.resize(image, (config['output_w'], config['output_h']), interpolation=cv2.INTER_AREA)

      depth = read_MiDaS_depth(sample['depth_fi'], 3.0, config['output_h'], config['output_w']) # read normalized depth computed 

      mean_loc_depth = depth[depth.shape[0]//2, depth.shape[1]//2]

      if not(config['load_ply'] is True and os.path.exists(mesh_fi)):
          vis_photos, vis_depths = sparse_bilateral_filtering(depth.copy(), image.copy(), config, num_iter=config['sparse_iter'], spdb=False) # do bilateral filtering
          depth = vis_depths[-1]
          model = None
          torch.cuda.empty_cache()
          
          ## MODEL INITS

          print("Start Running 3D_Photo ...")
          print(f"Loading edge model at {time.time()}")
          depth_edge_model = Inpaint_Edge_Net(init_weights=True) # init edge inpainting model
          depth_edge_weight = torch.load(config['depth_edge_model_ckpt'],
                                        map_location=torch.device(device))
          depth_edge_model.load_state_dict(depth_edge_weight)
          depth_edge_model = depth_edge_model.to(device)
          depth_edge_model.eval() # in eval mode

          print(f"Loading depth model at {time.time()}")
          depth_feat_model = Inpaint_Depth_Net() # init depth inpainting model
          depth_feat_weight = torch.load(config['depth_feat_model_ckpt'],
                                        map_location=torch.device(device))
          depth_feat_model.load_state_dict(depth_feat_weight, strict=True)
          depth_feat_model = depth_feat_model.to(device)
          depth_feat_model.eval()
          depth_feat_model = depth_feat_model.to(device)

          print(f"Loading rgb model at {time.time()}") # init color inpainting model
          rgb_model = Inpaint_Color_Net()
          rgb_feat_weight = torch.load(config['rgb_feat_model_ckpt'],
                                      map_location=torch.device(device))
          rgb_model.load_state_dict(rgb_feat_weight)
          rgb_model.eval()
          rgb_model = rgb_model.to(device)
          graph = None


          print(f"Writing depth ply (and basically doing everything) at {time.time()}")
          # do some mesh work
          starty=time.time()
          rt_info = write_ply(image, 
                                depth,
                                sample['int_mtx'],
                                mesh_fi,
                                config,
                                rgb_model,
                                depth_edge_model,
                                depth_edge_model,
                                depth_feat_model)

          if rt_info is False:
              continue
          rgb_model = None
          color_feat_model = None
          depth_edge_model = None
          depth_feat_model = None
          torch.cuda.empty_cache()
      print(f'Total Time taken: {time.time()-starty}')
      if config['save_ply'] is True or config['load_ply'] is True:
          verts, colors, faces, Height, Width, hFov, vFov = read_ply(mesh_fi) # read from whatever mesh thing has done
      else:
          verts, colors, faces, Height, Width, hFov, vFov = rt_info

      startx = time.time()
      print(f"Making video at {time.time()}")
      videos_poses, video_basename = copy.deepcopy(sample['tgts_poses']), sample['tgt_name']
      top = (config.get('original_h') // 2 - sample['int_mtx'][1, 2] * config['output_h'])
      left = (config.get('original_w') // 2 - sample['int_mtx'][0, 2] * config['output_w'])
      down, right = top + config['output_h'], left + config['output_w']
      border = [int(xx) for xx in [top, down, left, right]]
      normal_canvas, all_canvas = output_3d_photo(verts.copy(), colors.copy(), faces.copy(), copy.deepcopy(Height), copy.deepcopy(Width), copy.deepcopy(hFov), copy.deepcopy(vFov),
                          copy.deepcopy(sample['tgt_pose']), sample['video_postfix'], copy.deepcopy(sample['ref_pose']), copy.deepcopy(config['video_folder']),
                          image.copy(), copy.deepcopy(sample['int_mtx']), config, image,
                          videos_poses, video_basename, config.get('original_h'), config.get('original_w'), border=border, depth=depth, normal_canvas=normal_canvas, all_canvas=all_canvas,
                          mean_loc_depth=mean_loc_depth)
      print(f"Total Time taken: {time.time()-startx}")