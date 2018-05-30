#!/usr/bin/env python
# -*- coding: utf-8 -*-

#for reading images and proces image
from __future__ import print_function

#for recordname retrieve
import glob
#for visualization
import cv2
#for system utils
import os
import numpy as np
import pickle
import pandas as pd
#apollo official utils for label
from utilities.labels_apollo import Label

dataset_dir = '/media/denny/storage/dataset/apolloscape/'
scene       = 'road02'
level       = 'ins'
camera      = '/Camera 5/'
label_path  = './utilities/labels.pickle'

print ('***** reading dataset from dir: ', dataset_dir)
print ('***** reading scene: ', scene)
print ('***** level: ', level)
print ('***** using camera: ', camera)
max_record_id = 100

def get_filenames_from_dir(colorimg_dir):
  filenames = [img for img in glob.glob(colorimg_dir + '*.jpg')]
  filenames.sort()
  return filenames

available_recordids = []
for record_id in range(1, max_record_id):
  record_string = 'Record' + '{0:03}'.format(record_id)
  record_folder = dataset_dir + scene + '_' + level + '/ColorImage/' + record_string
  #util record not exist anymore
  if not os.path.isdir(record_folder):
    continue;
  available_recordids.append(record_id)

print ('Available record ids: ')
print (available_recordids)

#visualize sequentially
#cv2.namedWindow('three_in_one', cv2.WINDOW_NORMAL)
vis_resize_coeff = 0.1
user_interupt = False

if (os.path.exists(label_path)):
  label_data = pickle.load(open(label_path, 'rb'))

name2label      = label_data[0]
id2label        = label_data[1]
trainId2label   = label_data[2] 
category2labels = label_data[3] 
color2label     = label_data[4]

def convert_label_class_to_colormap(label_image):
  h, w, c = label_image.shape
  colormap = np.copy(label_image)

  for u in range(h):
    for v in range(w):
      pixel_id = label_image[u, v, 0]
      if pixel_id not in id2label.keys():
        pixel_id = 255

      #calculate colorcode for this label
      label = id2label[pixel_id]
      colorcode = label.color
      r = colorcode // (256*256)
      g = (colorcode-256*256*r) // 256
      b = (colorcode-256*256*r-256*g)
      
      #opencv use bgr
      colormap[u,v,0] = b
      colormap[u,v,1] = g
      colormap[u,v,2] = r

  return colormap

for record_id in available_recordids:
  print ('>>>>>> Current record: ', record_id)

  record_string = 'Record' + '{0:03}'.format(record_id)
  colorimg_dir = dataset_dir + scene + '_' + level + '/ColorImage/' + record_string + camera
  label_dir    = dataset_dir + scene + '_' + level + '/Label/'      + record_string + camera
  depth_dir    = dataset_dir + scene + '_' + 'depth/'               + record_string + camera
  pose_dir     = dataset_dir + scene + '_' + level + '/Pose/'       + record_string + camera

  colorimg_names = get_filenames_from_dir(colorimg_dir)
  poses_name     = pose_dir + 'pose.txt'

  pose_df = pd.read_csv(poses_name, delimiter =' ', header = None, names=['r00', 'r01', 'r02', 't0', 'r10', 'r11', 'r12', 't1', 'r20', 'r21', 'r22', 't2', 'p40', 'p41', 'p42', 'p43', 'image_name'])
  pose_df.set_index('image_name')

  print (pose_df)

  for colorimg_name in colorimg_names:
    image_list     = [] #original images 
    vis_image_list = [] #images for visualization

    colorimg = cv2.imread(colorimg_name)
    if colorimg is not None:
      resized_color = cv2.resize(colorimg, (0,0), fx=vis_resize_coeff, fy=vis_resize_coeff) 
      image_list.append(colorimg)
      vis_image_list.append(resized_color)
    else:
      print ('ERROR: Cannot read color image:', colorimg_name)

    #get file name
    recordname = os.path.splitext(os.path.basename(colorimg_name))[0]
    
    #get corresponding segmentation image
    binlabelimg_name = label_dir + recordname + '_bin.png'
    binimg = cv2.imread(binlabelimg_name)

    if binimg is not None:
      resized_bin = cv2.resize(binimg, (0,0), fx=vis_resize_coeff, fy=vis_resize_coeff) 
      #color_bin = convert_label_class_to_colormap(resized_bin)
      image_list.append(binimg)
      vis_image_list.append(resized_bin)
    else:
      print ('ERROR: Cannot find bin for:', recordname)

    #get depth image
    depthimg = cv2.imread(depth_dir+recordname + '.png')
    if depthimg is not None:
      resized_depth = cv2.resize(depthimg, (0,0), fx=vis_resize_coeff, fy=vis_resize_coeff) 
      image_list.append(depthimg)
      vis_image_list.append(resized_depth)
    else:
      print ('ERROR: Cannot find depth for:', depth_dir + recordname)
    
    #get instance image and polygons (if any)
    inslabelimg_name = label_dir + recordname + '_instanceIds.png'
    insimg = cv2.imread(inslabelimg_name)
    if insimg is not None:
      resized_ins = cv2.resize(insimg, (0,0), fx=vis_resize_coeff, fy=vis_resize_coeff) 
      image_list.append(insimg)
      vis_image_list.append(resized_ins)
    else:
      print ('No instance', inslabelimg_name)

    #get pose of this image
    #pose = pose_df[pose_df[:,16] == recordname]
    select_pose = pose_df[pose_df['image_name'].str.contains(recordname)]
    print (select_pose['t0'].values[0], select_pose['t1'].values[0], select_pose['t2'].values[0])

    # concatenate visualization image
    vis_img = vis_image_list[0]
    for i in range(1, len(vis_image_list)):
      vis_img = np.concatenate((vis_img, vis_image_list[i]), axis=1)

    cv2.imshow('three_in_one', vis_img)
    k = cv2.waitKey(1)

    if k == 27: #wait for ESC key to exit
      cv2.destroyAllWindows()
      user_interupt = True
      break;


  if user_interupt:
    break;




