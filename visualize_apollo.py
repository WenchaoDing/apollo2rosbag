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
cameras     = ['/Camera 5/'] 
#cameras     = ['/Camera 5/', '/Camera 6/'] 
label_path  = './utilities/labels.pickle'

print ('***** reading dataset from dir: ', dataset_dir)
print ('***** reading scene: ', scene)
print ('***** level: ', level)
print ('***** using camera: ', cameras)
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
vis_resize_coeff = 0.07
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


def read_image(recordname, prefix, pendix, cameras):
  #read color image from dataset/scene/level/record/cameras
  img_list = []
  #iterate throught all the cameras
  for camera in cameras:
    #overwrite camera name
    camerastr = list(camera)
    recordstr = list(recordname)
    recordstr[-1] = camerastr[-2]
    recordname = ''.join(recordstr) 

    #generate name
    imagename = prefix + camera + recordname + pendix

    #use opencv to get the image
    img = cv2.imread(imagename)
    if img is not None:
      img_list.append(img)
    # else:
    #   print ('ERROR: Cannot read image:', imagename)

  return img_list


def read_label_image(recordname, prefix, cameras):
  labelimg_list = []

  for camera in cameras:
    camerastr = list(camera)
    recordstr = list(recordname)
    recordstr[-1] = camerastr[-2]
    recordname = ''.join(recordstr) 

    #generate name
    labelimg_name = prefix + camera + recordname + '.png'

    #use opencv to get the image
    labelimg = cv2.imread(labelimg_name)
    if labelimg is not None:
      labelimg_list.append(labelimg)
    else:
      #there are two different naming conventions
      labelimg_name = prefix + camera + recordname + '_bin.png'
      labelimg = cv2.imread(labelimg_name)
      if labelimg is not None:
        labelimg_list.append(labelimg)
#      else:
        #print ('ERROR: Cannot read label image:', labelimg_name)
  return labelimg_list


for record_id in available_recordids[2:]:
  #get record folder name
  print ('>>>>>> Current record: ', record_id)
  record_string = 'Record' + '{0:03}'.format(record_id)

  #use arbitrary camera name to get all the colorimg_names
  colorimg_dir = dataset_dir + scene + '_' + level + '/ColorImage/' + record_string + cameras[0]
  colorimg_names = get_filenames_from_dir(colorimg_dir)
  
  #collect all the records names
  records_list = []
  for colorimg_name in colorimg_names:
    recordname   = os.path.splitext(os.path.basename(colorimg_name))[0]
    records_list.append(recordname) 

  #get pose records
  pose_dfs = []
  for camera in cameras:
    pose_dir   = dataset_dir + scene + '_' + level + '/Pose/' + record_string + camera
    poses_name = pose_dir + 'pose.txt' 

    pose_df = pd.read_csv(poses_name, delimiter =' ', header = None, names=['r00', 'r01', 'r02', 't0', 'r10', 'r11', 'r12', 't1', 'r20', 'r21', 'r22', 't2', 'p40', 'p41', 'p42', 'p43', 'image_name'])
    pose_df.set_index('image_name')
    pose_dfs.append(pose_df)
  
  print('Finish reading all the posed for this record.', len(pose_dfs) )

  #iterate through records
  for recordname in records_list:
    img_list = []

    #read color image
    colorprefix = dataset_dir + scene + '_' + level + '/ColorImage/' + record_string
    colorimg_list = read_image(recordname, colorprefix, '.jpg', cameras)
    img_list += colorimg_list

    #read labels 
    labelprefix = dataset_dir + scene + '_' + level + '/Label/' + record_string
    labelimg_list = read_label_image(recordname, labelprefix, cameras)
    img_list += labelimg_list 

    #read depth
    depthprefix = dataset_dir + scene + '_' + level + '_depth/' + record_string
    depthimg_list = read_image(recordname, depthprefix, '.png', cameras)
    img_list += depthimg_list

    #read pose
    poses = []
    for pose_df in pose_dfs:
      select_pose = pose_df[pose_df['image_name'].str.contains(recordname)]
      pose_tuple  = (select_pose['t0'].values[0], select_pose['t1'].values[0], select_pose['t2'].values[0])
      poses.append(pose_tuple)

    #resize and visualize
    vis_img = None
    for img in img_list:
      resized = cv2.resize(img, (0,0), fx=vis_resize_coeff, fy=vis_resize_coeff) 
      if vis_img is not None:
        vis_img = np.concatenate((vis_img, resized), axis=1)
      else:
        vis_img = resized
    
    cv2.imshow('all in one', vis_img)
    k = cv2.waitKey(1)

    if k == 27:
      user_interupt = True
      break

  if user_interupt:
    break