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
import random
#apollo official utils for label
from utilities.labels_apollo import Label

#visualize 3d
import matplotlib

dataset_dir = '/media/denny/storage/dataset/apolloscape/'
scene       = 'road02'
level       = 'ins'
#cameras     = ['/Camera 5/'] 
cameras     = ['/Camera 5/', '/Camera 6/'] 
label_path  = './utilities/labels.pickle'

camera_intrisics = {cameras[0]: {'fx': 2304.54786556982, 'fy': 2305.875668062, 'cx':1686.23787612802, 'cy':1354.98486439791 } ,
                    cameras[1]: {'fx': 2304.54786556982, 'fy': 2305.875668062, 'cx':1686.23787612802, 'cy':1354.98486439791 } }

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

def get_recordname_with_shift(recordname, frameshift):
  #shift the original record name
  splits = recordname.split('_')
  stamp2 = int(splits[1]) #get the second split
  stamp2 += frameshift
  splits[1] = str(stamp2)

  newrecordname = splits[0] + '_' + splits[1] + '_' + splits[2] + '_' + splits[3]

  return newrecordname

def get_recordname_with_camera(recordname, camera):
  camerastr = list(camera)
  recordstr = list(recordname)
  recordstr[-1] = camerastr[-2]
  recordname = ''.join(recordstr) 

  return recordname

def get_pose_tuple(pose_df, recordname):
  pose_tuple = None
  select_pose = pose_df[pose_df['image_name'].str.contains(recordname)]

  if select_pose.shape[0] == 0:
    newrecord = get_recordname_with_shift(recordname,1)
    select_pose = pose_df[pose_df['image_name'].str.contains(recordname)]
    if select_pose.shape[0] != 0:
      pose_tuple  = select_pose.values[0][0:16].astype(np.float)
      pose_tuple  =  np.resize(pose_tuple, (4,4))
  else:
    pose_tuple  = select_pose.values[0][0:16].astype(np.float)
    pose_tuple  =  np.resize(pose_tuple, (4,4))

  return pose_tuple


def read_image(recordname, prefix, pendix, cameras, flags = cv2.IMREAD_COLOR):
  #read color image from dataset/scene/level/record/cameras
  img_list = {}
  #iterate throught all the cameras
  for camera in cameras:
    #overwrite camera name
    recordname = get_recordname_with_camera(recordname, camera)

    #generate name
    imagename = prefix + camera + recordname + pendix

    #use opencv to get the image
    img = cv2.imread(imagename, flags)
    if img is not None:
      img_list[camera] = img
    else:
      newrecord = get_recordname_with_shift(recordname, 1)
      img = cv2.imread(imagename)
      if img is not None:
        img_list[camera] = img
    #   print ('ERROR: Cannot read image:', imagename)

  return img_list


def read_label_image(recordname, prefix, cameras):
  labelimg_list = {}

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
      labelimg_list[camera] = labelimg
    else:
      #there are two different naming conventions
      labelimg_name = prefix + camera + recordname + '_bin.png'
      labelimg = cv2.imread(labelimg_name)
      if labelimg is not None:
        labelimg_list[camera] = labelimg
#      else:
        #print ('ERROR: Cannot read label image:', labelimg_name)
  return labelimg_list

def visualize_monocular_3d(camera_pose, semantic_image, depth_image, intrisics):
  #use camera pose (R_w_c, T_w_c) to obtain the 3d semantics in the world frame
  #semantic image and depth image should be of the same size
  h, w, c = semantic_image.shape
  P_w_c = camera_pose

  for u in range(h):
    for v in range(w):

      pixel_id = semantic_image[u, v, 0]
      if pixel_id not in id2label.keys():
        pixel_id = 255

      #calculate colorcode for this label
      label = id2label[pixel_id]
      colorcode = label.color
      r = colorcode // (256*256)
      g = (colorcode-256*256*r) // 256
      b = (colorcode-256*256*r-256*g)
      
      #based on the camera pose and depth image, project this pixel to world coordinate
      P_c = np.zeros((4,1), dtype = np.float)
      depth = depth_image[u,v]
      P_c[0] = (u-intrisics['cx']) * depth/intrisics['fx']
      P_c[1] = (v-intrisics['cy']) * depth/intrisics['fy']
      P_c[2] = depth
      P_c[3] = 1.

      P_w = np.matmul(P_w_c, P_c)



for record_id in available_recordids:
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
  pose_dfs = {}
  for camera in cameras:
    pose_dir   = dataset_dir + scene + '_' + level + '/Pose/' + record_string + camera
    poses_name = pose_dir + 'pose.txt' 

    pose_df = pd.read_csv(poses_name, delimiter =' ', header = None, names=['r00', 'r01', 'r02', 't0', 'r10', 'r11', 'r12', 't1', 'r20', 'r21', 'r22', 't2', 'p40', 'p41', 'p42', 'p43', 'image_name'])
    pose_df.set_index('image_name')
    pose_dfs[camera] = pose_df

  print('Finish reading all the posed for this record, # of pose files', len(pose_dfs.keys()) )

  #iterate through records
  for recordname in records_list:
    img_list = []

    #read color image
    colorprefix = dataset_dir + scene + '_' + level + '/ColorImage/' + record_string
    colorimg_dict = read_image(recordname, colorprefix, '.jpg', cameras)
    img_list.append(colorimg_dict)

    #read labels 
    labelprefix = dataset_dir + scene + '_' + level + '/Label/' + record_string
    labelimg_list = read_label_image(recordname, labelprefix, cameras)
    img_list.append(labelimg_list) 

    #read depth
    depthprefix = dataset_dir + scene + '_' + level + '_depth/' + record_string
    depthimg_list = read_image(recordname, depthprefix, '.png', cameras, flags = cv2.IMREAD_GRAYSCALE)

    #read pose
    poses = {}
    for camera in cameras:
      recordname = get_recordname_with_camera(recordname, camera)
      pose = get_pose_tuple(pose_dfs[camera], recordname)

      if pose is not None:
        poses[camera] = pose

    #do 3d visualization
    for camera in [cameras[0]]:
      if camera in depthimg_list.keys() and camera in labelimg_list.keys() and camera in poses.keys():
        visualize_monocular_3d(poses[camera], labelimg_list[camera], depthimg_list[camera], camera_intrisics[camera])
      else:
        print('Information missing, cannot output 3d projections')


    #resize and visualize
    vis_img = None
    for dic in img_list:
      for camera in cameras:
        if camera in dic.keys():
          img = dic[camera]
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