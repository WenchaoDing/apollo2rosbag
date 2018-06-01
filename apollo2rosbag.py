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
from math import sqrt
#apollo official utils for label
from utilities.labels_apollo import Label

#visualize 3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#import ros utils
import rosbag
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CameraInfo
from nav_msgs.msg import Odometry


def convert_label_class_to_colormap(label_image):
  h, w, c = label_image.shape
  colormap = np.copy(label_image)

  for v in range(h):
    for u in range(w):
      pixel_id = label_image[v, u, 0]
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

def get_filenames_from_dir(colorimg_dir):
  filenames = [img for img in glob.glob(colorimg_dir + '*.*')]
  filenames.sort()
  return filenames

def get_recordname_with_shift(recordname, frameshift):
  #shift the original record name
  splits = recordname.split('_')
  stamp2 = int(splits[1]) #get the second split
  stamp2 += frameshift
  splits[1] = '{0:09d}'.format(stamp2) 
  newrecordname = splits[0] + '_' + splits[1] + '_' + splits[2] + '_' + splits[3]

  return newrecordname

def get_recordname_with_camera(recordname, camera):
  camerastr = list(camera)
  recordstr = list(recordname)
  recordstr[-1] = camerastr[-2]
  recordname = ''.join(recordstr) 

  return recordname

def get_timestamp_from_recordname(recordname, correction = 0.0):
  splits = recordname.split('_')
  stamp  = rospy.Time.from_sec(int(splits[1])/1000. - correction)
  return stamp

def get_pose_tuple(pose_df, recordname):
  pose_tuple = None
  #search with some tolerence in stamp
  for shift in range(-1,2):
    newrecord = get_recordname_with_shift(recordname, shift)
    select_pose = pose_df[pose_df['image_name'].str.contains(newrecord)]
    if select_pose.shape[0] != 0:
      pose_tuple  = select_pose.values[0][0:16].astype(np.float)
      pose_tuple  =  np.resize(pose_tuple, (4,4))
      break

  return pose_tuple

def read_image(recordname, prefix, pendix, cameras, flags = cv2.IMREAD_COLOR):
  #read color image from dataset/scene/level/record/cameras
  img_list = {}
  #iterate throught all the cameras
  for camera in cameras:
    #overwrite camera name
    recordname = get_recordname_with_camera(recordname, camera)
    #search with some tolerence in stamp
    for shift in range(-1, 2):
      newrecordname = get_recordname_with_shift(recordname, shift)
      imagename = prefix + camera + newrecordname + pendix
      img = cv2.imread(imagename, flags)
      if img is not None:
        img_list[camera] = img
        break

  return img_list


def check_camera_data_valid(colorimg_dict, labelimg_dict, depthimg_dict, poses, camera):
  if camera in colorimg_dict.keys() and camera in labelimg_dict.keys() and camera in depthimg_dict.keys() and camera in poses.keys():
    return True
  return False


def record_next_bag(scene, level, record_id, bag_index_this_record):
  bagname = 'bags/' + scene + '_' + level + '_record' + str(record_id) + '_part' + str(bag_index_this_record) + '.bag'
  bag     = rosbag.Bag(bagname, 'w')
  bag_index_this_record += 1 
  return bag, bag_index_this_record

def merge_two_dicts(x, y):
  z = x.copy()   # start with x's keys and values
  z.update(y)    # modifies z with y's keys and values & returns None
  return z

def trace_method(matrix):
  """
  This code uses a modification of the algorithm described in: 
  https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
  which is itself based on the method described here:
  http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
  
  Altered to work with the column vector convention instead of row vectors
  """
  m = matrix.conj().transpose() # This method assumes row-vector and postmultiplication of that vector
  if m[2, 2] < 0:
      if m[0, 0] > m[1, 1]:
          t = 1 + m[0, 0] - m[1, 1] - m[2, 2]
          q = [m[1, 2]-m[2, 1],  t,  m[0, 1]+m[1, 0],  m[2, 0]+m[0, 2]]
      else:
          t = 1 - m[0, 0] + m[1, 1] - m[2, 2]
          q = [m[2, 0]-m[0, 2],  m[0, 1]+m[1, 0],  t,  m[1, 2]+m[2, 1]]
  else:
      if m[0, 0] < -m[1, 1]:
          t = 1 - m[0, 0] - m[1, 1] + m[2, 2]
          q = [m[0, 1]-m[1, 0],  m[2, 0]+m[0, 2],  m[1, 2]+m[2, 1],  t]
      else:
          t = 1 + m[0, 0] + m[1, 1] + m[2, 2]
          q = [t,  m[1, 2]-m[2, 1],  m[2, 0]-m[0, 2],  m[0, 1]-m[1, 0]]

  q = np.array(q)
  q *= 0.5 / sqrt(t);
  return q

def pose_to_rosmsg(extrinsic, intrinsic, h, w):
  #generate camera info msg
  info_msg = CameraInfo()
  info_msg.header.frame_id = '/world'
  info_msg.height          = h
  info_msg.width           = w
  info_msg.K[0]            = intrinsic['fx']
  info_msg.K[1]            = 0.
  info_msg.K[2]            = intrinsic['cx']
  info_msg.K[3]            = 0.
  info_msg.K[4]            = intrinsic['fy']
  info_msg.K[5]            = intrinsic['cy']
  info_msg.K[6]            = 0.
  info_msg.K[7]            = 0.
  info_msg.K[8]            = 1.

  #generate camera pose odometry

  odom_msg = Odometry()
  odom_msg.header.frame_id = '/world'

  odom_msg.pose.pose.position.x  = extrinsic[0,3]
  odom_msg.pose.pose.position.y  = extrinsic[1,3]
  odom_msg.pose.pose.position.z  = extrinsic[2,3]

  R = extrinsic[:3, :3]
  q = trace_method(R)

  odom_msg.pose.pose.orientation.w = q[0]
  odom_msg.pose.pose.orientation.x = q[1]
  odom_msg.pose.pose.orientation.y = q[2]
  odom_msg.pose.pose.orientation.z = q[3]

  return info_msg, odom_msg

def report_missing(colorimg_dict, labelimg_dict,  depthimg_dict, poses, cameras):
  for camera in cameras:
    if camera not in colorimg_dict.keys():
      print('Missing color image for camera {}', camera)

    if camera not in labelimg_dict.keys():
      print('Missing label image for camera {}', camera)

    if camera not in depthimg_dict.keys():
      print('Missing depth image for camera {}', camera)

    if camera not in poses.keys():
      print('Missing pose for camera {}', camera)

if __name__ == "__main__":    

  dataset_dir = '/media/denny/storage/dataset/apolloscape/'
  scene       = 'road02'
  level       = 'ins'
  #cameras     = ['/Camera 5/'] 
  cameras     = ['/Camera 5/', '/Camera 6/'] 
  label_path  = './utilities/labels.pickle'
  max_record_id = 100

  camera_intrisics = {cameras[0]: {'fx': 2304.54786556982, 'fy': 2305.875668062, 'cx':1686.23787612802, 'cy':1354.98486439791 } ,
                      cameras[1]: {'fx': 2300.39065314361, 'fy': 2301.31478860597, 'cx':1713.21615190657, 'cy':1342.91100799715 } }

  print ('==================================================')
  print ('reading dataset from dir: ', dataset_dir)
  print ('reading scene: ', scene)
  print ('level: ', level)
  print ('using camera: ', cameras)
  print ('==================================================')

  bags_save_dir = 'bags/'
  if not os.path.exists(bags_save_dir):
    os.makedirs(bags_save_dir)

  available_recordids = []
  for record_id in range(1, max_record_id):
    record_string = 'Record' + '{0:03}'.format(record_id)
    record_folder = dataset_dir + scene + '_' + level + '_depth/' + record_string
    #util record not exist anymore
    if not os.path.isdir(record_folder):
      continue;
    available_recordids.append(record_id)

  print ('Available record ids: ')
  print (available_recordids)

  vis_resize_coeff = 0.07
  user_interupt = False

  #get label semantics
  if (os.path.exists(label_path)):
    label_data = pickle.load(open(label_path, 'rb'))

  name2label      = label_data[0]
  id2label        = label_data[1]
  trainId2label   = label_data[2] 
  category2labels = label_data[3] 
  color2label     = label_data[4]

  #iterate through records
  bridge = CvBridge()

  for record_id in available_recordids:
    #for this record, start with index 1
    bag_index_this_record = 1
    bag, bag_index_this_record = record_next_bag(scene, level, record_id, bag_index_this_record)
    bag_ticked  = True
    last_write_stamp = None
    correction  = 0. 

    #get record folder name
    print ('>>>>>> Current record: ', record_id)
    record_string = 'Record' + '{0:03}'.format(record_id)

    #scan the depth folder to get all the file names, depth is the most incomplete
    for camera in cameras:
      img_dir = dataset_dir + scene + '_' + level + '_depth/' + record_string + camera
      img_names = get_filenames_from_dir(img_dir)
      if len(img_names) != 0:
        break    

    #collect all the records names
    records_list = []
    for img_name in img_names:
      recordname   = os.path.splitext(os.path.basename(img_name))[0]
      records_list.append(recordname) 

    #get pose records
    pose_dfs = {}
    for camera in cameras:
      pose_dir   = dataset_dir + scene + '_' + level + '/Pose/' + record_string + camera
      poses_name = pose_dir + 'pose.txt' 

      pose_df = pd.read_csv(poses_name, delimiter =' ', header = None, names=['r00', 'r01', 'r02', 't0', 'r10', 'r11', 'r12', 't1', 'r20', 'r21', 'r22', 't2', 'p40', 'p41', 'p42', 'p43', 'image_name'])
      pose_df.set_index('image_name')
      pose_dfs[camera] = pose_df
    print('Finish reading all the poses for this record, # of pose files', len(pose_dfs.keys()) )


    #iterate through records
    for recordname in records_list:
      img_list = []
      print ('Scanning record (including all the cameras)', recordname)

      #read color image
      colorprefix = dataset_dir + scene + '_' + level + '/ColorImage/' + record_string
      colorimg_dict = read_image(recordname, colorprefix, '.jpg', cameras)
      img_list.append(colorimg_dict)

      #read labels 
      labelprefix = dataset_dir + scene + '_' + level + '/Label/' + record_string
      labelimg_dict = read_image(recordname, labelprefix,  '.png', cameras)
      if len(labelimg_dict.keys()) == 0:
        labelimg_dict = read_image(recordname, labelprefix,  '_bin.png', cameras)
      elif len(labelimg_dict.keys()) < len(cameras):
        add_dict = read_image(recordname, labelprefix,  '_bin.png', cameras)        
        if len(add_dict.keys()) > 0:
          labelimg_dict = merge_two_dicts(labelimg_dict, add_dict)

      img_list.append(labelimg_dict) 

      #read depth
      depthprefix = dataset_dir + scene + '_' + level + '_depth/' + record_string
      depthimg_dict = read_image(recordname, depthprefix, '.png', cameras, flags = cv2.IMREAD_GRAYSCALE)

      #read instance
      insprefix   = dataset_dir + scene + '_' + level + '/Label/' + record_string
      insimg_dict = read_image(recordname, insprefix, '_instanceIds.png', cameras)

      #read pose
      poses = {}
      for camera in cameras:
        newrecordname = get_recordname_with_camera(recordname, camera)
        pose = get_pose_tuple(pose_dfs[camera], newrecordname)

        if pose is not None:
          poses[camera] = pose

      #if record is complete, add to the current bag file
      write_valid = False
      for camera in cameras:
        if check_camera_data_valid(colorimg_dict, labelimg_dict, depthimg_dict, poses, camera):
          write_valid = True


      if write_valid:
        #just write to the current bag, and reset tick state
        bag_ticked = False

        for camera in cameras:
          if check_camera_data_valid(colorimg_dict, labelimg_dict, depthimg_dict, poses, camera):
            #if data is valid for this camera, just write the messages to bag
            newrecordname = get_recordname_with_camera(recordname, camera)
            stamp = get_timestamp_from_recordname(newrecordname, correction)
            
            if last_write_stamp is not None:
              time_diff = (stamp - last_write_stamp).to_sec()
              print ('Stamp difference: {}s'.format(time_diff) )

              if time_diff > 10.0:
                correction += time_diff - 0.15 #assume typical delay is 0.15s
                stamp = get_timestamp_from_recordname(newrecordname, correction)

            print ('Writing camera {} message with stamp {}'.format(camera, stamp) )

            topicname = list(camera)
            del(topicname[7])
            topicname = ''.join(topicname)

            #get colorimg msg
            colormsg = bridge.cv2_to_imgmsg(colorimg_dict[camera], encoding="bgr8")
            colormsg.header.stamp    = stamp
            colormsg.header.frame_id = topicname 
            #get labelimg msg
            labelmsg = bridge.cv2_to_imgmsg(labelimg_dict[camera], encoding="bgr8")            
            labelmsg.header.stamp    = stamp
            labelmsg.header.frame_id = topicname
            #get depth
            depthmsg = bridge.cv2_to_imgmsg(depthimg_dict[camera], encoding="mono8")            
            depthmsg.header.stamp    = stamp
            depthmsg.header.frame_id = topicname
            #get instance 

            if camera in insimg_dict.keys():
              insmsg = bridge.cv2_to_imgmsg(insimg_dict[camera], encoding="bgr8")            
              insmsg.header.stamp    = stamp
              insmsg.header.frame_id = topicname
              bag.write(topicname + 'instance/', insmsg, stamp)
              #print ('putting in instance')

            #get camera info and odom msg
            h, w = depthimg_dict[camera].shape
            info_msg, pose_msg = pose_to_rosmsg(poses[camera], camera_intrisics[camera], h, w)
            info_msg.header.stamp = stamp
            pose_msg.header.stamp = stamp

            #write the messages
            bag.write(topicname + 'color/', colormsg, stamp)
            bag.write(topicname + 'label/', labelmsg, stamp)
            bag.write(topicname + 'depth/', depthmsg, stamp)
            bag.write(topicname + 'info/',  info_msg, stamp)
            bag.write(topicname + 'pose/',  pose_msg, stamp)

            last_write_stamp = stamp

            bag_size = bag.size/1024/1024/1024 
            print ('Current bag size: {}(GB)'.format(bag_size))
            # if bag_size >= 12.:
            #   bag.close()
            #   bag, bag_index_this_record = record_next_bag(scene, level, record_id, bag_index_this_record)
            #   last_write_stamp = None
          else:
            report_missing(colorimg_dict, labelimg_dict,  depthimg_dict, poses, [camera])

      else:
        #if write not valid, check whether next bag has been ticked
        print ('------------------>No valid record found <--------------')
        report_missing(colorimg_dict, labelimg_dict,  depthimg_dict, poses, cameras)

        #if found record missing in the middle, start a new bag file
        if bag_ticked == False:
          #move to next bag
          bag.close()
          bag, bag_index_this_record = record_next_bag(scene, level, record_id, bag_index_this_record)
          bag_ticked        = True
          last_write_stamp  = None

      #resize and visualize
      vis_img = None
      for dic in img_list:
        for camera in cameras:
          resized = None

          if camera in dic.keys():
            img = dic[camera]
            resized = cv2.resize(img, (0,0), fx=vis_resize_coeff, fy=vis_resize_coeff) 

          if resized is None:
            continue 
          
          if vis_img is not None:
            vis_img = np.concatenate((vis_img, resized), axis=1)
          else:
            vis_img = resized
      try:
        cv2.imshow('all in one', vis_img)
        k = cv2.waitKey(1)
      except KeyboardInterrupt:      
        user_interupt = True
        bag.close()
        break

      if k == 27:
        user_interupt = True
        break

    print ('closing the bag.')
    bag.close()
    if user_interupt:
      break