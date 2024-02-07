import matplotlib.pyplot as plt
from rosbag_as_dataset import RosbagReader, MissingTopicError, get_topics_in_path
from ros_to_numpy_helpers import (ros_pose_to_np_se3_matrix,
                                                     ros_image_to_np,
                                                     se3_mat_to_position_and_quaterion_vec)
import os
import sys
import logging
import pathlib as pl
from copy import deepcopy
from typing import List, Dict, Optional

import cv2
import pickle
import numpy as np
from PIL import Image
import rosbag
import tf2_ros
from ros_tf2_wrapper import Tf2Wrapper, get_populated_tf2_wrapper
from rospy import Time, Duration


def _setup_rosbag_dataset(data_dir, 
                          topics_and_names:dict, 
                          time_slack:Optional[float]=None, 
                          reference_topic:str=None
                          ):
    dataset = RosbagReader(data_dir, topics_and_names, 
                           permissible_asynchronisity_sec=time_slack,
                           reference_topic=reference_topic)
    return dataset


def main():
    bag_path = '/home/nirmal/project/spot_ros_data/2023-12-12-21-43-32.bag'
    bag = rosbag.Bag(bag_path)
    # print(bag)
    type_and_topic = bag.get_type_and_topic_info()[1].keys()
    # print(type_and_topic)
    # wrapper = Tf2Wrapper(cache_time=Duration(60))
    # dataset = RosbagReader(bag_path, {'/tf':'tf'}, permissible_asynchronisity_sec=0.15)
    # d_time = Time(606)
    # for d in dataset:
    #     print(d)
    #     for i, transform in enumerate(d['tf'].transforms):
    #         # print(transform)
    #         wrapper.set(transform, False)
    #         # if i == 0:
    #             #  break
    # # print(wrapper.get('base_footprint', 'camera_rgb_optical_frame', d_time))
    # print(wrapper)
    # for i, (topic, msg, t) in enumerate(bag.read_messages(topics=['/tf'])):
    #     print(msg)
    #     if i == 1:
    #         break
    
    # print(get_topics_in_path(bag_path))
    # bag = RosbagReader(bag_path)
    # print(bag.get_topics())
    # print(bag[0].keys())
    topics_names = {'/spot/camera/hand_color/image': 'image'}
    slack_time = 0.03
    data = _setup_rosbag_dataset(bag_path, topics_names, slack_time, reference_topic='spot/camera/hand_color/image')
    print(len(data))
    # for i in range(0, 300, 20)
        # plt.imshow(reader[i]['camera/rgb/image_raw'])
        # plt.show()


if __name__ == '__main__':
    main()