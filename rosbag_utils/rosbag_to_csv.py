import sys
import os
import rosbag
import rospy
import csv

def main():
    bagpath = '/home/nirmal/project/spot_ros_data/2023-12-12-21-43-32_backup.bag'
    bag = rosbag.Bag(bagpath)
    topic_list = bag.get_type_and_topic_info()[1].keys()
    print(topic_list)
    

if __name__ == '__main__':
    main()