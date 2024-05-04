import numpy as np
import rosbag
import rospy
from geometry_msgs.msg import PoseArray, PoseStamped, Pose, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
import os
from scipy.spatial.transform import Rotation as R
import tf2_ros 
from ros_tf2_wrapper import get_populated_tf2_wrapper
from sensor_msgs.msg import Image, CameraInfo
from scipy.spatial.transform import Rotation as R

def extract_matrix(tf):
    position = [tf.transform.translation.x,
                tf.transform.translation.y,
                tf.transform.translation.z]
    quat = [tf.transform.rotation.x,
            tf.transform.rotation.y,
            tf.transform.rotation.z,
            tf.transform.rotation.w]
    r = R.from_quat(quat)
    matrix = r.as_matrix()
    print(tf.header.stamp)
    print(position)
    print(matrix)


def main():
    rospy.init_node('tf_listener')
    # bagpath = '/home/nirmal/project/spot_ros_data'
    # input_bag_name = '2023-12-12-21-43-32_backup.bag'
    # input_bag_path = os.path.join(bagpath, input_bag_name)
    # input_bag = rosbag.Bag(input_bag_path)
    # topic_list = input_bag.get_type_and_topic_info()[1].keys()
    # output_bag_name = 'with_tf.bag'
    # output_bag_path = os.path.join(bagpath, output_bag_name)
    # print(topic_list)
    buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(buffer)
    rate = rospy.Rate(10.0)
    
    while not rospy.is_shutdown():
        t = rospy.Time().now()
        tf = buffer.lookup_transform_full('odom', t, 'hand_color_image_sensor', t,  'odom', timeout=rospy.Duration(1.0))
        # print(tf)
        extract_matrix(tf)
        rate.sleep()
        

    # bag_path = '/home/nirmal/project/spot_ros_data/2023-12-12-21-43-32.bag'
    # dynamic_topics =['/tf'] 
    # static_topics = ['/tf_static']
    # tf_wrapper = get_populated_tf2_wrapper(bag_path, static_topics, dynamic_topics)
    # print(tf_wrapper)


if __name__ == "__main__":
    main()