import numpy as np
import rosbag
import cv2
import rospy
from geometry_msgs.msg import PoseArray, PoseStamped, Pose, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
import os
# import tf2
from scipy.spatial.transform import Rotation as R

def norm_pos(pos):
    mag = np.sqrt(pos.x**2 + pos.y**2 + pos.z**2)
    # out =
    return mag 

def norm_quat(quat):
    mag = np.sqrt(quat.x**2 + quat.y**2 + quat.z**2 + quat.w**2)
    return mag

def extract_pose_from_tf(element:TransformStamped):
            p = PoseStamped()
            p.header.seq = element.header.seq
            p.header.stamp.secs = element.header.stamp.secs
            p.header.stamp.nsecs = element.header.stamp.nsecs
            p.header.frame_id = element.header.frame_id
            p.pose.position.x = element.transform.translation.x 
            p.pose.position.y = element.transform.translation.y 
            p.pose.position.z = element.transform.translation.z 
            p.pose.orientation.x = element.transform.rotation.x 
            p.pose.orientation.y = element.transform.rotation.y 
            p.pose.orientation.z = element.transform.rotation.z 
            p.pose.orientation.w = element.transform.rotation.w 
            return p

def generate_and_vizualize_particles(p:PoseStamped):
    poses = []
    noise_pos = np.random.normal(0, 0.1, (5,3))
    noise_ang = R.random(5).as_quat()
    for i in range(5):
        particles = Pose()
        particles.position.x = p.pose.position.x + noise_pos[i, 0]
        particles.position.y = p.pose.position.y + noise_pos[i, 1]
        particles.position.z = p.pose.position.z + noise_pos[i, 2]
        particles.orientation.x = p.pose.orientation.x + noise_ang[i, 0]
        particles.orientation.y = p.pose.orientation.y + noise_ang[i, 1]
        particles.orientation.z = p.pose.orientation.z + noise_ang[i, 2]
        particles.orientation.w = p.pose.orientation.w + noise_ang[i, 3]
        poses.append(particles)
        poses_array = PoseArray()
        poses_array.header.frame_id = 'body'
        poses_array.poses = poses
    return poses_array
    
def vizualize_path(p:PoseStamped, element:TransformStamped,i:int):
    pose = Marker()
    ros_tf = Marker()
    pose.header.frame_id = p.header.frame_id
    pose.header.stamp = p.header.stamp
    pose.ns = 'spot_pose'
    pose.id = i
    pose.type = Marker.SPHERE
    pose.action = Marker.ADD
    pose.pose = p.pose
    pose.scale.x = 0.1
    pose.scale.y = 0.1
    pose.scale.z = 0.1
    pose.color.a = 1.0
    pose.color.r = 1.0
    pose.color.g = 0.0
    pose.color.b = 0.0

    ros_tf.header.frame_id = element.header.frame_id
    ros_tf.header.stamp = element.header.stamp
    ros_tf.ns = 'spot_tf_pose'
    ros_tf.id = i
    ros_tf.type = Marker.SPHERE
    ros_tf.action = Marker.ADD
    ros_tf.pose.position = element.transform.translation
    ros_tf.pose.orientation = element.transform.rotation
    ros_tf.scale.x = 0.1
    ros_tf.scale.y = 0.1
    ros_tf.scale.z = 0.1
    ros_tf.color.a = 1.0
    ros_tf.color.r = 0.0
    ros_tf.color.g = 0.0
    ros_tf.color.b = 1.0
    return pose, ros_tf

def publish_map():
    
    pass


def main():
    bagpath = '/home/nirmal/project/spot_ros_data'
    input_bag_name = '2023-12-12-21-43-32_backup.bag'
    input_bag_path = os.path.join(bagpath, input_bag_name)
    input_bag = rosbag.Bag(input_bag_path)
    topic_list = input_bag.get_type_and_topic_info()[1].keys()
    output_bag_name = 'with_pose.bag'
    output_bag_path = os.path.join(bagpath, output_bag_name)
    print(topic_list)
    viz_marker = False
    with rosbag.Bag(output_bag_path, 'w') as output_bag:
        for i,(topic, msg, t) in enumerate(input_bag.read_messages()):
            if viz_marker:
                marker_pose_array = MarkerArray()
                marker_tf_array = MarkerArray()
            output_bag.write(topic, msg, t)
            #adding map to odom transform as a tf_static
            # if topic == '/tf_static':
            #     for element in msg.transforms:
            #         if element.child_frame_id == 'base_arm_link':
            #             map_to_odom = TransformStamped()
            #             map_to_odom.header = element.header
            #             map_to_odom.header.frame_id = 'map'
            #             map_to_odom.child_frame_id = 'odom'
            #             map_to_odom.transform.translation.x = 0.0
            #             map_to_odom.transform.translation.y = 0.0
            #             map_to_odom.transform.translation.z = 0.0
            #             map_to_odom.transform.rotation.x = 0.0
            #             map_to_odom.transform.rotation.y = 0.0
            #             map_to_odom.transform.rotation.z = 0.0
            #             map_to_odom.transform.rotation.w = 1.0
            #             output_bag.write('/tf_static', map_to_odom, t)
            if topic == '/tf':
                for element in msg.transforms:
                    if element.child_frame_id == 'body':
                        p = extract_pose_from_tf(element)
                        if viz_marker:
                            spot_pose, ros_tf = vizualize_path(p, element, i)
                            marker_pose_array.markers.append(spot_pose)
                            marker_tf_array.markers.append(ros_tf)
                        output_bag.write('/spot_pose', p, t)           

            if topic == '/spot/camera/hand_color/image':
                poses_array = generate_and_vizualize_particles(p)
                output_bag.write('/spot_particles', poses_array, t)   

            if viz_marker:
                output_bag.write('/spot_pose_marker', marker_pose_array, t)
                output_bag.write('/spot_tf_marker', marker_tf_array, t)   
            


if __name__ == '__main__':
    main()