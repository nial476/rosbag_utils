import json
import numpy as np
import rospy
import rosbag
import os
from rosbag_as_dataset import RosbagReader
from message_filters import ApproximateTimeSynchronizer, Subscriber, SimpleFilter
from sensor_msgs.msg import Image, CameraInfo
from tf2_msgs.msg import TFMessage
from tf2_ros import TransformListener
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
from scipy.spatial.transform import Rotation as R
import json
import datetime
from quaternion import as_rotation_matrix



def Rz(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta), np.cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def Rx(theta):
    return np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta), -np.sin(theta), 0],
        [0, np.sin(theta), np.cos(theta), 0],
        [0, 0, 0, 1]
    ])

def Ry(theta):
    return np.array([
        [np.cos(theta), 0, np.sin(theta), 0],
        [0, 1, 0, 0],
        [-np.sin(theta), 0, np.cos(theta), 0],
        [0, 0, 0, 1]
    ])

def rotate_camera(pose):
    quat =  [-0.464208177088759, 0.4635662471881856, 
                         -0.5336398774941149, 0.5337092691155544]
    position= [[0.4966196631143907], 
               [0.020318435251030054], 
                [0.2872542849219689]]
    r = R.from_quat(quat)
    matrix = r.as_matrix()
    # matrix = as_rotation_matrix(quat)
    t_matrix = np.hstack((matrix, np.array(position)))
    t_matrix = np.vstack((t_matrix, np.array([0.0,0.0,0.0,1.0])))

    # rot_pose = pose @ t_matrix @ Rz(np.pi) @ Ry(np.pi)
    rot_pose = pose @ t_matrix
    print(rot_pose)
    average_pos = np.zeros_like(rot_pose)
    average_pos[:3, 3] = np.array([0.9140401281928979, -2.0105372615147967, 1.2859104469189946]).reshape(3,)
    rot_pose = rot_pose - average_pos
    rot_pose = rot_pose @ Rx(np.pi)
    return rot_pose

def rotate_cam_from_issues(pose):

    def Rx(theta):
      return np.matrix([[ 1, 0            , 0            ],
                        [ 0, np.cos(theta),-np.sin(theta)],
                        [ 0, np.sin(theta), np.cos(theta)]])
    def Ry(theta):
      return np.matrix([[ np.cos(theta), 0, np.sin(theta)],
                        [ 0            , 1, 0            ],
                        [-np.sin(theta), 0, np.cos(theta)]])
    def Rz(theta):
      return np.matrix([[ np.cos(theta), -np.sin(theta), 0 ],
                        [ np.sin(theta), np.cos(theta) , 0 ],
                        [ 0            , 0             , 1 ]])

    quat =  [-0.464208177088759, 0.4635662471881856, 
                         -0.5336398774941149, 0.5337092691155544]
    position= [[0.4966196631143907], 
               [0.020318435251030054], 
                [0.2872542849219689]]
    r = R.from_quat(quat)
    matrix = r.as_matrix()
    # matrix = as_rotation_matrix(quat)
    t_matrix = np.hstack((matrix, np.array(position)))
    t_matrix = np.vstack((t_matrix, np.array([0.0,0.0,0.0,1.0])))

    rot_pose = pose @ t_matrix
    xf_rot = np.eye(4)
    xf_rot[:3,:3] = rot_pose[:3,:3]

    xf_pos = np.eye(4)
    pos = rot_pose[:3,3]
    average_position = [1.195461037970006, -2.253110737665569, 1.2866036216390748]
    # average_position = [0, -2.0105372615147967, 0]
    xf_pos[:3,3] = pos - average_position
    xf_pos[:3,3] = rot_pose[:3,3]
    # Don't ask me...
    extra_xf = np.matrix([
        [-1, 0, 0, 0],
        [ 0, 0, 1, 0],
        [ 0, 1, 0, 0],
        [ 0, 0, 0, 1]])
    # NerF will cycle forward, cycling backward.
    shift_coords = np.matrix([
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]])
    xf = shift_coords @ extra_xf @ xf_pos
    assert np.abs(np.linalg.det(xf) - 1.0) < 1e-4
    xf = xf @ xf_rot
    return xf


def callback(cam_info, cam, body_tf, odom):
    global count
    global output_path
    global transforms
    bridge = CvBridge()
    image = bridge.imgmsg_to_cv2(cam)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    position = [[body_tf.pose.position.x], 
                [body_tf.pose.position.y],
                [body_tf.pose.position.z]]
    quat = [body_tf.pose.orientation.x,
                  body_tf.pose.orientation.y,
                  body_tf.pose.orientation.z,
                  body_tf.pose.orientation.w]
    r = R.from_quat(quat)
    matrix = r.as_matrix()
    # matrix = as_rotation_matrix(quat)
    t_matrix = np.hstack((matrix, np.array(position)))
    t_matrix = np.vstack((t_matrix, np.array([0.0,0.0,0.0,1.0])))
    transform_matrix = rotate_camera(t_matrix)
    # temp = transform_matrix[0, -1]
    # transform_matrix[0, -1] = transform_matrix[1, -1]
    # transform_matrix[1, -1] = temp
    current_frame = {"file_path": "./images/"+str(count)+".jpg",
                     "transform_matrix": transform_matrix.tolist()}
    transforms['frames'].append(current_frame)
    with open(os.path.join(output_path, "transforms.json"), "w") as file:
        json.dump(transforms, file)
    cv2.imwrite(output_path + '/images/' + str(count) + '.jpg', image)
    count += 1

def extract_pose(bag_path, output_path):
    cam_info = Subscriber('/spot/camera/hand_color/camera_info', CameraInfo)
    image = Subscriber('/spot/camera/hand_color/image', Image)
    spot_body = Subscriber('/spot_pose', PoseStamped)
    odom = Subscriber('/spot/odometry', Odometry)
    filter = ApproximateTimeSynchronizer([
        cam_info, image, spot_body, odom
        ],
        queue_size=1000, slop=1
    )
    filter.registerCallback(callback)
    rospy.spin()
    
def main():
    rospy.init_node('test')
    bag_dir = '/home/nirmal/project/spot_ros_data'
    bag_file = 'with_pose.bag'
    bag_path = os.path.join(bag_dir, bag_file)
    file_path = '/home/nirmal/project/spot_ros_data'
    file_name = bag_file[:-4] + str(datetime.datetime.now())
    global output_path
    global count
    global transforms
    transforms = {"camera_angle_x": 1.050688,
    "camera_angle_y": 0.8202161279220551,
    "fl_x": 552.0291012161067,
    "fl_y": 552.0291012161067,
    "k1": 0,
    "k2": 0,
    "k3": 0,
    "k4": 0,
    "p1": 0,
    "p2": 0,
    "is_fisheye": False,
    "cx": 320.0,
    "cy": 240.0,
    "w": 640.0,
    "h": 480.0,
    "aabb_scale": 32,
    "scale": 1,
    "frames": []}
    count = 0
    output_path = os.path.join(file_path, file_name)
    if os.path.exists(output_path):
        print("Directory already exist")
    else:
        print("Creating directory")
        os.makedirs(os.path.join(output_path, "images"))
    extract_pose(bag_path, output_path)

if __name__ == "__main__":
    main()