import tf2_ros
import tf_conversions
from geometry_msgs.msg import TransformStamped
import rospy

def main():
    rospy.init_node('map_transform')
    broadcaster = tf2_ros.StaticTransformBroadcaster()
    ts = TransformStamped()
    ts.header.stamp = rospy.Time.now()
    ts.header.frame_id = "map"
    ts.child_frame_id = "odom"
    ts.transform.translation.x = 0.0
    ts.transform.translation.y = 0.0
    ts.transform.translation.z = 0.0
    ts.transform.rotation.x = 0.0
    ts.transform.rotation.y = 0.0
    ts.transform.rotation.z = 0.0
    ts.transform.rotation.w = 1.0
    broadcaster.sendTransform(ts)
    rospy.spin()


if __name__ == '__main__':
    main()