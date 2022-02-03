#!/usr/bin/env python3

import sys
from numpy.core.fromnumeric import size
import json

from perception.utils import ros
from yaml.error import Mark
import rospy
import numpy as np
import cv2
import tf2_ros
from geometry_msgs.msg import PointStamped, PoseStamped, PoseArray, Pose
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


class Keypoints3dVisualizer:

    def __init__(self):
        rospy.loginfo("self.kp_markers node init")
        self.kp_marker_array = MarkerArray()
        self.kp_marker_array_publisher = rospy.Publisher("/kp_markers", MarkerArray, queue_size=10)
        self.kp_subscriber = rospy.Subscriber("/kp_3d_poses", PoseArray, self.callback, queue_size=1)

    def update_kp_marker(self, msg):
        for i in range(self.kp_num):
            self.kp_marker = Marker()
            # self.kp_marker.header.stamp  = rospy.Time.now()
            self.kp_marker.header.stamp = msg.header.stamp
            self.kp_marker.header.frame_id = msg.header.frame_id
            self.kp_marker.type = Marker.SPHERE
            self.kp_marker.action = Marker.ADD
            self.kp_marker.scale.x = 0.01
            self.kp_marker.scale.y = 0.01
            self.kp_marker.scale.z = 0.01
            self.kp_marker.color.a = 1.0
            self.kp_marker.color.r = 1.0
            self.kp_marker.color.g = 0.0
            self.kp_marker.color.b = 0.0
            self.kp_marker.pose.position.x = msg.poses[i].position.x
            self.kp_marker.pose.position.y = msg.poses[i].position.y
            self.kp_marker.pose.position.z = msg.poses[i].position.z
            self.kp_marker.pose.orientation.w = 1.0
            self.kp_marker.id = i
            self.kp_marker_array.markers.append(self.kp_marker)

    def callback(self, msg):
        self.kp_marker_array.markers.clear()
        self.kp_num = len(msg.poses)
        self.update_kp_marker(msg)
        self.kp_marker_array_publisher.publish(self.kp_marker_array)

if __name__ == "__main__":
    rospy.init_node("kp_3d_makers_vis_node")
        
    kp_vis = Keypoints3dVisualizer()
    rospy.spin()
   