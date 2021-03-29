#!/usr/bin/env python3
import json

import rospy
import message_filters
import torch
import numpy as np
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from perception.utils import ros as ros_utils
from sensor_msgs.msg import Image
from perception import pipeline
from perception.utils import camera_utils
from matplotlib import cm

def _to_msg(keypoint, time, frame):
    msg = PointStamped()
    msg.header.stamp = time
    msg.header.frame_id = frame
    msg.point.x = keypoint[0]
    msg.point.y = keypoint[1]
    msg.point.z = keypoint[2]
    return msg

class ObjectKeypointPipeline:
    def __init__(self):
        left_image_topic = rospy.get_param("object_keypoints_ros/left_image_topic", "/zedm/zed_node/left_raw/image_raw_color")
        right_image_topic = rospy.get_param("object_keypoints_ros/right_image_topic", "/zedm/zed_node/right_raw/image_raw_color")
        self.left_camera_frame = rospy.get_param('object_keypoints_ros/left_camera_frame')
        self.left_sub = rospy.Subscriber(left_image_topic, Image, callback=self._right_image_callback, queue_size=1)
        self.right_sub = rospy.Subscriber(right_image_topic, Image, callback=self._left_image_callback, queue_size=1)
        self.left_image = None
        self.left_image_ts = None
        self.right_image = None
        self.right_image_ts = None
        self.bridge = CvBridge()
        self.input_size = (360, 640)
        model = rospy.get_param('object_keypoints_ros/load_model', "/home/ken/Hack/catkin_ws/src/object_keypoints/model/modelv2.pt")
        self.pipeline = pipeline.PnPKeypointPipeline(model, self._read_keypoints(), False)
        self.rgb_mean = torch.tensor([0.5, 0.5, 0.5], requires_grad=False, dtype=torch.float32)[:, None, None]
        self.rgb_std = torch.tensor([0.25, 0.25, 0.25], requires_grad=False, dtype=torch.float32)[:, None, None]
        self._read_calibration()
        self.prediction_size = (90, 160)
        scaling_factor = np.array(self.image_size) / np.array(self.prediction_size)
        self.pipeline.reset(self.K, self.Kp, self.D, self.Dp, self.T_RL, scaling_factor)

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)

        # Publishers
        self.center_point_publisher = rospy.Publisher("object_keypoints_ros/center", PointStamped, queue_size=1)
        self.point0_pub = rospy.Publisher("object_keypoints/0", PointStamped, queue_size=1)
        self.point1_pub = rospy.Publisher("object_keypoints/1", PointStamped, queue_size=1)
        self.point2_pub = rospy.Publisher("object_keypoints/2", PointStamped, queue_size=1)
        self.point3_pub = rospy.Publisher("object_keypoints/3", PointStamped, queue_size=1)
        self.left_heatmap_pub = rospy.Publisher("object_keypoints/heatmap_left", Image, queue_size=1)
        self.right_heatmap_pub = rospy.Publisher("object_keypoints/heatmap_right", Image, queue_size=1)

    def _read_calibration(self):
        path = rospy.get_param('object_keypoints_ros/calibration')
        params = camera_utils.load_calibration_params(path)
        self.K = params['K']
        self.Kp = params['Kp']
        self.D = params['D']
        self.Dp = params['Dp']
        self.T_RL = params['T_RL']
        self.image_size = params['image_size']

    def _read_keypoints(self):
        path = rospy.get_param('object_keypoints_ros/keypoints')
        with open(path, 'rt') as f:
            return np.array(json.loads(f.read())['3d_points'])

    def _right_image_callback(self, image):
        img = self.bridge.imgmsg_to_cv2(image, 'rgb8')
        self.right_image = img
        self.right_image_ts = image.header.stamp

    def _left_image_callback(self, image):
        img = self.bridge.imgmsg_to_cv2(image, 'rgb8')
        self.left_image = img
        self.left_image_ts = image.header.stamp

    def _preprocess_image(self, image):
        image = image.transpose([2, 0, 1])
        image = torch.tensor(image / 255.0, dtype=torch.float32)
        image -= self.rgb_mean
        image /= self.rgb_std
        image = image[None]
        return torch.nn.functional.interpolate(image, size=self.input_size, mode='bilinear', align_corners=False).detach()

    def _publish_keypoints(self, keypoints, time):
        for i in range(min(keypoints.shape[0], 4)):
            msg = _to_msg(keypoints[i], rospy.Time(0), self.left_camera_frame)
            getattr(self, f'point{i}_pub').publish(msg)

    def _publish_heatmaps(self, left, right):
        left = ((left + 1.0) * 0.5).sum(axis=0)
        right = ((right + 1.0) * 0.5).sum(axis=0)
        left = np.clip(cm.inferno(left) * 255.0, 0, 255.0).astype(np.uint8)
        right = np.clip(cm.inferno(right) * 255.0, 0, 255.0).astype(np.uint8)
        left_msg = self.bridge.cv2_to_imgmsg(left[:, :, :3], encoding='passthrough')
        right_msg = self.bridge.cv2_to_imgmsg(right[:, :, :3], encoding='passthrough')
        self.left_heatmap_pub.publish(left_msg)
        self.right_heatmap_pub.publish(right_msg)

    def step(self):
        I = torch.eye(4)[None]
        if self.left_image is not None and self.right_image is not None:
            left_image = self._preprocess_image(self.left_image)
            right_image = self._preprocess_image(self.right_image)
            out = self.pipeline(left_image, right_image)
            self.left_image = None
            self.right_image = None
            self._publish_keypoints(out['keypoints'][0], self.left_image_ts)
            self._publish_heatmaps(out['heatmap_left'][0], out['heatmap_right'][0])

if __name__ == "__main__":
    with torch.no_grad():
        rospy.init_node("object_keypoints_ros")
        keypoint_pipeline = ObjectKeypointPipeline()
        rate = rospy.Rate(10)
        with torch.no_grad():
            while not rospy.is_shutdown():
                keypoint_pipeline.step()
                rate.sleep()

