#!/usr/bin/env python3
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

class ObjectKeypointPipeline:
    def __init__(self):
        left_image_topic = rospy.get_param("object_keypoints/left_image_topic", "/zedm/zed_node/left_raw/image_raw_color")
        right_image_topic = rospy.get_param("object_keypoints/right_image_topic", "/zedm/zed_node/right_raw/image_raw_color")
        self.left_camera_frame = rospy.get_param('object_keypoints/left_camera_frame')
        self.right_camera_frame = rospy.get_param('object_keypoints/right_camera_frame')
        self.left_sub = rospy.Subscriber(left_image_topic, Image, callback=self._right_image_callback, queue_size=1)
        self.right_sub = rospy.Subscriber(right_image_topic, Image, callback=self._left_image_callback, queue_size=1)
        self.left_image = None
        self.left_image_ts = None
        self.right_image = None
        self.right_image_ts = None
        self.bridge = CvBridge()
        self.input_size = (360, 640)
        model = rospy.get_param('object_keypoints/load_model', "/home/ken/Hack/catkin_ws/src/object_keypoints/model/modelv2.pt")
        self.pipeline = pipeline.KeypointPipeline(model, 3)
        self.rgb_mean = torch.tensor([0.5, 0.5, 0.5], requires_grad=False, dtype=torch.float32)[:, None, None]
        self.rgb_std = torch.tensor([0.25, 0.25, 0.25], requires_grad=False, dtype=torch.float32)[:, None, None]
        self._read_calibration()
        self.prediction_size = (90, 160)
        scaling_factor = np.array(self.image_size) / np.array(self.prediction_size)
        self.pipeline.reset(self.K, self.Kp, self.T_RL, scaling_factor)

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)

        # Publishers
        self.center_point_publisher = rospy.Publisher("object_keypoints/center", PointStamped)

    def _read_calibration(self):
        path = rospy.get_param('object_keypoints/calibration')
        params = camera_utils.load_calibration_params(path)
        self.K = params['K']
        self.Kp = params['Kp']
        self.T_RL = params['T_RL']
        self.image_size = params['image_size']

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
        msg = PointStamped()
        msg.header.stamp = time
        msg.header.frame_id = 'base_link'
        msg.point.x = keypoints[0, 0]
        msg.point.y = keypoints[0, 1]
        msg.point.z = keypoints[0, 2]
        self.center_point_publisher.publish(msg)

    def step(self):
        I = torch.eye(4)[None]
        if self.left_image is not None and self.right_image is not None:
            left_image = self._preprocess_image(self.left_image)
            right_image = self._preprocess_image(self.right_image)
            try:
                T_LW = self.tf_buffer.lookup_transform(self.left_camera_frame, 'panda_link0', time=self.left_image_ts)
                T_RW = self.tf_buffer.lookup_transform(self.right_camera_frame, 'panda_link0', time=self.right_image_ts)
            except tf2_ros.ExtrapolationException:
                self.left_image = None
                self.right_image = None
                return
            T_LW = ros_utils.message_to_transform(T_LW)
            T_RW = ros_utils.message_to_transform(T_RW)
            out = self.pipeline(left_image.cuda(), T_LW[None], right_image.cuda(), T_RW[None])
            self.left_image = None
            self.right_image = None
            self._publish_keypoints(out['keypoints_world'][0], self.left_image_ts)

if __name__ == "__main__":
    with torch.no_grad():
        rospy.init_node("object_keypoints")
        keypoint_pipeline = ObjectKeypointPipeline()
        rate = rospy.Rate(5)
        with torch.no_grad():
            while True:
                keypoint_pipeline.step()
                rate.sleep()

