#!/usr/bin/env python3

# To find catkin python3 build of tf2_py
from pickle import NONE
import sys

from numpy.core.fromnumeric import size
# sys.path.insert(0, '/home/user/catkin_ws/devel/lib/python3/dist-packages')
# sys.path.insert(0, '/home/usr/thesis/perception/kp_new')
import json

import rospy
import message_filters
import torch
import numpy as np
import cv2
import tf2_ros
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import PointStamped, PoseStamped, PoseArray, Pose
from perception.utils import ros as ros_utils
from sensor_msgs.msg import Image
from perception.datasets.video import SceneDataset
from perception import pipeline
from perception.utils import camera_utils
from matplotlib import cm
from vision_msgs.msg import BoundingBox3D
#from . import utils
import utils
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

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
        self.left_camera_frame = rospy.get_param('object_keypoints_ros/left_camera_frame')
        self.left_sub = rospy.Subscriber(left_image_topic, Image, callback=self._left_image_callback, queue_size=1)
        self.left_image = None
        self.left_image_ts = None
        self.right_image = None
        self.right_image_ts = None
        self.bridge = CvBridge()
        self.use_gpu = rospy.get_param("object_keypoints_ros/gpu", True)
        
        # params
        self.input_size = (511, 511)  # TODO: hard-coded
        self.IMAGE_SIZE = (1280, 720)
        
        self.prediction_size = SceneDataset.prediction_size # 64 x 64
        self.image_offset = SceneDataset.image_offset
        
        # 2D objects
        self.points_left = []
            
        self.keypoint_config_path = rospy.get_param('object_keypoints_ros/keypoints')
        with open(self.keypoint_config_path, 'rt') as f:
            self.keypoint_config = json.load(f)

        model = rospy.get_param('object_keypoints_ros/load_model', "/home/ken/Hack/catkin_ws/src/object_keypoints/model/modelv2.pt")
       
        if rospy.get_param('object_keypoints_ros/pnp', False):
            self.pipeline = pipeline.PnPKeypointPipeline(model, self._read_keypoints(), torch.cuda.is_available())
        else:
            _3d_point = []
            self.pipeline = pipeline.LearnedKeypointTrackingPipeline(model, self.use_gpu,
                self.prediction_size, _3d_point, self.keypoint_config)

        self.rgb_mean = torch.tensor([0.5, 0.5, 0.5], requires_grad=False, dtype=torch.float32)[:, None, None]
        self.rgb_std = torch.tensor([0.25, 0.25, 0.25], requires_grad=False, dtype=torch.float32)[:, None, None]
        
        self._read_calibration()
        self.scaling_factor = np.array(self.IMAGE_SIZE) / np.array(self.prediction_size)
        rospy.loginfo("scalling facotr: ")
        rospy.loginfo(self.scaling_factor)
        
        # Monocular Version
        self.pipeline.reset(self.camera_small)

        self._compute_bbox_dimensions()

        
        # kp marker
        self.kpArray = MarkerArray()
        self.kp_publisher = rospy.Publisher('kp_world_pos',MarkerArray, queue_size=10)
        # kp points s
        self.kp_poses = PoseArray()
        self.kp_poses.header.frame_id = self.left_camera_frame
        self.kp_pose_publisher = rospy.Publisher('kp_3d_poses', PoseArray, queue_size=10)
        
        # Publishers
        self.center_point_publisher = rospy.Publisher("object_keypoints_ros/center", PointStamped, queue_size=1)
        self.left_heatmap_pub = rospy.Publisher("object_keypoints_ros/heatmap_left", Image, queue_size=1)
        self.right_heatmap_pub = rospy.Publisher("object_keypoints_ros/heatmap_right", Image, queue_size=1)
        self.pose_pub = rospy.Publisher("object_keypoints_ros/pose", PoseStamped, queue_size=1)
        self.result_img_pub = rospy.Publisher("object_keypoints_ros/result_img", Image, queue_size=1)
        # Only used if an object mesh is set.
        if self.bbox_size is not None:
            self.bbox_pub = rospy.Publisher("object_keypoints_ros/bbox", BoundingBox3D, queue_size=1)
        else:
            self.bbox_pub = None

    def _read_calibration(self):
        path = rospy.get_param('object_keypoints_ros/calibration')
        params = camera_utils.load_calibration_params(path)
        self.K = params['K']
        self.Kp = params['Kp']
        self.D = params['D']
        self.Dp = params['Dp']
        self.T_RL = params['T_RL']
        self.image_size = params['image_size']
      
        left_camera = camera_utils.FisheyeCamera(params['K'], params['D'], params['image_size'])
        self.left_camera = left_camera

        # Cooperate with monocular version
        camera = camera_utils.FisheyeCamera(params['K'], params['D'], params['image_size'])
        camera = camera.scale(SceneDataset.height_resized / SceneDataset.height)
        self.camera = camera.cut(self.image_offset)

        scale_small = self.prediction_size[0] / SceneDataset.height_resized
        self.camera_small = camera.cut(self.image_offset).scale(scale_small)

    def _read_keypoints(self):  # not included in prediction process
        path = rospy.get_param('object_keypoints_ros/keypoints')
        with open(path, 'rt') as f:
            return np.array(json.loads(f.read())['3d_points'])

    def _compute_bbox_dimensions(self):
        mesh_file = rospy.get_param('object_keypoints_ros/object_mesh', None)
        if mesh_file is not None:
            bounding_box = utils.compute_bounding_box(mesh_file)
            # Size is in both directions, surrounding the object from the object center.
            self.bbox_size = (bounding_box.max(axis=0) - bounding_box.min(axis=0)) * 0.5
        else:
            self.bbox_size = None

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

    def _publish_pose(self, pose_msg, time):
        pose_msg = ros_utils.transform_to_pose(T, self.left_camera_frame, rospy.Time(0))
        self.pose_pub.publish(pose_msg)
        self._publish_bounding_box(pose_msg)

    def _publish_heatmaps(self, left, right, left_keypoints, right_keypoints):
        left = ((left + 1.0) * 0.5).sum(axis=0)
        left = np.clip(cm.inferno(left) * 255.0, 0, 255.0).astype(np.uint8)

        for kp in left_keypoints:
            kp = kp.round().astype(int)
            left = cv2.circle(left, (kp[0], kp[1]), radius=2, color=(0, 255, 0), thickness=-1)
        left_msg = self.bridge.cv2_to_imgmsg(left[:, :, :3], encoding='passthrough')
        self.left_heatmap_pub.publish(left_msg)

    def _publish_bounding_box(self, T, pose_msg):
        if self.bbox_size is not None:
            msg = BoundingBox3D()
            msg.pose = pose_msg.pose
            msg.size.x = self.bbox_size[0]
            msg.size.y = self.bbox_size[1]
            msg.size.z = self.bbox_size[2]
            self.bbox_pub.publish(msg)

    def _publish_result(self,image):
        image_msg = self.bridge.cv2_to_imgmsg(image[:, :, :3], encoding='passthrough')
        image_msg.header.stamp = self.left_image_ts
        self.result_img_pub.publish(image_msg)
        #self.kp_publisher.publish(self.kpArray)

    def _publish_kp_3d_poses(self):
        self.kp_poses.header.stamp = self.left_image_ts
        self.kp_pose_publisher.publish(self.kp_poses)

    def _to_heatmap(self, target):
        target = np.clip(target, 0.0, 1.0)
        target = target.sum(axis=0)
        target = (cm.inferno(target) * 255.0).astype(np.uint8)[:, :, :3]
        return cv2.resize(target[:, :, :3], self.IMAGE_SIZE)

    def _to_image(self, frame):
        frame = SceneDataset.to_image(frame)
        return cv2.resize(frame, self.IMAGE_SIZE)
    
    def kp_map_to_image(self, kp):
        kp_ = []
        for i, pt in enumerate(kp):
            if pt.size == 0:
                kp_.append(np.array([]))
                continue
            pt = np.squeeze(pt)     
            pt = np.multiply( pt, self.scaling_factor).astype(np.int64)
            kp_.append( np.flip(pt) )   # TODO: notice kp idx is flipped from image idx in 2D 
        return kp_
    
    def step(self):
        
        I = torch.eye(4)[None]
        
        if self.left_image is not None:
            left_image = self._preprocess_image(self.left_image)
            objects, heatmap = self.pipeline(left_image)
            self.left_image = None
 
            heatmap_left = self._to_heatmap(heatmap[0].numpy())
            
            left_image = left_image.cpu().numpy()[0]
            
            #rospy.loginfo("image size: ")              # [3, 511, 511]
            #rospy.loginfo(np.shape(left_image))
            
            left_rgb = self._to_image(left_image)
            image_left = (0.3 * left_rgb + 0.7 * heatmap_left).astype(np.uint8)
            
            # 2D objects
            self.points_left = []
            self.kp_poses.poses.clear()
            self.kpArray.markers.clear()
            
            for obj in objects:
                #p_left = np.concatenate([p + 1.0 for p in obj['keypoints'] if p.size != 0], axis=0)
                kp_num = len(obj['p_C'])
                
                if kp_num == 0:
                    return 
                
                for i, p_l in enumerate(obj['p_C']):
                    
                    if p_l is None:  # TODO (boyang): use the modified msg to replace this
        
                        self.kp_pose = Pose()
                        self.kp_pose.orientation.w = 1.0
                        self.kp_pose.position.z = -100
                        self.kp_poses.poses.append(self.kp_pose)
                        continue

                    self.kp_pose = Pose()
                    self.kp_pose.orientation.w = 1.0
                    self.kp_pose.orientation.x = 0.0
                    self.kp_pose.orientation.y = 0.0
                    self.kp_pose.orientation.z = 0.0
                    self.kp_pose.position.x = p_l[0][0]
                    self.kp_pose.position.y = p_l[0][1]
                    self.kp_pose.position.z = p_l[0][2]
                    self.kp_poses.poses.append(self.kp_pose)
                
                rospy.loginfo("2D kp: ")
                rospy.loginfo(obj['keypoints'])  # indx in 64,64 image

                # extract 2D keypoints
                self.points_left = self.kp_map_to_image(obj['keypoints'])
                
                rospy.loginfo("after scalling: ")
                rospy.loginfo(self.points_left)  # indx in 64,64 image

                for i, pt in enumerate(self.points_left):
                    if pt.size != 0:
                        green = np.zeros((20,20,3))
                        green[:,:,1] = 100
                        image_left[self.points_left[i][0]-10:self.points_left[i][0]+10, self.points_left[i][1]-10:self.points_left[i][1]+10,:] += green.astype(np.uint8)
                
            # pub result image
            self._publish_result(image_left)
            # pub 3D keypoints
            self._publish_kp_3d_poses()
            
                     
if __name__ == "__main__":
    with torch.no_grad():
        rospy.init_node("object_keypoints_ros")
        keypoint_pipeline = ObjectKeypointPipeline()
        rate = rospy.Rate(10)
        with torch.no_grad():
            while not rospy.is_shutdown():
                keypoint_pipeline.step()
                rate.sleep()

