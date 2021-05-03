import numpy as np
import torch
import cv2
from sklearn import cluster
from perception.utils import camera_utils, clustering_utils, linalg
from scipy.spatial.transform import Rotation
from sklearn import metrics
import skimage
from scipy import ndimage

class InferenceComponent:
    name = "inference"

    def __init__(self, model, cuda):
        self.cuda = cuda
        model = torch.jit.load(model)
        if cuda:
            self.model = model.half().cuda()
        else:
            self.model = model.cpu().float()

    def __call__(self, left_frames, right_frames):
        N = left_frames.shape[0]
        if self.cuda:
            frames = torch.cat([left_frames, right_frames], 0).half()
            frames = frames.cuda()
        else:
            frames = torch.cat([left_frames, right_frames], 0)
        predictions = self.model(frames).cpu().numpy()

        left_pred = predictions[:N]
        right_pred = predictions[N:]
        return left_pred, right_pred


class KeypointExtractionComponent:
    name = "keypoints"
    PROBABILITY_CUTOFF = 0.5

    def __init__(self, keypoint_config, prediction_size):
        # Add center point.
        self.keypoint_config = [1] + keypoint_config['keypoint_config']
        self.n_keypoints = sum(self.keypoint_config)
        prediction_size = prediction_size
        self.indices = np.zeros((*prediction_size, 2), dtype=np.float32)
        for y in range(prediction_size[0]):
            for x in range(prediction_size[1]):
                self.indices[y, x, 0] = np.float32(x) + 0.5
                self.indices[y, x, 1] = np.float32(y) + 0.5
        self.clustering = clustering_utils.KeypointClustering(bandwidth=2.5)
        self.kernel = np.ones((5, 5)) / 25.0
        self.kernel[2, 2] = 0.0

    def _cluster(self, heatmap):
        indices = self.indices[heatmap > self.PROBABILITY_CUTOFF]
        if indices.shape[0] == 0:
            return []
        out = [x for x in self.clustering(indices)]
        return out

    def _extract_keypoints(self, heatmap):
        out_points = []
        assert heatmap.shape[0] == len(self.keypoint_config)
        for i, n_keypoints in enumerate(self.keypoint_config):
            points = self._cluster(heatmap[i])
            out_points.append(points)
        return out_points

    @staticmethod
    def _preprocess_heatmap(heatmap):
        out = heatmap.copy().astype(np.float32)
        out[heatmap > 0.75] = 1.0
        return out * 2.0

    def _points(self, heatmap):
        out_points = []
        heatmap_processed = self._preprocess_heatmap(heatmap)

        for i, n_keypoints in enumerate(self.keypoint_config):
            # convolved = cv2.filter2D(heatmap[i], -1, self.kernel)
            # convolved[convolved < self.PROBABILITY_CUTOFF] = 0.0
            image_max = ndimage.maximum_filter(heatmap_processed[i], size=1, mode='constant')
            points = skimage.feature.peak_local_max(image_max, min_distance=5,
                    threshold_abs=self.PROBABILITY_CUTOFF * 2.0, p_norm=1)
            # from matplotlib import pyplot
            # pyplot.imshow(image_max)
            # pyplot.scatter(points[:, 1], points[:, 0], alpha=0.5, c='red')
            # pyplot.show()
            # print(points.shape)
            # points = points[:, ::-1] + 0.5
            out_points.append(points[:, ::-1] + 1.0)
        return out_points

    def __call__(self, left, right):
        N = left.shape[0]

        keypoints_left = []
        keypoints_right = []
        for i in range(left.shape[0]):
            keypoints_left.append(self._points(left[i]))
            keypoints_right.append(self._points(right[i]))

        return keypoints_left, keypoints_right

def _get_depth(xy, depth):
    indices = np.argwhere(depth)
    try:
        yx = xy[:, ::-1]
    except IndexError:
        import ipdb; ipdb.set_trace()
    distances = np.linalg.norm(indices - yx[:, None], axis=2)
    depth_index = indices[distances.argmin(axis=1)]
    return depth[depth_index[:, 0], depth_index[:, 1]]

def project_3d(points, depth, K):
    xys = points.round().astype(np.int32)
    zs = _get_depth(xys, depth[1:])
    p_W = np.zeros((points.shape[0], 3))
    p_W[:, 0] = (points[:, 0] - K[0, 2]) * zs / K[0, 0]
    p_W[:, 1] = (points[:, 1] - K[1, 2]) * zs / K[1, 1]
    p_W[:, 2] = zs
    return p_W

    # for i, point in enumerate(keypoints):
    #     xy = point.round().astype(np.int32)
    #     z = self._get_depth(xy, left_depth[heatmap])
    #     p_W = np.ones((3,))
    #     p_W[0] = (point[0] - self.K[0, 2]) * z / self.K[0, 0]
    #     p_W[1] = (point[1] - self.K[1, 2]) * z / self.K[1, 1]
    #     p_W[2] = z
    #     R, _ = cv2.Rodrigues(self.T_RL[:3, :3])
    #     x_p, _ = cv2.fisheye.projectPoints(p_W[None, None, :], R, self.T_RL[:3, 3], self.Kp, self.Dp)
    #     right_points[i] = x_p.ravel()

class AssociationComponent:
    def __init__(self):
        self.K = None
        self.D = None
        self.Kp = None
        self.Dp = None
        self.T_RL = None
        self.F = None

    def reset(self, K, Kp, D, Dp, T_RL, scaling_factor):
        self.K = camera_utils.scale_camera_matrix(K, 1.0 / scaling_factor)
        self.Kp = camera_utils.scale_camera_matrix(Kp, 1.0 / scaling_factor)
        self.D = D
        self.Dp = Dp
        self.T_RL = T_RL.astype(np.float32)
        self.Kinv = np.linalg.inv(K)
        self.Kpinv = np.linalg.inv(Kp)

    def __call__(self, left_keypoints, left_depth, right_keypoints, right_depth):
        for heatmap, keypoints in enumerate(left_keypoints):
            if len(right_keypoints[heatmap]) == 0:
                continue
            p_W = project_3d(keypoints, left_depth[heatmap], self.K)
            R, _ = cv2.Rodrigues(self.T_RL[:3, :3])
            x_p, _ = cv2.fisheye.projectPoints(p_W[:, None], R, self.T_RL[:3, 3], self.Kp, self.Dp)
            assert x_p.shape[1] == 1
            right_points = x_p[:, 0, :]
            found_right_points = np.stack(right_keypoints[heatmap])
            distances = metrics.pairwise_distances(found_right_points, right_keypoints[heatmap])
            right_keypoints[heatmap][distances.argmin(axis=1)]

class TriangulationComponent:
    name = "triangulation"

    def __init__(self):
        self.K = None
        self.Kp = None
        self.D = None
        self.Dp = None
        self.T_RL = None

    def reset(self, K, Kp, D, Dp, T_RL, scaling_factor):
        self.K = camera_utils.scale_camera_matrix(K, 1.0 / scaling_factor)
        self.Kp = camera_utils.scale_camera_matrix(Kp, 1.0 / scaling_factor)
        self.D = D
        self.Dp = Dp
        self.T_RL = T_RL

    def __call__(self, left_keypoints, right_keypoints):
        try:
            left_keypoints = left_keypoints[:, None, :].astype(np.float32)
            right_keypoints = right_keypoints[:, None, :].astype(np.float32)
        except IndexError as e:
            import ipdb; ipdb.set_trace()
        left_keypoints = cv2.fisheye.undistortPoints(left_keypoints, self.K, self.D, P=self.K)[:, 0, :]
        right_keypoints = cv2.fisheye.undistortPoints(right_keypoints, self.Kp, self.Dp, P=self.Kp)[:, 0, :]
        P1 = self.K @ np.eye(3, 4, dtype=np.float32)
        P2 = self.Kp @ self.T_RL[:3].astype(np.float32)
        p_LK = cv2.triangulatePoints(
            P1, P2, left_keypoints.T, right_keypoints.T
        ).T  # N x 4

        p_LK = p_LK / p_LK[:, 3:4]
        return p_LK[:, :3]

class ObjectExtraction:
    def __init__(self, keypoint_config):
        self.keypoint_config = keypoint_config
        self.prediction_size = [180, 320]
        self.max = np.array(self.prediction_size[::-1], dtype=np.int32) - 1
        self.min = np.zeros(2, dtype=np.int32)
        self.image_indices = np.zeros((*self.prediction_size, 2))
        for i in range(self.prediction_size[0]):
            for j in range(self.prediction_size[1]):
                self.image_indices[i, j] = (j + 0.5, i + 0.5)

    def __call__(self, keypoints, centers):
        centers = self.image_indices + centers.transpose([1, 2, 0])
        objects = []
        center_points = np.stack(keypoints[0])
        for center in center_points:
            obj = { 'center': center, 'heatmap_points': [], 'p_centers': [] }
            obj['heatmap_points'] = [[] for _ in range((len(keypoints) - 1))]
            objects.append(obj)
        for i, points in enumerate(keypoints[1:]):
            points_of_type = np.stack(points)
            for point in points:
                xy = np.clip(point.round().astype(np.int32), self.min, self.max)
                predicted_center = centers[xy[1], xy[0], :]
                distance_to_centers = np.linalg.norm(center_points - predicted_center, axis=1)
                obj = objects[distance_to_centers.argmin(axis=0)]
                obj['p_centers'].append(predicted_center)
                obj['heatmap_points'][i].append(point)
        for obj in objects:
            for i in range(len(obj['heatmap_points'])):
                if len(obj['heatmap_points'][i]) > 0:
                    obj['heatmap_points'][i] = np.stack(obj['heatmap_points'][i])
                else:
                    # Keypoint was associated with the wrong object.
                    obj['heatmap_points'][i] = np.array([])
        return objects

class PnPComponent:
    def __init__(self, points_3d):
        self.points_3d = points_3d
        self.ones = np.ones((self.points_3d.shape[0], 1))
        self.K = None
        self.D = None
        self.prev_pose = np.eye(4)

    def reset(self, K, D, scaling_factor):
        self.K = camera_utils.scale_camera_matrix(K, 1.0 / scaling_factor)
        self.D = D

    def __call__(self, keypoints_2d):
        """
        keypoints_2d: K x 2
        """
        N = keypoints_2d.shape[0]
        points = np.zeros((N, self.points_3d.shape[0], 3))
        poses = np.zeros((N, 4, 4))
        for i in range(N):
            poses[i], points[i] = self._compute_keypoints(keypoints_2d[i])
        return poses, points

    def _compute_keypoints(self, keypoints_2d):
        if keypoints_2d.shape[0] < 4:
            rospy.logwarn(f"Only found {keypoints_2d.shape[0]} keypoints.")
            T = self.prev_pose
        else:
            center = keypoints_2d[0]
            keypoints_2d[1:] = sorted(keypoints_2d[1:], key=lambda x: x[1])
            first = keypoints_2d[1]
            def sorting_function(v1):
                def inner(point):
                    v2 = point - center
                    return np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
                return inner
            # Sort counter clockwise from direction defined between center and first keypoints.
            keypoints_2d[2:] = sorted(keypoints_2d[2:], key=sorting_function(first - center))
            n_points = min(keypoints_2d.shape[0], self.points_3d.shape[0])

            success, rvec, tvec, inliers = cv2.solvePnPRansac(self.points_3d[:n_points], keypoints_2d[:n_points], self.K, self.D, flags=cv2.SOLVEPNP_EPNP)
            if not success:
                print("failed to solve pnp")
            T = np.eye(4)
            R, _ = cv2.Rodrigues(rvec)
            T[:3, :3] = R
            T[:3, 3] = tvec[:, 0]
            self.prev_pose = T

        points_3d = np.concatenate([self.points_3d, self.ones], axis=1)
        points_3d = (T @ points_3d[:, :, None])[:, :, 0]
        points_3d = (points_3d / points_3d[:, 3:4])[:, :3]
        return T, points_3d

class PoseSolveComponent:
    def __init__(self, points_3d):
        self.points_3d = np.zeros((points_3d.shape[0] + 1, 3))
        self.points_3d[0] = points_3d.mean(axis=0)
        self.points_3d[1:] = points_3d

    def __call__(self, keypoints):
        keypoints = keypoints[:, :, :3]
        T = np.zeros((keypoints.shape[0], 4, 4))
        for i in range(keypoints.shape[0]):
            object_keypoints = keypoints[i]
            p_center = object_keypoints.mean(axis=0)
            gt_center = self.points_3d.mean(axis=0)
            translation = p_center - gt_center
            T[i, :3, 3] = translation
            # Sort object keypoints along z-axis, this should make the detections
            # more consistent across time steps.
            object_keypoints = sorted(object_keypoints, key=lambda v: v[2])
            to_align = (object_keypoints - translation)
            rotation, _ = Rotation.align_vectors(to_align, self.points_3d)
            T[i, :3, :3] = rotation.as_matrix()
        return T

class ObjectKeypointPipeline:
    def __init__(self, prediction_size, points_3d, keypoint_config):
        self.keypoint_extraction = KeypointExtractionComponent(keypoint_config, prediction_size)
        self.object_extraction = ObjectExtraction(keypoint_config)
        self.association = AssociationComponent()
        self.triangulation = TriangulationComponent()
        self.K = self.Kp = self.D = self.Dp = self.T_RL = None

    def reset(self, K, Kp, D, Dp, T_RL, scaling_factor):
        self.K = camera_utils.scale_camera_matrix(K, 1.0 / scaling_factor)
        self.Kp = camera_utils.scale_camera_matrix(Kp, 1.0 / scaling_factor)
        self.D = D
        self.Dp = Dp
        self.T_RL = T_RL
        self.T_LR = np.linalg.inv(T_RL)
        self.association.reset(K, Kp, D, Dp, T_RL, scaling_factor)
        self.triangulation.reset(K, Kp, D, Dp, T_RL, scaling_factor)

    def _recover_from_left(self, depth, left, right):
        print("Recovering right points from left.")
        p_W = project_3d(left, depth, self.K)
        T_CW = self.T_RL
        R, _ = cv2.Rodrigues(T_CW[:3, :3])
        x_p, _ = cv2.fisheye.projectPoints(p_W[:, None], R, T_CW[:3, 3], self.Kp, self.Dp)
        return x_p[:, 0, :]

    def _recover_from_right(self, depth, left, right):
        print("Recovering left points from right.")
        #TODO: some points might be ok. Only recover the missing ones.
        p_W = project_3d(right, depth, self.Kp)
        T_CW = self.T_LR
        R, _ = cv2.Rodrigues(T_CW[:3, :3])
        x, _ = cv2.fisheye.projectPoints(p_W[:, None], R, T_CW[:3, 3], self.K, self.D)
        return x[:, 0, :]

    def __call__(self, heatmap_left, depth_left, centers_left, heatmap_right, depth_right, centers_right):
        assert heatmap_left.shape[0] == 1, "One at the time, please."
        depth_left = depth_left[0].numpy()
        depth_right = depth_right[0].numpy()
        heatmap_left = heatmap_left.numpy()
        heatmap_right = heatmap_right.numpy()
        left_points, right_points = self.keypoint_extraction(heatmap_left, heatmap_right)
        left_points, right_points = left_points[0], right_points[0]
        objects_left = self.object_extraction(left_points, centers_left[0].numpy())
        objects_right = self.object_extraction(right_points, centers_right[0].numpy())
        objects = []
        for object_left, object_right in zip(objects_left, objects_right):
            for i, (left, right) in enumerate(zip(object_left['heatmap_points'], object_right['heatmap_points'])):
                if len(left) == 0 and len(right) == 0:
                    # Both keypoints where not extracted. This is ignored further down the pipeline.
                    continue

                if len(left) > len(right):
                    # Some keypoints were not appropriately detected in both frames.
                    object_right['heatmap_points'][i] = self._recover_from_left(depth_left[1 + i], left, right)
                elif len(right) > len(left):
                    object_left['heatmap_points'][i] = self._recover_from_right(depth_right[1 + i], left, right)

            self.association(object_left['heatmap_points'], depth_left,
                    object_right['heatmap_points'], depth_right)
            world_points = [self.triangulation(object_left['center'][None], object_right['center'][None])]
            for i in range(len(object_left['heatmap_points'])):
                if len(object_left['heatmap_points'][i]) == 0:
                    continue
                p_W = self.triangulation(object_left['heatmap_points'][i], object_right['heatmap_points'][i])
                world_points.append(p_W)
            objects.append({
                'centers_left': object_left['p_centers'],
                'centers_right': object_right['p_centers'],
                'keypoints_left': [object_left['center']] + object_left['heatmap_points'],
                'keypoints_right': [object_right['center']] + object_right['heatmap_points'],
                'p_L': world_points
            })
        return objects


class KeypointPipeline:
    def __init__(self, model, prediction_size, points_3d, keypoint_config, cuda):
        self.inference = InferenceComponent(model, cuda)
        self.keypoint_extraction = KeypointExtractionComponent(points_3d.shape[0]-1, prediction_size)
        self.assocation = AssociationComponent(keypoint_config)
        self.triangulation = TriangulationComponent(points_3d.shape[0])
        self.pose_solve = PoseSolveComponent(points_3d)

    def reset(self, K, Kp, D, Dp, T_RL, scaling_factor):
        self.keypoint_extraction.reset()
        self.association.reset(K, Kp, D, Dp, T_RL, scaling_factor)
        self.triangulation.reset(K, Kp, D, Dp, T_RL, scaling_factor)

    def __call__(self, left, right):
        heatmap_l, heatmap_r = self.inference(left, right)
        keypoints_l, keypoints_r = self.keypoint_extraction(heatmap_l, heatmap_r)
        out_points = self.triangulation(keypoints_l, keypoints_r)
        out = {
            "heatmap_left": heatmap_l,
            "heatmap_right": heatmap_r,
            "keypoints_left": keypoints_l,
            "keypoints_right": keypoints_r,
            "keypoints": out_points,
            "pose": self.pose_solve(out_points)
        }
        return out

class PnPKeypointPipeline:
    def __init__(self, model, prediction_size, keypoints, cuda):
        self.inference = InferenceComponent(model, cuda)
        # Cluster all but center point.
        self.keypoint_extraction = KeypointExtractionComponent(keypoints.shape[0] - 1, prediction_size)
        self.pnp_l = PnPComponent(keypoints)
        self.pnp_r = PnPComponent(keypoints)

    def reset(self, K, Kp, D, Dp, T_RL, scaling_factor):
        self.keypoint_extraction.reset()
        self.pnp_l.reset(K, D, scaling_factor)
        self.pnp_r.reset(Kp, Dp, scaling_factor)
        self.T_RL = T_RL
        self.T_LR = np.linalg.inv(T_RL)

    def __call__(self, left, right):
        heatmap_l, heatmap_r = self.inference(left, right)
        keypoints_l, keypoints_r = self.keypoint_extraction(heatmap_l, heatmap_r)
        T_L_O, p_L = self.pnp_l(keypoints_l)
        T_R_O, p_R = self.pnp_r(keypoints_r)
        out = {
            "heatmap_left": heatmap_l,
            "heatmap_right": heatmap_r,
            "keypoints_left": keypoints_l,
            "keypoints_right": keypoints_r,
            "keypoints": p_L,
            "pose": T_L_O
        }
        return out


