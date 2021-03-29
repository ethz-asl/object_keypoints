import numpy as np
from scipy.spatial.transform import Rotation
from geometry_msgs import msg as geometry_msgs

def message_to_transform(message):
    T = np.eye(4)
    t = message.transform.translation
    r = message.transform.rotation
    R = Rotation.from_quat([r.x, r.y, r.z, r.w])
    T[:3, 3]  = np.array([t.x, t.y, t.z])
    T[:3, :3] = R.as_matrix()
    return T

def transform_to_message(T, parent_frame, child_frame, timestamp):
    msg = geometry_msgs.TransformStamped()
    msg.header.stamp = timestamp
    msg.header.frame_id = parent_frame
    msg.child_frame_id = child_frame
    msg.transform.translation.x = T[0, 3]
    msg.transform.translation.y = T[1, 3]
    msg.transform.translation.z = T[2, 3]
    quat = Rotation.from_matrix(T[:3, :3]).as_quat()
    msg.transform.rotation.x = quat[0]
    msg.transform.rotation.y = quat[1]
    msg.transform.rotation.z = quat[2]
    msg.transform.rotation.w = quat[3]
    return msg

def transform_to_pose(T, frame, timestamp):
    msg = geometry_msgs.PoseStamped()
    msg.header.stamp = timestamp
    msg.header.frame_id = frame
    msg.pose.position.x = T[0, 3]
    msg.pose.position.y = T[1, 3]
    msg.pose.position.z = T[2, 3]
    q = Rotation.from_matrix(T[:3, :3]).as_quat()
    msg.pose.orientation.x = q[0]
    msg.pose.orientation.y = q[1]
    msg.pose.orientation.z = q[2]
    msg.pose.orientation.w = q[3]
    return msg
