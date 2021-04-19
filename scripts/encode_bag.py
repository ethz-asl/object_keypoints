import os
import copy
import rospy
import shutil
import rosbag
import subprocess
import numpy as np
import tf2_py as tf2
import h5py
import skvideo.io
from time import time
from argparse import ArgumentParser
from PIL import Image
from cv_bridge import CvBridge
from geometry_msgs import msg
from perception.utils import ros as ros_utils

def read_args():
    parser = ArgumentParser()
    parser.add_argument('--bags', required=True, help="Path to directory containing rosbags.")
    parser.add_argument('--out', '-o', required=True, help="Where to write output files.")
    parser.add_argument('--skip', default=0, type=int, help="Skip the first n bags.")
    parser.add_argument('--until', default=None, type=int, help="Encode until the nth bag.")
    return parser.parse_args()

def left_to_right_transform():
    message = msg.TransformStamped()
    message.transform.translation.x = 0.06242168917
    message.transform.translation.y = 0.000234
    message.transform.translation.z = 5.759928320e-05
    message.transform.rotation.x = 0.0
    message.transform.rotation.y = 0.0
    message.transform.rotation.z = 0.0
    message.transform.rotation.w = 1.0
    message.header.seq = 0
    message.header.frame_id = "zedm_left_camera_optical_frame"
    message.child_frame_id = "zedm_right_camera_optical_frame"
    return message

left_right = left_to_right_transform()

bridge = CvBridge()

def _add_poses(hdf, group_name, poses):
    group = hdf.create_group(group_name)
    transforms = group.create_dataset('camera_transform', (len(poses), 4, 4), dtype=np.float64)
    for i, pose in enumerate(poses):
        transforms[i] = pose['camera_pose']

def _write_images(folder, data):
    for item in data:
        image = bridge.imgmsg_to_cv2(item['message'], desired_encoding='rgb8')
        image = Image.fromarray(image)
        image.save('/tmp/encode_bags_tmp/{}/{:05}.png'.format(folder, item['i']))
        print('Writing /tmp/encode_bags_tmp/{}/{:05}.png'.format(folder, item['i']), end='\r')
    print("")

def _encode_full_video(data, filepath):
    writer = skvideo.io.FFmpegWriter(filepath, outputdict={
        '-vcodec': 'libx264',
        '-crf': '0',
        '-preset': 'fast',
        '-framerate': '30'
    })
    try:
        for i, item in enumerate(data):
            print(f"Encoding frame {i:06}", end='\r')
            frame = bridge.imgmsg_to_cv2(item['message'], desired_encoding='rgb8')
            writer.writeFrame(frame)
    finally:
        writer.close()

def _encode_preview(video_file, preview_file):
    subprocess.run(['ffmpeg', '-i', video_file, '-c:a', 'copy',
        '-framerate', '30', '-c:v', 'libx264', '-crf', '24', '-vf', 'scale=1280:-1',
        '-preset', 'fast', '-y', preview_file])

class Runner:
    def __init__(self):
        self.flags = read_args()
        self._find_bags()

    def _find_bags(self):
        filenames = os.listdir(self.flags.bags)
        self._bags = []
        for filename in filenames:
            path = os.path.join(self.flags.bags, filename)
            if '.bag' in path:
                self._bags.append(path)
        self._bags.sort()

    def _read_poses(self, out_folder, bag):
        print("Reading poses")
        tf_tree = tf2.BufferCore(rospy.Duration(360000.0))
        for topic, message, t in bag.read_messages(topics=["/tf", "/tf_static"]):
            for tf_message in message.transforms:
                if tf_message.header.frame_id == 'zedm_left_optical_frame' and tf_message.child_frame_id == 'zedm_right_optical_frame':
                    tf_message.transform = left_right.transform

                if topic == '/tf_static':
                    tf_tree.set_transform_static(tf_message, f"bag/{topic}")
                else:
                    tf_tree.set_transform(tf_message, f'bag/{topic}')

        return tf_tree

    def _gather_images(self, bag):
        left_messages = []
        for topic, message, t in bag.read_messages(topics="/zedm/zed_node/left_raw/image_raw_color"):
            i = len(left_messages)
            print("left image {:05} time: {}".format(i, t), end="\r")
            left_messages.append({
                'message': message,
                'i': i,
                't': t.to_sec()
            })
        right_messages = []
        for topic, message, t in bag.read_messages(topics="/zedm/zed_node/right_raw/image_raw_color"):
            i = len(right_messages)
            print("right image {:05} time: {}".format(i, t), end="\r")
            right_messages.append({'message': message, 'i': i, 't': t.to_sec()})
        print("")
        print(f"left images: {len(left_messages)} right images: {len(right_messages)}")
        return left_messages, right_messages

    def _gather_poses(self, tf_tree, left, right):
        print("Looking up poses")
        left_data = []
        i = 0
        for item in left:
            try:
                # Reminder: ^{B}T^{A} = T_BA = lookup_transform(source_frame=A, target_frame=B)
                T_BL = ros_utils.message_to_transform(tf_tree.lookup_transform_core(target_frame='panda_link0',
                        source_frame='zedm_left_optical_frame', time=item['message'].header.stamp))
                item['camera_pose'] = T_BL
                item['i'] = i # Override index as some frames might have been skipped.
                left_data.append(item)
                i += 1
            except tf2.ExtrapolationException:
                print("Extrapolation exception. Skipping entry {} left.".format(i))

        right_data = []
        i = 0
        for item in right:
            try:
                T_BR = ros_utils.message_to_transform(tf_tree.lookup_transform_core(target_frame='panda_link0',
                        source_frame='zedm_right_optical_frame', time=item['message'].header.stamp))
                item['camera_pose'] = T_BR
                item['i'] = i
                right_data.append(item)
                i += 1
            except tf2.ExtrapolationException:
                print("Extrapolation exception. Skipping entry {} right.".format(i))
        return left_data, right_data

    def _create_out_folder(self, bag_name):
        out_folder = os.path.join(self.flags.out, bag_name.split(os.path.extsep)[0])
        os.makedirs(out_folder, exist_ok=True)
        return out_folder

    def _write_poses(self, out_file, left_poses, right_poses):
        _add_poses(out_file, 'left', left_poses)
        _add_poses(out_file, 'right', right_poses)

    def _write_calibration(self, h5_file, bag):
        left_calibration = None
        right_calibration = None
        for topic, message, t in bag.read_messages(topics="/zedm/zed_node/left_raw/camera_info"):
            left_calibration = message
        for topic, message, t in bag.read_messages(topics="/zedm/zed_node/right_raw/camera_info"):
            right_calibration = message
        if left_calibration is None and right_calibration is None:
            return

        group = h5_file.create_group('calibration')
        left = group.create_group('left')
        right = group.create_group('right')
        left_D = left.create_dataset('D', 5, dtype=np.float64)
        left_K = left.create_dataset('K', 9, dtype=np.float64)
        right_D = right.create_dataset('D', 5, dtype=np.float64)
        right_K = right.create_dataset('K', 9, dtype=np.float64)
        left_D[:] = left_calibration.D
        left_K[:] = left_calibration.K
        right_D[:] = right_calibration.D
        right_K[:] = right_calibration.K

    def _encode_video(self, bag_name, left_data, right_data):
        out_folder = os.path.join(self.flags.out, bag_name.split(os.path.extsep)[0])

        out_file = os.path.join(out_folder, 'left.mp4')
        preview = os.path.join(out_folder, 'left_preview.mp4')
        print("Encoding video {} left".format(bag_name))

        _encode_full_video(left_data, out_file)
        _encode_preview(out_file, preview)

        out_file = os.path.join(out_folder, 'right.mp4')
        preview = os.path.join(out_folder, 'right_preview.mp4')
        print("Encoding video {} right".format(bag_name))
        _encode_full_video(right_data, out_file)
        _encode_preview(out_file, preview)

    def main(self):
        for path in self._bags[self.flags.skip:self.flags.until]:
            with rosbag.Bag(path, 'r') as bag:
                bag_name = os.path.basename(path)
                out_folder = self._create_out_folder(bag_name)
                filename = os.path.join(out_folder, 'data.hdf5')

                with h5py.File(filename, 'w') as h5_file:
                    tf_tree = self._read_poses(out_folder, bag)
                    self._write_calibration(h5_file, bag)
                    left_frames, right_frames = self._gather_images(bag)
                    left_poses, right_poses = self._gather_poses(tf_tree, left_frames, right_frames)
                    self._write_poses(h5_file, left_poses, right_poses)
                    self._encode_video(bag_name, left_poses, right_poses)

                print(f"Done with bag {bag_name}.")

    def __enter__(self):
        os.makedirs('/tmp/encode_bags_tmp', exist_ok=True)
        os.makedirs(self.flags.out, exist_ok=True)
        return self

    def __exit__(self, *args):
        shutil.rmtree('/tmp/encode_bags_tmp')

if __name__ == '__main__':
    with Runner() as runner:
        runner.main()

