import os
import rospy
import shutil
import rosbag
import yaml
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
    parser.add_argument('--topics', nargs="+", required=True, help="Which topics to encode into the stream.")
    parser.add_argument('--frames', nargs='+', required=True, help="The coordinate frames corresponding to the optical frames of each camera topic given to the --topics argument.")
    parser.add_argument('--calibration-topic', default="/camera_info", help="The camera_info topic containing the camera calibration.")
    parser.add_argument('--base-frame', default='panda_link0', help="The name of the base tf frame.")
    return parser.parse_args()

bridge = CvBridge()

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

    def _read_calibration(self, bag):
        if (self.flags.calibration_topic):
            print("No calibration topic provided, ignoring")
            return

        print("Reading calibration")
        for topic, camera_info, t in bag.read_messages(topics=self.flags.calibration_topic):
            return {
                'cam0': {
                    'cam_overlaps': [1],
                    'camera_model': 'pinhole' if camera_info.distortion_model == 'plumb_bob' else 'unknown',
                    'distortion_coeffs': camera_info.D,
                    'distortion_model': 'radtan' if camera_info.distortion_model == 'plumb_bob' else camera_info.distortion_model,
                    'intrinsics': [camera_info.K[0], camera_info.K[4], camera_info.K[2], camera_info.K[5]],
                    'resolution': [camera_info.width, camera_info.height]
                }
            }

    def _read_poses(self, bag):
        print("Reading poses")
        tf_tree = tf2.BufferCore(rospy.Duration(360000.0))
        for topic, message, t in bag.read_messages(topics=["/tf", "/tf_static"]):
            for tf_message in message.transforms:
                if topic == '/tf_static':
                    tf_tree.set_transform_static(tf_message, f"bag/{topic}")
                else:
                    tf_tree.set_transform(tf_message, f'bag/{topic}')

        return tf_tree

    def _gather_images(self, bag):
        image_messages = []
        for topic in self.flags.topics:
            for _, message, t in bag.read_messages(topics=topic):
                i = len(image_messages)
                print("image {:05} time: {}".format(i, t), end="\r")
                image_messages.append({
                    'message': message,
                    'i': i,
                    't': t.to_sec()
                })

        return image_messages

    def _gather_poses(self, tf_tree, image_messages):
        print("Looking up poses")
        pose_data = []
        i = 0
        for frame in self.flags.frames:
            for item in image_messages:
                try:
                    # Reminder: ^{B}T^{A} = T_BA = lookup_transform(source_frame=A, target_frame=B)
                    T_BC = ros_utils.message_to_transform(tf_tree.lookup_transform_core(target_frame=self.flags.base_frame,
                            source_frame=frame, time=item['message'].header.stamp))
                    item['camera_pose'] = T_BC
                    item['i'] = i # Override index as some frames might have been skipped.
                    pose_data.append(item)
                    i += 1
                except tf2.ExtrapolationException:
                    print("Extrapolation exception. Skipping entry {}.".format(i))

        return pose_data

    def _create_out_folder(self, bag_name):
        out_folder = os.path.join(self.flags.out, bag_name.split(os.path.extsep)[0])
        os.makedirs(out_folder, exist_ok=True)
        return out_folder

    def _write_poses(self, out_file, poses):
        transforms = out_file.create_dataset('camera_transform', (len(poses), 4, 4), dtype=np.float64)
        for i, pose in enumerate(poses):
            transforms[i] = pose['camera_pose']

    def _encode_video(self, bag_name, frame_data):
        out_folder = os.path.join(self.flags.out, bag_name.split(os.path.extsep)[0])

        out_file = os.path.join(out_folder, 'frames.mp4')
        preview = os.path.join(out_folder, 'frames_preview.mp4')
        print("Encoding video {}".format(bag_name))

        _encode_full_video(frame_data, out_file)
        _encode_preview(out_file, preview)

    def main(self):
        for path in self._bags[self.flags.skip:self.flags.until]:
            with rosbag.Bag(path, 'r') as bag:
                bag_name = os.path.basename(path)
                out_folder = self._create_out_folder(bag_name)

                calibration = self._read_calibration(bag)
                if calibration:
                    calibration_filename = os.path.join(out_folder, 'calibration.yaml')
                    with open(calibration_filename, 'w') as calibration_file:
                        yaml.dump(calibration, calibration_file, default_flow_style=False)

                h5_filename = os.path.join(out_folder, 'data.hdf5')
                with h5py.File(h5_filename, 'w') as h5_file:
                    tf_tree = self._read_poses(bag)
                    image_frames = self._gather_images(bag)
                    poses = self._gather_poses(tf_tree, image_frames)
                    self._write_poses(h5_file, poses)
                    self._encode_video(bag_name, poses)

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

