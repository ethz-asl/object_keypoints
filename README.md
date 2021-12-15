
# Object Keypoint Tracking ROS Wrapper

Performs stereo keypoint tracking using the [Object Keypoints](https://github.com/ethz-asl/object_keypoints) package.

Dependencies:
- [Object Keypoints](https://github.com/ethz-asl/object_keypoints)
    -  Use the `monocular` branch 

The example launch file at `launch/keypoints.launch` shows an example configuration.

The arguments to the `run_pipeline.py` node are:
- `left_image_topic` the topic for the left camera image stream.
- `left_image_topic` the topic for the right camera image stream.
- `left_camera_frame` the tf frame for the left camera optical frame.
- `right_camera_frame` the tf frame for the right camera optical frame.
- `load_model` path to the trained model to use for inference.
- `calibration` path to a [Kalibr](https://github.com/ethz-asl/kalibr) yaml calibration file containing camera intrinsics and extrinsics.

An example could be: `roslaunch object_keypoints_ros keypoints.launch load_vistools:=<whether-want-vis-result>, load_cam:=<whether-want-launch-cam>, load_bag:=<whether-want-to-play-rosbag> `