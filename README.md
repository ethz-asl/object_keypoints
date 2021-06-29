# Object Keypoint Tracking

This repository contains a toolkit for collecting, labeling and tracking object keypoints. Object keypoints are semantic points in an object's coordinate frame.

The project allows collecting images from multiple viewpoints using a robot with a wrist mounted camera. These image sequences can then be labeled using a labeling tool StereoLabel.

![StereoLabel image labeling](assets/images/stereo_label.jpg)

Once the images are labeled, a model can be learned to detect keypoints in the images and compute 3D keypoints in the camera's coordinate frame using our tracking pipeline.

## Installation

External Dependencies:
- [HUD](https://github.com/ethz-asl/hud)
- ROS melodic/noetic

Install HUD. Then install dependencies with `pip install -r requirements.txt` and the package using `pip3 install -e .`.

## Usage

Here we describe the process we used to arrive at our labeled datasets and learned models.

### Calibration and setup

First, calibrate your camera and obtain a hand-eye-calibration. Calibrating the camera can be done using [Kalibr](https://github.com/ethz-asl/kalibr). Hand-eye-calibration can be done using the [ethz-asl/hand_eye_calibration](https://github.com/ethz-asl/hand_eye_calibration) package.

The software currently assumes that the Kalibr `equidistant` calibration model was used for calibrating the camera.

Kalibr will spit out a yaml file like the one at `config/calibration.yaml`.

Once you have obtained the hand-eye calibration, configure your robot description so that the tf tree includes both the left and right camera optical frames.

### Collecting data

The script `scripts/collect_bags.py` is a helper program to assist in collecting data. It will use [rosbag](http://wiki.ros.org/rosbag) to record the camera topics and and transform messages.

Run it with `python3 scripts/collect_bags.py --out <path-to-bag-output-folder>`.

Press enter to start recording a new sequence. Recording will start after a 5 second grace period, after which the topics will be recorded for 30 seconds. During the 30 seconds, slowly guide the robot arm to different viewpoints observing your target objects.

### Encoding data

Since rosbag is not a very convenient or efficient format, we encode the data into a format that is better to work with and uses up less disk space. This is done using the script `scripts/encode_bag.py`.

Run it with `python3 scripts/encode_bags.py --bags <path-to-bag-output-folder> --out <path-to-dataset-output> --calibration <path-to-kalibr-calibration.yaml>`.

### Labeling data

![Valve](assets/images/valve.jpg)

First decide how many keypoints you will use for your object class and what their configuration is. Write a keypoint configuration file, like `config/valve.json` and `config/cups.json`. For example, in the case of our valve above, we define four different keypoints, which are of two types. The first type is the center keypoint type and the second is the spoke keypoint type. For our valve, there are three spokes, so we write our keypoint configuration as:
```
{ "keypoint_config": [1, 3] }
```
What this means, is that the is first going to be one keypoint of the first type and then three keypoint of the next type.

StereoLabel can be launched with `python3 scripts/label.py <path-to-dataset-folder>`. To label keypoints, click on the keypoints in the same order in each image. Make sure to label the points consistent with the keypoint configuration that you defined, so that the keypoints end up on the right heatmaps downstream.

If you have multiple objects in the scene, it is important that you annotate one object at the time, sticking to the keypoint order, as the tool makes the assumption that one object's keypoints follow each other. The amount of keypoints you label should equal the amount of objects times the total number of keypoints.

Once you have labeled an equal number of points on the left and right image, points will be backprojected, so that you can make sure that everything is correctly configured and that you didn't make a mistake. The points are saved at the same time to a file `keypoints.json` in each scene's directory.

Here are some keyboard actions the tool supports:
- `a` change the left frame with a random frame from the current sequence.
- `b` change the right frame with a random frame from the current sequence.
- `<tab>` go to next sequence.

Once the points have been saved and backprojected, you can freely press `a` and `b` to swap out the frames to different ones in the sequence. It will project the 3D points back into 2D onto the new frames. You can check that the keypoints project nicely to each frame. If not, you likely misclicked, the viewpoints are too close to each other, there could be an issue with your camera or hand-eye calibration or the camera poses are not accurate for some other reason.


### Checking the data

Once all your sequences have been labeled, you can check that the labels make sense using `python scripts/show_keypoints.py <path-to-dataset-folder>`, which will play the images one by one and show the backprojected points.


### Learning a model

First split your dataset into a training and validation set. You can train a model with `python --train <path-to-training-dataset> --val <path-to-validation-dataset>`.

Once done, you can package a model with `python scripts/package_model.py --model lightning_logs/version_x/checkpoints/<checkpoint>.ckpt --out model.pt`

You can then run and check the metrics on a test set using `python scripts/eval_model.py <path-to-dataset> --model model.pt --keypoints <keypoint-config>`.

### General tips

Here are some general tips that might be of use:
- Collect data at something like 4-5 fps. Generally, frames that are super close to each other aren't that useful and you don't really need every single frame.
- Increase the publishing rate of your `robot_state_publisher` node to something like 100 or 200.
- Move your robot slowly when collecting the data such that the time synchronization between your camera and robot is not that big of a problem.
- Keep the scenes reasonable.
- Collect data in all the operating conditions in which you will want to be detecting keypoints at.


## Using your own sensor and robot

Currently, the package assumes that the data was collected using a stereo camera, specifically a StereoLabs ZED Mini and makes some assumptions about the names of the coordinate frames. When running on with a different setup, check the following things:
- Update the `collect_bags.py` script to record the appropriate topics.
- Update the `encode_bag.py` script to use the correct coordinate frame names.
- Use the `--base-frame` parameter of the `encode_bag.py` script to set the name of the base frame. This should be a coordinate frame which is static relative to your objects when recording data.

Moving the platform specific variables into a configuration file, would be a nice addition.

The current implementation requires a stereo camera, but changing this to make use of a monocular camera would be possible. At least the following changes would have to be made:
- Update `encode_bag.py` to encode only one camera stream. In the stereo case, frames could be added after each other.
- Update `label.py` to make use of the single sequence of images.
- Update the dataloader to use the new data format.
- Update the evaluation script and visualizer to only use one stream of images.

