import os
import h5py
from torch.utils.data import IterableDataset, DataLoader
from skvideo import io as video_io


class StereoVideoDataset(IterableDataset):
    def __init__(self, base_dir):
        self.base_dir = os.path.expanduser(base_dir)

    def __enter__(self):
        self._init_videos()
        self._init_metadata()
        return self

    def __exit__(self, *args):
        self.hdf.close()

    def _init_videos(self):
        left_video = os.path.join(self.base_dir, "left.mp4")
        right_video = os.path.join(self.base_dir, "right.mp4")
        self.left_video = video_io.vreader(left_video)
        self.right_video = video_io.vreader(right_video)

    def _init_metadata(self):
        filepath = os.path.join(self.base_dir, "data.hdf5")
        self.hdf = h5py.File(filepath)

    def __iter__(self):
        for left_frame, right_frame in zip(self.left_video, self.right_video):
            yield left_frame, right_frame


if __name__ == "__main__":
    with StereoVideoDataset("~/data/valve/000") as dataset:
        for left, right in dataset:
            print(f"left: {left.shape} right: {right.shape}")
