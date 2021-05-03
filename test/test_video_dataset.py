import unittest
from unittest import mock
import numpy as np
from perception.datasets.video import StereoVideoDataset, _compute_kernel

class VideoDatasetTest(unittest.TestCase):
    def test_add_kernel(self):
        kernel = _compute_kernel(50, 25)
        target = np.zeros((120, 160), dtype=np.float32)
        StereoVideoDataset.kernel = kernel
        StereoVideoDataset.kernel_center = 25
        StereoVideoDataset.kernel_size = 50
        StereoVideoDataset._add_kernel(target, np.array([[80., 60.]]))
        self.assertEqual(target.max(), kernel[25, 25])
        self.assertEqual(target[60, 80], target.max())

        target = np.zeros((120, 160), dtype=np.float32)
        StereoVideoDataset._add_kernel(target, np.array([[1., 1.]]))
        self.assertEqual(target.max(), kernel[25, 25])
        self.assertEqual(target[1, 1], target.max())
        self.assertGreater(target.max(), 1e-3)

        # Past the end along x-axis.
        target = np.zeros((120, 160), dtype=np.float32)
        StereoVideoDataset._add_kernel(target, np.array([[165., 60.]]))
        self.assertNotEqual(target.max(), kernel[25, 25])
        self.assertEqual(target[60, 159], target.max())

        # Past end along both axes
        target = np.zeros((120, 160), dtype=np.float32)
        StereoVideoDataset._add_kernel(target, np.array([[165., 130.]]))
        self.assertEqual(target[119, 159], target.max())

        # Before beginning.
        target = np.zeros((120, 160), dtype=np.float32)
        StereoVideoDataset._add_kernel(target, np.array([[-10., -130.]]))
        self.assertEqual(target[0, 1], target.max())

        target = np.zeros((720, 1280), dtype=np.float32)
        StereoVideoDataset.kernel_size = 50
        StereoVideoDataset.kernel_center = 25
        StereoVideoDataset.width = 1280
        StereoVideoDataset.height = 720
        StereoVideoDataset._add_kernel(target, np.array([[456.02, 34.744]]))
        self.assertGreater(target.max(), 1e-3)

        target = np.zeros((360, 640), dtype=np.float32)
        StereoVideoDataset._add_kernel(target, np.array([[353.5, 153.8]]))
        self.assertEqual(target[154, 354], target.max())



if __name__ == "__main__":
    unittest.main()
