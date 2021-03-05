import unittest
from unittest import mock
import numpy as np
from perception.datasets.video import StereoVideoDataset, _compute_kernel

class VideoDatasetTest(unittest.TestCase):
    def test_add_kernel(self):
        kernel = _compute_kernel(50, 25)
        target = np.zeros((120, 160), dtype=np.float32)
        dataset = mock.Mock()
        dataset.kernel_center = 25
        dataset.kernel_size = 50
        dataset.width = 160
        dataset.height = 120
        StereoVideoDataset._add_kernel(dataset, target, kernel, np.array([[80., 60.]]))
        self.assertEqual(target.max(), kernel[25, 25])
        self.assertEqual(target[60, 80], target.max())

        target = np.zeros((120, 160), dtype=np.float32)
        StereoVideoDataset._add_kernel(dataset, target, kernel, np.array([[1., 1.]]))
        self.assertEqual(target.max(), kernel[25, 25])
        self.assertEqual(target[1, 1], target.max())

        # Past the end along x-axis.
        target = np.zeros((120, 160), dtype=np.float32)
        StereoVideoDataset._add_kernel(dataset, target, kernel, np.array([[165., 60.]]))
        self.assertNotEqual(target.max(), kernel[25, 25])
        self.assertEqual(target[60, 159], target.max())

if __name__ == "__main__":
    unittest.main()
