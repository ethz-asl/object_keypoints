import sys
import unittest
import rospy
import rostest
import numpy as np
from scipy.spatial.transform import Rotation
from perception.utils import ros as ros_utils

class RosUtilsTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.node = rospy.init_node('test_ros_utils')

    def test_identity(self):
        T = np.eye(4)
        message = ros_utils.transform_to_message(T, 'parent', 'child', rospy.Time.now())
        T_out = ros_utils.message_to_transform(message)
        np.testing.assert_allclose(T_out, T)

    def test_random_rotation(self):
        T = np.eye(4)
        T[:3, :3] = Rotation.random().as_matrix()
        message = ros_utils.transform_to_message(T, 'parent', 'child', rospy.Time.now())
        T_out = ros_utils.message_to_transform(message)
        np.testing.assert_allclose(T_out, T)

    def test_random_rotation_with_translation(self):
        T = np.eye(4)
        T[:3, :3] = Rotation.random().as_matrix()
        T[:3, 3] = np.random.uniform(-1, 1, size=3)
        message = ros_utils.transform_to_message(T, 'parent', 'child', rospy.Time.now())
        T_out = ros_utils.message_to_transform(message)
        np.testing.assert_allclose(T_out, T)


if __name__ == "__main__":
    unittest.main()

