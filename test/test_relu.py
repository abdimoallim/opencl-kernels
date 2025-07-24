import unittest
import pyopencl as cl
import numpy as np
from relu import ReLU


class TestReLU(unittest.TestCase):
  def setUp(self):
    import device as device  # type: ignore

    self.ctx = device.create_gpu_context()
    self.queue = cl.CommandQueue(self.ctx)
    self.relu = ReLU(self.ctx, self.queue)

  def test_positive_values(self):
    x = np.array([1.0, 2.5, 0.1, 10.0], dtype=np.float32)
    result = self.relu.forward(x)
    expected = np.array([1.0, 2.5, 0.1, 10.0], dtype=np.float32)
    np.testing.assert_array_almost_equal(result, expected)

  def test_negative_values(self):
    x = np.array([-1.0, -2.5, -0.1, -10.0], dtype=np.float32)
    result = self.relu.forward(x)
    expected = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    np.testing.assert_array_almost_equal(result, expected)

  def test_mixed_values(self):
    x = np.array([-1.0, 0.0, 1.0, -2.5, 3.7], dtype=np.float32)
    result = self.relu.forward(x)
    expected = np.array([0.0, 0.0, 1.0, 0.0, 3.7], dtype=np.float32)
    np.testing.assert_array_almost_equal(result, expected)

  def test_zero_values(self):
    x = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    result = self.relu.forward(x)
    expected = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    np.testing.assert_array_almost_equal(result, expected)

  def test_large_array(self):
    x = np.random.randn(1000).astype(np.float32)
    result = self.relu.forward(x)
    expected = np.maximum(0, x)
    np.testing.assert_array_almost_equal(result, expected)

  def test_2d_array(self):
    x = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    result = self.relu.forward(x)
    expected = np.array([[0.0, 2.0], [3.0, 0.0]], dtype=np.float32)
    np.testing.assert_array_almost_equal(result, expected)


if __name__ == "__main__":
  unittest.main()
