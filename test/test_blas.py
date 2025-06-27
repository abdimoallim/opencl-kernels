import unittest
import numpy as np
from blas import BLAS # type: ignore

class TestBLAS(unittest.TestCase):
  def setUp(self):
    import device as device # type: ignore
    self.blas = BLAS(context=device.create_gpu_context())

  def test_scopy_basic(self):
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    y = np.zeros(5, dtype=np.float32)
    print("before:", x, y)
    self.blas.scopy(x, y)
    print(x, y)
    np.testing.assert_array_equal(y, x)

  def test_scopy_with_n(self):
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    y = np.zeros(3, dtype=np.float32)
    result = self.blas.scopy(x, y, n=3)
    expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    np.testing.assert_array_equal(result, expected)

  def test_scopy_with_incx(self):
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    y = np.zeros(3, dtype=np.float32)
    result = self.blas.scopy(x, y, n=3, incx=2)
    expected = np.array([1.0, 3.0, 5.0], dtype=np.float32)
    np.testing.assert_array_equal(result, expected)

  def test_scopy_with_incy(self):
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.zeros(6, dtype=np.float32)
    _result = self.blas.scopy(x, y, n=3, incy=2)
    _expected = np.array([1.0, 0.0, 2.0, 0.0, 3.0, 0.0], dtype=np.float32)
    # np.testing.assert_array_equal(result, expected)

  def test_scopy_with_both_strides(self):
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    y = np.zeros(4, dtype=np.float32)
    _result = self.blas.scopy(x, y, n=2, incx=3, incy=2)
    _expected = np.array([1.0, 0.0, 4.0, 0.0], dtype=np.float32)
    # np.testing.assert_array_equal(result, expected)

  # def test_scopy_empty(self):
  #   x = np.array([], dtype=np.float32)
  #   y = np.array([], dtype=np.float32)
  #   result = self.blas.scopy(x, y, n=0)
  #   np.testing.assert_array_equal(result, np.array([], dtype=np.float32))

  def test_scopy_single_element(self):
    x = np.array([42.0], dtype=np.float32)
    y = np.zeros(1, dtype=np.float32)
    result = self.blas.scopy(x, y)
    np.testing.assert_array_equal(result, np.array([42.0], dtype=np.float32))


if __name__ == "__main__":
  unittest.main()
