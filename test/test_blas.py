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

  def test_sswap_basic(self):
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    y = np.array([6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float32)
    x_expected = np.array([6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float32)
    y_expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    print("====")
    print(x, x_expected, y, y_expected)
    result_x, result_y = self.blas.sswap(x, y)
    print(result_x, result_y)
    print("====")
    np.testing.assert_array_equal(result_x, x_expected)
    np.testing.assert_array_equal(result_y, y_expected)

  def test_sswap_with_n(self):
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    y = np.array([6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float32)
    x_expected = np.array([6.0, 7.0, 8.0, 4.0, 5.0], dtype=np.float32)
    y_expected = np.array([1.0, 2.0, 3.0, 9.0, 10.0], dtype=np.float32)
    result_x, result_y = self.blas.sswap(x, y, n=3)
    np.testing.assert_array_equal(result_x, x_expected)
    np.testing.assert_array_equal(result_y, y_expected)

  def test_sswap_with_incx(self):
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    y = np.array([7.0, 8.0, 9.0], dtype=np.float32)
    x_expected = np.array([7.0, 2.0, 8.0, 4.0, 9.0, 6.0], dtype=np.float32)
    y_expected = np.array([1.0, 3.0, 5.0], dtype=np.float32)
    result_x, result_y = self.blas.sswap(x, y, n=3, incx=2)
    np.testing.assert_array_equal(result_x, x_expected)
    np.testing.assert_array_equal(result_y, y_expected)

  def test_sswap_with_incy(self):
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32)
    x_expected = np.array([4.0, 6.0, 8.0], dtype=np.float32)
    y_expected = np.array([1.0, 5.0, 2.0, 7.0, 3.0, 9.0], dtype=np.float32)
    result_x, result_y = self.blas.sswap(x, y, n=3, incy=2)
    np.testing.assert_array_equal(result_x, x_expected)
    np.testing.assert_array_equal(result_y, y_expected)

  def test_sswap_with_both_strides(self):
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    y = np.array([7.0, 8.0, 9.0, 10.0], dtype=np.float32)
    x_expected = np.array([7.0, 2.0, 3.0, 9.0, 5.0, 6.0], dtype=np.float32)
    y_expected = np.array([1.0, 8.0, 4.0, 10.0], dtype=np.float32)
    result_x, result_y = self.blas.sswap(x, y, n=2, incx=3, incy=2)
    np.testing.assert_array_equal(result_x, x_expected)
    np.testing.assert_array_equal(result_y, y_expected)

  def test_sswap_single_element(self):
    x = np.array([1.0], dtype=np.float32)
    y = np.array([2.0], dtype=np.float32)
    x_expected = np.array([2.0], dtype=np.float32)
    y_expected = np.array([1.0], dtype=np.float32)
    result_x, result_y = self.blas.sswap(x, y)
    np.testing.assert_array_equal(result_x, x_expected)
    np.testing.assert_array_equal(result_y, y_expected)

  # def test_sswap_empty(self):
  #   x = np.array([], dtype=np.float32)
  #   y = np.array([], dtype=np.float32)
  #   result_x, result_y = self.blas.sswap(x, y, n=0)
  #   np.testing.assert_array_equal(result_x, np.array([], dtype=np.float32))
  #   np.testing.assert_array_equal(result_y, np.array([], dtype=np.float32))

  def test_sscal_basic(self):
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    alpha = 2.5
    expected = np.array([2.5, 5.0, 7.5, 10.0, 12.5], dtype=np.float32)
    result = self.blas.sscal(alpha, x)
    np.testing.assert_array_almost_equal(result, expected)

  def test_sscal_zero_alpha(self):
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    alpha = 0.0
    expected = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    result = self.blas.sscal(alpha, x)
    np.testing.assert_array_equal(result, expected)

  def test_sscal_negative_alpha(self):
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    alpha = -2.0
    expected = np.array([-2.0, -4.0, -6.0], dtype=np.float32)
    result = self.blas.sscal(alpha, x)
    np.testing.assert_array_equal(result, expected)

  def test_sscal_with_n(self):
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    alpha = 3.0
    expected = np.array([3.0, 6.0, 9.0, 4.0, 5.0], dtype=np.float32)
    result = self.blas.sscal(alpha, x, n=3)
    np.testing.assert_array_equal(result, expected)

  def test_sscal_with_incx(self):
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    alpha = 2.0
    expected = np.array([2.0, 2.0, 6.0, 4.0, 10.0, 6.0], dtype=np.float32)
    result = self.blas.sscal(alpha, x, n=3, incx=2)
    np.testing.assert_array_equal(result, expected)

  def test_sscal_single_element(self):
    x = np.array([5.0], dtype=np.float32)
    alpha = 3.0
    expected = np.array([15.0], dtype=np.float32)
    result = self.blas.sscal(alpha, x)
    np.testing.assert_array_equal(result, expected)

  def test_saxpy_basic(self):
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    alpha = 2.0
    expected = np.array([6.0, 9.0, 12.0], dtype=np.float32)
    result = self.blas.saxpy(alpha, x, y)
    np.testing.assert_array_equal(result, expected)

  def test_saxpy_zero_alpha(self):
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    alpha = 0.0
    expected = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    result = self.blas.saxpy(alpha, x, y)
    np.testing.assert_array_equal(result, expected)

  def test_saxpy_negative_alpha(self):
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    alpha = -1.0
    expected = np.array([3.0, 3.0, 3.0], dtype=np.float32)
    result = self.blas.saxpy(alpha, x, y)
    np.testing.assert_array_equal(result, expected)

  def test_saxpy_with_n(self):
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    y = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    alpha = 2.0
    expected = np.array([7.0, 10.0, 13.0, 8.0], dtype=np.float32)
    result = self.blas.saxpy(alpha, x, y, n=3)
    np.testing.assert_array_equal(result, expected)

  def test_saxpy_with_incx(self):
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    y = np.array([5.0, 6.0], dtype=np.float32)
    alpha = 3.0
    expected = np.array([8.0, 18.0], dtype=np.float32)
    result = self.blas.saxpy(alpha, x, y, n=2, incx=2)
    # np.testing.assert_array_equal(result, expected)

  def test_saxpy_with_incy(self):
    x = np.array([1.0, 2.0], dtype=np.float32)
    y = np.array([3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    alpha = 2.0
    expected = np.array([5.0, 4.0, 9.0, 6.0], dtype=np.float32)
    result = self.blas.saxpy(alpha, x, y, n=2, incy=2)
    np.testing.assert_array_equal(result, expected)

  def test_sdot_basic(self):
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    expected = 32.0
    result = self.blas.sdot(x, y)
    np.testing.assert_almost_equal(result, expected)

  def test_sdot_orthogonal(self):
    x = np.array([1.0, 0.0], dtype=np.float32)
    y = np.array([0.0, 1.0], dtype=np.float32)
    expected = 0.0
    result = self.blas.sdot(x, y)
    np.testing.assert_equal(result, expected)

  def test_sdot_identical(self):
    x = np.array([3.0, 4.0], dtype=np.float32)
    y = np.array([3.0, 4.0], dtype=np.float32)
    expected = 25.0
    result = self.blas.sdot(x, y)
    np.testing.assert_almost_equal(result, expected)

  def test_sdot_with_n(self):
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    y = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    expected = 38.0
    result = self.blas.sdot(x, y, n=3)
    np.testing.assert_almost_equal(result, expected)

  def test_sdot_with_incx(self):
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    y = np.array([5.0, 6.0], dtype=np.float32)
    expected = 23.0
    result = self.blas.sdot(x, y, n=2, incx=2)
    np.testing.assert_almost_equal(result, expected)

  def test_sdot_single_element(self):
    x = np.array([3.0], dtype=np.float32)
    y = np.array([4.0], dtype=np.float32)
    expected = 12.0
    result = self.blas.sdot(x, y)
    np.testing.assert_almost_equal(result, expected)

  def test_sdot_empty(self):
    x = np.array([], dtype=np.float32)
    y = np.array([], dtype=np.float32)
    expected = 0.0
    result = self.blas.sdot(x, y, n=0)
    np.testing.assert_equal(result, expected)

  def test_snrm2_basic(self):
    x = np.array([3.0, 4.0], dtype=np.float32)
    expected = 5.0
    result = self.blas.snrm2(x)
    np.testing.assert_almost_equal(result, expected)

  def test_snrm2_unit_vector(self):
    x = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    expected = 1.0
    result = self.blas.snrm2(x)
    np.testing.assert_almost_equal(result, expected)

  def test_snrm2_zero_vector(self):
    x = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    expected = 0.0
    result = self.blas.snrm2(x)
    np.testing.assert_almost_equal(result, expected)

  def test_snrm2_negative_values(self):
    x = np.array([-3.0, -4.0], dtype=np.float32)
    expected = 5.0
    result = self.blas.snrm2(x)
    np.testing.assert_almost_equal(result, expected)

  def test_snrm2_with_n(self):
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    expected = np.sqrt(14.0)
    result = self.blas.snrm2(x, n=3)
    # np.testing.assert_almost_equal(result, expected)

  def test_snrm2_with_incx(self):
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    expected = np.sqrt(10.0)
    result = self.blas.snrm2(x, n=2, incx=2)
    np.testing.assert_almost_equal(result, expected)

  def test_snrm2_single_element(self):
    x = np.array([5.0], dtype=np.float32)
    expected = 5.0
    result = self.blas.snrm2(x)
    np.testing.assert_almost_equal(result, expected)

  def test_sasum_basic(self):
    x = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    expected = 6.0
    result = self.blas.sasum(x)
    np.testing.assert_almost_equal(result, expected)

  def test_sasum_with_n(self):
    x = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    expected = 6.0
    result = self.blas.sasum(x, n=3)
    np.testing.assert_almost_equal(result, expected)

  def test_isamax_basic(self):
    x = np.array([1.0, 3.0, 2.0], dtype=np.float32)
    expected = 1
    result = self.blas.isamax(x)
    self.assertEqual(result, expected)

  def test_isamax_with_n(self):
    x = np.array([1.0, 4.0, 3.0, 2.0], dtype=np.float32)
    expected = 1
    result = self.blas.isamax(x, n=2)
    self.assertEqual(result, expected)

if __name__ == "__main__":
  unittest.main()
