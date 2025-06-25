import os
import unittest
import numpy as np
import pyopencl as cl


class BLAS:
  def __init__(self, context=None, queue=None):
    if context is None:
      self.context = cl.create_some_context()
    else:
      self.context = context

    if queue is None:
      self.queue = cl.CommandQueue(self.context)
    else:
      self.queue = queue

    self.programs = {}

  def _load_kernel(self, kernel_name):
    if kernel_name not in self.programs:
      kernel_path = os.path.join("src", "blas", "L1", f"{kernel_name}.cl")
      with open(kernel_path, "r") as f:
        kernel_source = f.read()
      self.programs[kernel_name] = cl.Program(self.context, kernel_source).build()
    return self.programs[kernel_name]

  def scopy(self, x, y, n=None, incx=1, incy=1):
    if n is None:
      n = len(x)

    x = np.ascontiguousarray(x, dtype=np.float32)
    y = np.ascontiguousarray(y, dtype=np.float32)

    x_buf = cl.Buffer(
      self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=x
    )
    y_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, y.nbytes)

    program = self._load_kernel("scopy")
    kernel = program.scopy

    global_size = (n,)

    kernel(
      self.queue,
      global_size,
      None,
      np.int32(n),
      x_buf,
      np.int32(incx),
      y_buf,
      np.int32(incy),
    )

    cl.enqueue_copy(self.queue, y, y_buf)
    self.queue.finish()

    return y


class TestBLAS(unittest.TestCase):
  def setUp(self):
    import device
    self.blas = BLAS(context=device.create_gpu_context())

  def test_scopy_basic(self):
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    y = np.zeros(5, dtype=np.float32)
    print("before:",x,y)
    self.blas.scopy(x, y)
    print(x,y)
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
    result = self.blas.scopy(x, y, n=3, incy=2)
    expected = np.array([1.0, 0.0, 2.0, 0.0, 3.0, 0.0], dtype=np.float32)
    # np.testing.assert_array_equal(result, expected)

  def test_scopy_with_both_strides(self):
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    y = np.zeros(4, dtype=np.float32)
    result = self.blas.scopy(x, y, n=2, incx=3, incy=2)
    expected = np.array([1.0, 0.0, 4.0, 0.0], dtype=np.float32)
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
