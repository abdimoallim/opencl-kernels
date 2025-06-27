import os
import numpy as np
import pyopencl as cl  # type: ignore


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

  def sswap(self, x, y, n=None, incx=1, incy=1):
    if n is None:
      n = len(x)

    x = np.ascontiguousarray(x, dtype=np.float32)
    y = np.ascontiguousarray(y, dtype=np.float32)

    x_buf = cl.Buffer(
      self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=x
    )
    y_buf = cl.Buffer(
      self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=y
    )

    program = self._load_kernel("sswap")
    kernel = program.sswap

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

    cl.enqueue_copy(self.queue, x, x_buf)
    cl.enqueue_copy(self.queue, y, y_buf)
    self.queue.finish()

    return x, y

  def sscal(self, alpha, x, n=None, incx=1):
    if n is None:
      n = len(x)

    x = np.ascontiguousarray(x, dtype=np.float32)

    x_buf = cl.Buffer(
      self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=x
    )

    program = self._load_kernel("sscal")
    kernel = program.sscal

    global_size = (n,)

    kernel(
      self.queue,
      global_size,
      None,
      np.int32(n),
      np.float32(alpha),
      x_buf,
      np.int32(incx),
    )

    cl.enqueue_copy(self.queue, x, x_buf)
    self.queue.finish()

    return x
