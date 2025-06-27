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

  def saxpy(self, alpha, x, y, n=None, incx=1, incy=1):
    if n is None:
      n = len(x)

    x = np.ascontiguousarray(x, dtype=np.float32)
    y = np.ascontiguousarray(y, dtype=np.float32)

    x_buf = cl.Buffer(
      self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=x
    )
    y_buf = cl.Buffer(
      self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=y
    )

    program = self._load_kernel("saxpy")
    kernel = program.saxpy

    global_size = (n,)

    kernel(
      self.queue,
      global_size,
      None,
      np.int32(n),
      np.float32(alpha),
      x_buf,
      np.int32(incx),
      y_buf,
      np.int32(incy),
    )

    cl.enqueue_copy(self.queue, y, y_buf)
    self.queue.finish()

    return y

  def sdot(self, x, y, n=None, incx=1, incy=1):
    if n is None:
      n = len(x)

    if n == 0:
      return 0.0

    x = np.ascontiguousarray(x, dtype=np.float32)
    y = np.ascontiguousarray(y, dtype=np.float32)

    x_buf = cl.Buffer(
      self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=x
    )
    y_buf = cl.Buffer(
      self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=y
    )

    max_wg_size = 256
    global_size = min(1024, ((n + 63) // 64) * 64)

    partial_sums_buf = cl.Buffer(
      self.context, cl.mem_flags.READ_WRITE, size=global_size * 4
    )
    result_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, size=4)

    zero = np.zeros(global_size, dtype=np.float32)
    cl.enqueue_copy(self.queue, partial_sums_buf, zero)

    program = self._load_kernel("sdot")

    # first pass: compute partial sums
    kernel_sdot = program.sdot
    kernel_sdot(
      self.queue,
      (global_size,),
      None,
      np.int32(n),
      x_buf,
      np.int32(incx),
      y_buf,
      np.int32(incy),
      partial_sums_buf,
    )

    # second pass: reduction
    local_size = min(max_wg_size, global_size)
    kernel_reduce = program.reduce_sum
    kernel_reduce(
      self.queue,
      (global_size,),
      (local_size,),
      partial_sums_buf,
      result_buf,
      cl.LocalMemory(local_size * 4),
    )

    if global_size > local_size:
      partial_results = np.zeros(global_size, dtype=np.float32)
      cl.enqueue_copy(self.queue, partial_results, partial_sums_buf)
      final_result = np.sum(partial_results)
    else:
      final_result = np.zeros(1, dtype=np.float32)
      cl.enqueue_copy(self.queue, final_result, result_buf)
      final_result = final_result[0]

    self.queue.finish()
    return float(final_result)

  def snrm2(self, x, n=None, incx=1):
    if n is None:
      n = len(x)

    x = np.ascontiguousarray(x, dtype=np.float32)

    x_buf = cl.Buffer(
      self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=x
    )
    result_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, 4)

    zero = np.array([0.0], dtype=np.float32)
    cl.enqueue_copy(self.queue, result_buf, zero)

    program = self._load_kernel("snrm2")
    kernel = program.snrm2

    local_size = min(256, n)
    global_size = ((n + local_size - 1) // local_size) * local_size

    kernel(
      self.queue,
      (global_size,),
      (local_size,),
      np.int32(n),
      x_buf,
      np.int32(incx),
      result_buf,
      cl.LocalMemory(local_size * 4),
    )

    result = np.zeros(1, dtype=np.float32)
    cl.enqueue_copy(self.queue, result, result_buf)
    self.queue.finish()

    return np.sqrt(result[0])

  def sasum(self, x, n=None, incx=1):
    if n is None:
      n = len(x)

    if n == 0:
      return 0.0

    x = np.ascontiguousarray(x, dtype=np.float32)
    x_buf = cl.Buffer(
      self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=x
    )

    max_wg_size = 256
    global_size = min(1024, ((n + 63) // 64) * 64)

    partial_sums_buf = cl.Buffer(
      self.context, cl.mem_flags.READ_WRITE, size=global_size * 4
    )
    result_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, size=4)

    zero = np.zeros(global_size, dtype=np.float32)
    cl.enqueue_copy(self.queue, partial_sums_buf, zero)

    program = self._load_kernel("sasum")

    kernel_partial = program.sasum_partial
    kernel_partial(
      self.queue,
      (global_size,),
      None,
      np.int32(n),
      x_buf,
      np.int32(incx),
      partial_sums_buf,
    )

    local_size = min(max_wg_size, global_size)
    kernel_reduce = program.sasum_reduce
    kernel_reduce(
      self.queue,
      (global_size,),
      (local_size,),
      partial_sums_buf,
      result_buf,
      cl.LocalMemory(local_size * 4),
    )

    if global_size > local_size:
      partial_results = np.zeros(global_size, dtype=np.float32)
      cl.enqueue_copy(self.queue, partial_results, partial_sums_buf)
      final_result = np.sum(partial_results)
    else:
      final_result = np.zeros(1, dtype=np.float32)
      cl.enqueue_copy(self.queue, final_result, result_buf)
      final_result = final_result[0]

    self.queue.finish()

    return float(final_result)
