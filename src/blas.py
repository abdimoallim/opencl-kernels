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

  def isamax(self, x, n=None, incx=1):
    if n is None:
      n = len(x)
    if n == 0:
      return -1
    x = np.ascontiguousarray(x, dtype=np.float32)
    x_buf = cl.Buffer(
      self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=x
    )
    result_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, 4)
    program = self._load_kernel("isamax")
    local_size = min(256, n)
    global_size = ((n + local_size - 1) // local_size) * local_size
    kernel = program.isamax
    kernel(
      self.queue,
      (global_size,),
      (local_size,),
      np.int32(n),
      x_buf,
      np.int32(incx),
      result_buf,
      cl.LocalMemory(local_size * 4),
      cl.LocalMemory(local_size * 4),
    )
    result = np.zeros(1, dtype=np.int32)
    cl.enqueue_copy(self.queue, result, result_buf)
    self.queue.finish()
    return result[0]

  def srot(self, x, y, c, s, n=None, incx=1, incy=1):
    if n is None:
      n = len(x)
    if n == 0:
      return x, y
    x = np.ascontiguousarray(x, dtype=np.float32)
    y = np.ascontiguousarray(y, dtype=np.float32)
    x_buf = cl.Buffer(
      self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=x
    )
    y_buf = cl.Buffer(
      self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=y
    )
    program = self._load_kernel("srot")
    kernel = program.srot
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
      np.float32(c),
      np.float32(s),
    )
    cl.enqueue_copy(self.queue, x, x_buf)
    cl.enqueue_copy(self.queue, y, y_buf)
    self.queue.finish()
    return x, y

  def srotg(self, a, b):
    a = np.float32(a)
    b = np.float32(b)
    a_buf = cl.Buffer(
      self.context,
      cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
      hostbuf=np.array([a]),
    )
    b_buf = cl.Buffer(
      self.context,
      cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
      hostbuf=np.array([b]),
    )
    c_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, 4)
    s_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, 4)

    program = self._load_kernel("srotg")
    program.srotg(self.queue, (1,), None, a_buf, b_buf, c_buf, s_buf)

    new_a = np.zeros(1, dtype=np.float32)
    new_b = np.zeros(1, dtype=np.float32)
    c = np.zeros(1, dtype=np.float32)
    s = np.zeros(1, dtype=np.float32)

    cl.enqueue_copy(self.queue, new_a, a_buf)
    cl.enqueue_copy(self.queue, new_b, b_buf)
    cl.enqueue_copy(self.queue, c, c_buf)
    cl.enqueue_copy(self.queue, s, s_buf)

    self.queue.finish()

    return (new_a[0], new_b[0], c[0], s[0])

  def srotm(self, x, y, param, n=None, incx=1, incy=1):
    if n is None:
      n = len(x)

    if n == 0:
      return x, y

    x = np.ascontiguousarray(x, dtype=np.float32)
    y = np.ascontiguousarray(y, dtype=np.float32)
    param = np.ascontiguousarray(param, dtype=np.float32)

    x_buf = cl.Buffer(
      self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=x
    )
    y_buf = cl.Buffer(
      self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=y
    )
    param_buf = cl.Buffer(
      self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=param
    )

    program = self._load_kernel("srotm")
    program.srotm(
      self.queue,
      (n,),
      None,
      np.int32(n),
      x_buf,
      np.int32(incx),
      y_buf,
      np.int32(incy),
      param_buf,
    )

    cl.enqueue_copy(self.queue, x, x_buf)
    cl.enqueue_copy(self.queue, y, y_buf)

    self.queue.finish()

    return x, y

  def srotmg(self, d1, d2, x1, y1):
    d1 = np.float32(d1)
    d2 = np.float32(d2)
    x1 = np.float32(x1)
    y1 = np.float32(y1)

    d1_buf = cl.Buffer(
      self.context,
      cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
      hostbuf=np.array([d1]),
    )
    d2_buf = cl.Buffer(
      self.context,
      cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
      hostbuf=np.array([d2]),
    )
    x1_buf = cl.Buffer(
      self.context,
      cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
      hostbuf=np.array([x1]),
    )
    param_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, size=5 * 4)

    program = self._load_kernel("srotmg")
    program.srotmg(
      self.queue, (1,), None, d1_buf, d2_buf, x1_buf, np.float32(y1), param_buf
    )

    new_d1 = np.zeros(1, dtype=np.float32)
    new_d2 = np.zeros(1, dtype=np.float32)
    new_x1 = np.zeros(1, dtype=np.float32)
    param = np.zeros(5, dtype=np.float32)

    cl.enqueue_copy(self.queue, new_d1, d1_buf)
    cl.enqueue_copy(self.queue, new_d2, d2_buf)
    cl.enqueue_copy(self.queue, new_x1, x1_buf)
    cl.enqueue_copy(self.queue, param, param_buf)

    self.queue.finish()

    return (new_d1[0], new_d2[0], new_x1[0], param)
