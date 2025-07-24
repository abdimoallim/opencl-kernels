import pyopencl as cl
import numpy as np
from kernels import relu_kernel


class ReLU:
  def __init__(self, ctx, queue):
    self.ctx = ctx
    self.queue = queue
    self.program = cl.Program(self.ctx, relu_kernel()).build()

  def forward(self, x):
    x = x.astype(np.float32)
    output = np.zeros_like(x)
    x_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=x)
    output_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, output.nbytes)
    kernel = self.program.relu
    kernel(self.queue, (x.size,), None, x_buf, output_buf, np.int32(x.size))
    cl.enqueue_copy(self.queue, output, output_buf)
    self.queue.finish()
    return output
