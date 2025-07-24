import pyopencl as cl
import numpy as np
from kernels import sdpa_kernel


class SDPA:
  def __init__(self, ctx, queue):
    self.ctx = ctx
    self.queue = queue
    self.program = cl.Program(self.ctx, sdpa_kernel()).build()
    self.kernel = cl.Kernel(self.program, "scaled_dot_product_attention")

  def forward(self, q, k, v, scale=None):
    seq_len, d_k = q.shape
    if scale is None:
      scale = 1.0 / np.sqrt(d_k)
    q = q.astype(np.float32)
    k = k.astype(np.float32)
    v = v.astype(np.float32)
    q_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=q)
    k_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=k)
    v_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=v)
    scores = np.zeros((seq_len, seq_len), dtype=np.float32)
    attn_weights = np.zeros((seq_len, seq_len), dtype=np.float32)
    output = np.zeros((seq_len, d_k), dtype=np.float32)
    scores_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, scores.nbytes)
    attn_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, attn_weights.nbytes)
    output_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, output.nbytes)
    kernel = self.kernel
    kernel(
      self.queue,
      (seq_len, seq_len),
      None,
      q_buf,
      k_buf,
      v_buf,
      scores_buf,
      attn_buf,  #
      output_buf,
      np.int32(seq_len),
      np.int32(d_k),
      np.float32(scale),
    )
    cl.enqueue_copy(self.queue, output, output_buf)
    self.queue.finish()
    return output
