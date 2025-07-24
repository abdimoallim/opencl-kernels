import unittest
import pyopencl as cl
import numpy as np
from sdpa import SDPA


class TestSDPA(unittest.TestCase):
  def setUp(self):
    import device as device  # type: ignore

    self.ctx = device.create_gpu_context()
    self.queue = cl.CommandQueue(self.ctx)
    self.sdpa = SDPA(self.ctx, self.queue)

  def test_basic_attention(self):
    seq_len, d_k = 4, 8
    q = np.random.randn(seq_len, d_k).astype(np.float32)
    k = np.random.randn(seq_len, d_k).astype(np.float32)
    v = np.random.randn(seq_len, d_k).astype(np.float32)

    result = self.sdpa.forward(q, k, v)
    self.assertEqual(result.shape, (seq_len, d_k))
    self.assertFalse(np.isnan(result).any())
    self.assertFalse(np.isinf(result).any())

  def test_identity_attention(self):
    seq_len, d_k = 2, 4
    q = np.eye(seq_len, d_k).astype(np.float32)
    k = np.eye(seq_len, d_k).astype(np.float32)
    v = np.ones((seq_len, d_k)).astype(np.float32)

    result = self.sdpa.forward(q, k, v)
    self.assertEqual(result.shape, (seq_len, d_k))

  def test_custom_scale(self):
    seq_len, d_k = 3, 6
    q = np.random.randn(seq_len, d_k).astype(np.float32)
    k = np.random.randn(seq_len, d_k).astype(np.float32)
    v = np.random.randn(seq_len, d_k).astype(np.float32)
    scale = 0.5

    result = self.sdpa.forward(q, k, v, scale=scale)
    self.assertEqual(result.shape, (seq_len, d_k))

  def test_different_sizes(self):
    for seq_len in [1, 2, 8, 16]:
      for d_k in [4, 8, 16, 32]:
        q = np.random.randn(seq_len, d_k).astype(np.float32)
        k = np.random.randn(seq_len, d_k).astype(np.float32)
        v = np.random.randn(seq_len, d_k).astype(np.float32)

        result = self.sdpa.forward(q, k, v)
        self.assertEqual(result.shape, (seq_len, d_k))

  # When Q and K are zero matrices, all attention scores are zero, so after
  # softmax normalization each position gets equal weight (1/seq_len). The output
  # should be the average of all V vectors which is np.ones((seq_len, d_k)) since
  # V is all ones and not divided by seq_len.

  def test_zero_inputs(self):
    seq_len, d_k = 2, 4
    q = np.zeros((seq_len, d_k)).astype(np.float32)
    k = np.zeros((seq_len, d_k)).astype(np.float32)
    v = np.ones((seq_len, d_k)).astype(np.float32)

    result = self.sdpa.forward(q, k, v)
    self.assertEqual(result.shape, (seq_len, d_k))
    # expected = np.ones((seq_len, d_k)) * (1.0 / seq_len)
    expected = np.ones((seq_len, d_k))
    np.testing.assert_array_almost_equal(result, expected, decimal=5)

  def test_attention_weights_sum_to_one(self):
    seq_len, d_k = 3, 4
    q = np.random.randn(seq_len, d_k).astype(np.float32)
    k = np.random.randn(seq_len, d_k).astype(np.float32)
    v = np.random.randn(seq_len, d_k).astype(np.float32)

    scores = np.dot(q, k.T) / np.sqrt(d_k)
    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    for i in range(seq_len):
      self.assertAlmostEqual(np.sum(attention_weights[i]), 1.0, places=5)


if __name__ == "__main__":
  unittest.main()
