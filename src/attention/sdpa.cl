__kernel void
scaled_dot_product_attention(__global const float *q, __global const float *k,
                             __global const float *v, __global float *scores,
                             __global float *attn_weights,
                             __global float *output, const int seq_len,
                             const int d_k, const float scale) {
  int i = get_global_id(0);
  int j = get_global_id(1);

  if (i >= seq_len || j >= seq_len)
    return;

  if (j == 0) {
    float sum = 0.0f;
    float max_val = -INFINITY;

    for (int k_idx = 0; k_idx < seq_len; k_idx++) {
      float score = 0.0f;

      for (int d = 0; d < d_k; d++) {
        score += q[i * d_k + d] * k[k_idx * d_k + d];
      }

      score *= scale;
      scores[i * seq_len + k_idx] = score;

      if (score > max_val) {
        max_val = score;
      }
    }

    for (int k_idx = 0; k_idx < seq_len; k_idx++) {
      float exp_val = exp(scores[i * seq_len + k_idx] - max_val);
      attn_weights[i * seq_len + k_idx] = exp_val;
      sum += exp_val;
    }

    for (int k_idx = 0; k_idx < seq_len; k_idx++) {
      attn_weights[i * seq_len + k_idx] /= sum;
    }

    for (int d = 0; d < d_k; d++) {
      float result = 0.0f;

      for (int k_idx = 0; k_idx < seq_len; k_idx++) {
        result += attn_weights[i * seq_len + k_idx] * v[k_idx * d_k + d];
      }

      output[i * d_k + d] = result;
    }
  }
}
