__kernel void stbmv(const int n, const int k, const int is_upper,
                    const int is_trans, const int is_unit_diag,
                    __global const float *A, const int lda, __global float *x,
                    const int incx) {
  int row = get_global_id(0);

  if (row < n) {
    float sum = 0.0f;
    int start = is_upper ? max(0, row - k) : 0;
    int end = is_upper ? min(row + 1, n) : min(row + k + 1, n);

    for (int col = start; col < end; col++) {
      int band_idx = is_upper ? (k + row - col) : (k + col - row);
      float a_val = A[band_idx + col * lda];

      if (col == row && is_unit_diag) {
        sum += x[col * incx];
      } else {
        sum += a_val * x[col * incx];
      }
    }

    x[row * incx] = sum;
  }
}
