__kernel void stbsv(const int n, const int k, const int is_upper,
                    __global const float *A, const int lda, __global float *x,
                    const int incx) {
  int row = get_global_id(0);

  if (row < n) {
    float sum = x[row * incx];

    if (is_upper) {
      for (int col = row + 1; col < min(n, row + k + 1); col++) {
        int band_idx = k + row - (col - row);
        sum -= A[band_idx + col * lda] * x[col * incx];
      }

      x[row * incx] = sum / A[k + row * lda];
    } else {
      for (int col = max(0, row - k); col < row; col++) {
        int band_idx = row - col;
        sum -= A[band_idx + col * lda] * x[col * incx];
      }

      x[row * incx] = sum / A[row * lda];
    }
  }
}
