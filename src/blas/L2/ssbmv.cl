__kernel void ssbmv(const int n, const int k, const float alpha,
                    __global const float *A, const int lda,
                    __global const float *x, const int incx, const float beta,
                    __global float *y, const int incy, const int upper) {
  int row = get_global_id(0);

  if (row < n) {
    float sum = 0.0f;

    if (upper) { /* U */
      for (int col = row; col <= min(n - 1, row + k); col++) {
        int band_index = k + row - col;
        sum += A[band_index + col * lda] * x[col * incx];
      }

      for (int col = max(0, row - k); col < row; col++) {
        int band_index = k + col - row;
        sum += A[band_index + row * lda] * x[col * incx];
      }
    } else { /* L */
      for (int col = max(0, row - k); col <= row; col++) {
        int band_index = k + row - col;
        sum += A[band_index + col * lda] * x[col * incx];
      }

      for (int col = row + 1; col <= min(n - 1, row + k); col++) {
        int band_index = k + col - row;
        sum += A[band_index + row * lda] * x[col * incx];
      }
    }

    y[row * incy] = alpha * sum + beta * y[row * incy];
  }
}
