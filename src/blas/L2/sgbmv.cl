__kernel void sgbmv(const int m, const int n, const int kl, const int ku,
                    const float alpha, __global const float *A, const int lda,
                    __global const float *x, const int incx, const float beta,
                    __global float *y, const int incy) {
  int row = get_global_id(0);

  if (row < m) {
    float sum = 0.0f;

    for (int j = 0; j < n; j++) {
      if (j >= row - kl && j <= row + ku) {
        int k = ku + row - j;

        sum += A[k + j * lda] * x[j * incx];
      }
    }

    y[row * incy] = alpha * sum + beta * y[row * incy];
  }
}
