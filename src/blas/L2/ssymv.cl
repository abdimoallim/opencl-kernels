__kernel void ssymv(const int n, const float alpha, __global const float *A,
                    const int lda, __global const float *x, const int incx,
                    const float beta, __global float *y, const int incy) {
  int row = get_global_id(0);

  if (row < n) {
    float sum = 0.0f;

    for (int col = 0; col < n; col++) {
      sum += A[row * lda + col] * x[col * incx];
    }

    y[row * incy] = alpha * sum + beta * y[row * incy];
  }
}
