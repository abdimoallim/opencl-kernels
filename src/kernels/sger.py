def sger_kernel():
    return r"""__kernel void sger(const int m, const int n, const float alpha,
                   __global const float *x, const int incx,
                   __global const float *y, const int incy, __global float *A,
                   const int lda) {
  int row = get_global_id(0);
  int col = get_global_id(1);

  if (row < m && col < n) {
    A[row * lda + col] += alpha * x[row * incx] * y[col * incy];
  }
}"""
