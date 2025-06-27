def sswap_kernel():
    return r"""__kernel void sswap(const int n, __global float *x, const int incx,
                    __global float *y, const int incy) {
  int gid = get_global_id(0);

  if (gid >= n)
    return;

  int x_idx = gid * incx;
  int y_idx = gid * incy;

  float temp = x[x_idx];

  x[x_idx] = y[y_idx];
  y[y_idx] = temp;
}"""
