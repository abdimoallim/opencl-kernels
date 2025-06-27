__kernel void saxpy(const int n, const float alpha, __global const float *x,
                    const int incx, __global float *y, const int incy) {
  int gid = get_global_id(0);

  if (gid >= n)
    return;

  int x_idx = gid * incx;
  int y_idx = gid * incy;

  y[y_idx] = alpha * x[x_idx] + y[y_idx];
}
