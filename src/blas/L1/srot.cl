__kernel void srot(const int n, __global float *x, const int incx,
                   __global float *y, const int incy, const float c,
                   const float s) {
  int gid = get_global_id(0);

  if (gid < n) {
    int x_idx = gid * incx;
    int y_idx = gid * incy;

    float x_temp = x[x_idx];
    float y_temp = y[y_idx];

    x[x_idx] = c * x_temp + s * y_temp;
    y[y_idx] = c * y_temp - s * x_temp;
  }
}
