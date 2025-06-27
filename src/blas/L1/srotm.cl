__kernel void srotm(const int n, __global float *x, const int incx,
                    __global float *y, const int incy,
                    __global const float *param) {
  int gid = get_global_id(0);

  if (gid < n) {
    int x_idx = gid * incx;
    int y_idx = gid * incy;

    float x_val = x[x_idx];
    float y_val = y[y_idx];

    float flag = param[0];

    if (flag == -1.0f) {
      float tmp_x = param[1] * x_val + param[3] * y_val;
      float tmp_y = param[2] * x_val + param[4] * y_val;

      // x' = h11*x + h12*y
      // y' = h21*x + h22*y

      x[x_idx] = tmp_x;
      y[y_idx] = tmp_y;
    } else if (flag == 0.0f) {

      // h11 = h22 = 1,
      // h21 = param[2],
      // h12 = param[3]

      float tmp_x = x_val + param[3] * y_val;
      float tmp_y = param[2] * x_val + y_val;

      x[x_idx] = tmp_x;
      y[y_idx] = tmp_y;
    } else if (flag == 1.0f) {

      // h11 = param[1],
      // h22 = param[4],
      // h21 = -1, h12 = 1

      // x' = h11*x + y
      // y' = -x + h22*y

      float tmp_x = param[1] * x_val + y_val;
      float tmp_y = -x_val + param[4] * y_val;

      x[x_idx] = tmp_x;
      y[y_idx] = tmp_y;
    }

    // flag == -2.0f (identity/noop)
  }
}
