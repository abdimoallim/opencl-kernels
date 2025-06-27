def srotmg_kernel():
    return r"""__kernel void srotmg(__global float *d1, __global float *d2, __global float *x1,
                     const float y1, __global float *param) {
  float d1_val = *d1;
  float d2_val = *d2;
  float x1_val = *x1;

  float flag = -2.0f;
  float h11 = 1.0f, h12 = 0.0f, h21 = 0.0f, h22 = 1.0f;

  if (d1_val < 0.0f) {
    flag = -1.0f;
    *d1 = 0.0f;
    *d2 = 0.0f;
    *x1 = 0.0f;
  } else {
    float p1 = d1_val * x1_val;
    float p2 = d2_val * y1;

    if (p2 == 0.0f) {
      flag = -2.0f;
      *x1 = x1_val;
    } else if (fabs(p2) > fabs(p1) * 4.0f) {
      flag = 1.0f;
      float u = d1_val / d2_val;
      h11 = 1.0f;
      h12 = y1 / x1_val;
      h21 = -u * h12;
      h22 = 1.0f;
      *x1 = p2 / d2_val;
      *d1 = d2_val;
      *d2 = d1_val;
    } else if (fabs(p1) > fabs(p2) * 4.0f) {
      flag = 0.0f;
      float u = d2_val / d1_val;
      h12 = p2 / p1;
      h21 = -y1 / x1_val;
      *x1 = p1 / d1_val;
    } else {
      flag = -1.0f;
      float u = d1_val / d2_val;
      float tmp = sqrt(x1_val * x1_val + y1 * y1);
      h11 = x1_val / tmp;
      h12 = y1 / tmp;
      h21 = -h12 / u;
      h22 = h11 / u;
      *x1 = tmp;
    }

    if (flag <= 0.0f) {
      float tmp_d1 = d1_val * h11 * h11 + d2_val * h21 * h21;
      float tmp_d2 = d1_val * h12 * h12 + d2_val * h22 * h22;

      *d1 = tmp_d1;
      *d2 = tmp_d2;
    }
  }

  param[0] = flag;
  param[1] = h11;
  param[2] = h21;
  param[3] = h12;
  param[4] = h22;
}"""
