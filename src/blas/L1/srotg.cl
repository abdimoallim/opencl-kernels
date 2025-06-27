__kernel void srotg(__global float *a, __global float *b, __global float *c,
                    __global float *s) {
  float a_val = *a;
  float b_val = *b;
  float r, z, roe, scale;

  roe = (fabs(a_val) > fabs(b_val)) ? a_val : b_val;
  scale = fabs(a_val) + fabs(b_val);

  if (scale == 0.0f) {
    *c = 1.0f;
    *s = 0.0f;
    *a = 0.0f;
    *b = 0.0f;
  } else {
    float a_scaled = a_val / scale;
    float b_scaled = b_val / scale;
    r = scale * sqrt(a_scaled * a_scaled + b_scaled * b_scaled);
    r = copysign(r, roe);
    *c = a_val / r;
    *s = b_val / r;
    z = 1.0f;
    if (fabs(a_val) > fabs(b_val))
      z = *s;
    if (fabs(b_val) >= fabs(a_val) && *c != 0.0f)
      z = 1.0f / *c;
    *a = r;
    *b = z;
  }
}
