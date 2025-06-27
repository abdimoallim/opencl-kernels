def sscal_kernel():
    return r"""__kernel void sscal(const int n, const float alpha, __global float *x,
                    const int incx) {
  int gid = get_global_id(0);

  if (gid >= n)
    return;

  int x_idx = gid * incx;

  x[x_idx] = alpha * x[x_idx];
}"""
