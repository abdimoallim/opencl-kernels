def snrm2_kernel():
    return r"""__kernel void snrm2(const int n, __global const float *x, const int incx,
                    __global float *result, __local float *local_sum) {
  int gid = get_global_id(0);
  int lid = get_local_id(0);
  int local_size = get_local_size(0);

  local_sum[lid] = 0.0f;

  if (gid < n) {
    int x_idx = gid * incx;
    float val = x[x_idx];
    local_sum[lid] = val * val;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  for (int stride = local_size / 2; stride > 0; stride /= 2) {
    if (lid < stride) {
      local_sum[lid] += local_sum[lid + stride];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (lid == 0) {
    // atomic_add_global(result, local_sum[0]);
    result[0] = local_sum[0];
  }
}"""
