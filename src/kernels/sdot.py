def sdot_kernel():
    return r"""__kernel void sdot(const int n, __global const float *x, const int incx,
                   __global const float *y, const int incy,
                   __global float *result) {
  int gid = get_global_id(0);
  float sum = 0.0f;

  for (int i = gid; i < n; i += get_global_size(0)) {
    sum += x[i * incx] * y[i * incy];
  }

  result[gid] = sum;
}

__kernel void reduce_sum(__global const float *partial_sums,
                         __global float *result, __local float *local_sums) {
  int gid = get_global_id(0);
  int lid = get_local_id(0);
  int group_size = get_local_size(0);

  local_sums[lid] = partial_sums[gid];

  barrier(CLK_LOCAL_MEM_FENCE);

  for (int stride = group_size / 2; stride > 0; stride >>= 1) {
    if (lid < stride) {
      local_sums[lid] += local_sums[lid + stride];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  /* one work-item per group should write */

  if (lid == 0) {
    // @todo: atomic_add_global can be replaced with a regular
    // addition since there is only one work group, PoCL on
    // IvyBridge does not support this
    // atomic_add_global(result, local_sum[0]);
    result[get_group_id(0)] = local_sums[0];
  }
}"""
