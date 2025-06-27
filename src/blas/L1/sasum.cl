__kernel void sasum_partial(const int n, __global const float *x,
                            const int incx, __global float *partial_sums) {
  int gid = get_global_id(0);
  float sum = 0.0f;

  for (int i = gid; i < n; i += get_global_size(0)) {
    sum += fabs(x[i * incx]);
  }

  partial_sums[gid] = sum;
}

__kernel void sasum_reduce(__global const float *partial_sums,
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

  if (lid == 0) {
    result[get_group_id(0)] = local_sums[0];
  }
}
