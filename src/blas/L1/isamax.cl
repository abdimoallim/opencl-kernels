__kernel void isamax(const int n, __global const float *x, const int incx,
                     __global int *result, __local float *local_max_values,
                     __local int *local_max_indices) {
  int gid = get_global_id(0);
  int lid = get_local_id(0);
  int group_size = get_local_size(0);

  float max_val = -INFINITY;
  int max_idx = -1;

  for (int i = gid; i < n; i += get_global_size(0)) {
    float val = fabs(x[i * incx]);

    if (val > max_val) {
      max_val = val;
      max_idx = i;
    }
  }

  local_max_values[lid] = max_val;
  local_max_indices[lid] = max_idx;

  barrier(CLK_LOCAL_MEM_FENCE);

  for (int stride = group_size / 2; stride > 0; stride >>= 1) {
    if (lid < stride) {
      if (local_max_values[lid + stride] > local_max_values[lid]) {
        local_max_values[lid] = local_max_values[lid + stride];
        local_max_indices[lid] = local_max_indices[lid + stride];
      }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (lid == 0) {
    result[get_group_id(0)] = local_max_indices[0];
  }
}
