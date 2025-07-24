__kernel void relu(__global const float *input, __global float *output,
                   const int size) {
  int gid = get_global_id(0);

  if (gid >= size)
    return;

  output[gid] = fmax(0.0f, input[gid]);
}
