__kernel void scopy(const int n,
                   __global const float* x,
                   const int incx,
                   __global float* y,
                   const int incy) {
    int gid = get_global_id(0);
    
    if (gid >= n) return;
    
    int x_idx = gid * incx;
    int y_idx = gid * incy;
    
    y[y_idx] = x[x_idx];
}
