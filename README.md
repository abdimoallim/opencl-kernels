### opencl-kernels

OpenCL implementation of various kernels—BLAS specification & LAPACK routines.

#### BLAS Level 1 (vector-vector operations)

- [ ] `?copy` - copy vector x to vector y ([`scopy`](/src/blas/L1/scopy.cl))
- [ ] `?swap` - swap vectors x and y ([`sswap`](/src/blas/L1/sswap.cl))
- [ ] `?scal` - scale vector by scalar: $x = \alpha \cdot x$ ([`sscal`](/src/blas/L1/sscal.cl))
- [ ] `?axpy` - vector plus scaled vector: $y = \alpha \cdot x + y$ ([`saxpy`](/src/blas/L1/saxpy.cl))
- [ ] `?dot` - dot product (real types only) ([`sdot`](/src/blas/L1/sdot.cl))
- [ ] `?dotc` - conjugate dot product (complex types only)
- [ ] `?dotu` - unconjugated dot product (complex types only)
- [ ] `?nrm2` - Euclidean norm ([`snrm2`](/src/blas/L1/snrm2.cl))
- [ ] `?asum` - sum of absolute values ([`sasum`](/src/blas/L1/sasum.cl))
- [ ] `i?amax` - index of maximum absolute value element ([`isamax`](/src/blas/L1/isamax.cl))
- [ ] `?rot` - apply plane rotation ([`srot`](/src/blas/L1/srot.cl))
- [ ] `?rotg` - generate plane rotation ([`srotg`](/src/blas/L1/srotg.cl))
- [ ] `?rotm` - apply modified plane rotation ([`srotm`](/src/blas/L1/srotm.cl))
- [ ] `?rotmg` - generate modified plane rotation ([`srotmg`](/src/blas/L1/srotmg.cl))

#### BLAS Level 2 (matrix-vector operations)

- [ ] `?gemv` - general matrix-vector multiply: $y = \alpha \cdot A \cdot x + \beta \cdot y$ ([`sgemv`](/src/blas/L2/sgemv.cl))
- [ ] `?ger` - general rank-1 update: $A = \alpha \cdot x \cdot y^T + A$ (real) ([`sger`](/src/blas/L2/sger.cl))
- [ ] `?geru` - general rank-1 update unconjugated (complex)
- [ ] `?gerc` - general rank-1 update conjugated (complex)
- [ ] `?gbmv` - general band matrix-vector multiply
- [ ] `?sbmv` - symmetric band matrix-vector multiply
- [ ] `?hbmv` - Hermitian band matrix-vector multiply (complex)
- [ ] `?tbmv` - triangular band matrix-vector multiply
- [ ] `?tbsv` - triangular band system solve
- [ ] `?symv` - symmetric matrix-vector multiply
- [ ] `?hemv` - Hermitian matrix-vector multiply (complex)
- [ ] `?syr` - symmetric rank-1 update
- [ ] `?her` - Hermitian rank-1 update (complex)
- [ ] `?syr2` - symmetric rank-2 update
- [ ] `?her2` - Hermitian rank-2 update (complex)
- [ ] `?spmv` - symmetric packed matrix-vector multiply
- [ ] `?hpmv` - Hermitian packed matrix-vector multiply (complex)
- [ ] `?spr` - symmetric packed rank-1 update
- [ ] `?hpr` - Hermitian packed rank-1 update (complex)
- [ ] `?spr2` - symmetric packed rank-2 update
- [ ] `?hpr2` - Hermitian packed rank-2 update (complex)
- [ ] `?trmv` - triangular matrix-vector multiply
- [ ] `?trsv` - triangular system solve
- [ ] `?tpmv` - triangular packed matrix-vector multiply
- [ ] `?tpsv` - triangular packed system solve

#### BLAS Level 3 (matrix-matrix operations)

- [ ] `?gemm` - general matrix-matrix multiply: $C = \alpha \cdot A \cdot B + \beta \cdot C$
- [ ] `?symm` - symmetric matrix-matrix multiply
- [ ] `?hemm` - Hermitian matrix-matrix multiply (complex)
- [ ] `?syrk` - symmetric rank-k update
- [ ] `?herk` - Hermitian rank-k update (complex)
- [ ] `?syr2k` - symmetric rank-2k update
- [ ] `?her2k` - Hermitian rank-2k update (complex)
- [ ] `?trmm` - triangular matrix-matrix multiply
- [ ] `?trsm` - triangular system solve with multiple right-hand sides
