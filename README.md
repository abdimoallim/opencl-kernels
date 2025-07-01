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
- [ ] `?gbmv` - general band matrix-vector multiply ([`sgbmv`](/src/blas/L2/sgbmv.cl))
- [ ] `?sbmv` - symmetric band matrix-vector multiply ([`ssbmv`](/src/blas/L2/ssbmv.cl))
- [ ] `?hbmv` - Hermitian band matrix-vector multiply (complex)
- [ ] `?tbmv` - triangular band matrix-vector multiply
- [ ] `?tbsv` - triangular band system solve
- [ ] `?symv` - symmetric matrix-vector multiply ([`ssymv`](/src/blas/L2/ssymv.cl))
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

#### LAPACK linear systems (driver routines)

- [ ] `?gesv` - solve $Ax = b$ using LU factorization
- [ ] `?gbsv` - solve $Ax = b$ for band matrix using LU factorization
- [ ] `?gtsv` - solve $Ax = b$ for tridiagonal matrix using LU factorization
- [ ] `?posv` - solve $Ax = b$ for symmetric positive definite matrix using Cholesky factorization
- [ ] `?pbsv` - solve $Ax = b$ for symmetric positive definite band matrix using Cholesky factorization
- [ ] `?ptsv` - solve $Ax = b$ for symmetric positive definite tridiagonal matrix
- [ ] `?sysv` - solve $Ax = b$ for symmetric indefinite matrix using Bunch-Kaufman factorization
- [ ] `?hesv` - solve $Ax = b$ for Hermitian indefinite matrix using Bunch-Kaufman factorization
- [ ] `?spsv` - solve $Ax = b$ for symmetric indefinite matrix in packed storage
- [ ] `?hpsv` - solve $Ax = b$ for Hermitian indefinite matrix in packed storage

#### LAPACK linear systems (computational routines)

- [ ] `?getrf` - LU factorization of general matrix
- [ ] `?getrs` - solve system using precomputed LU factorization
- [ ] `?getri` - matrix inverse using precomputed LU factorization
- [ ] `?gbtrf` - LU factorization of band matrix
- [ ] `?gbtrs` - solve system using precomputed band LU factorization
- [ ] `?gttrf` - LU factorization of tridiagonal matrix
- [ ] `?gttrs` - solve system using precomputed tridiagonal LU factorization
- [ ] `?potrf` - Cholesky factorization of positive definite matrix
- [ ] `?potrs` - solve system using precomputed Cholesky factorization
- [ ] `?potri` - matrix inverse using precomputed Cholesky factorization
- [ ] `?pbtrf` - Cholesky factorization of positive definite band matrix
- [ ] `?pbtrs` - solve system using precomputed band Cholesky factorization
- [ ] `?pttrf` - factorization of positive definite tridiagonal matrix
- [ ] `?pttrs` - solve system using precomputed tridiagonal factorization
- [ ] `?sytrf` - Bunch-Kaufman factorization of symmetric indefinite matrix
- [ ] `?sytrs` - solve system using precomputed symmetric factorization
- [ ] `?sytri` - matrix inverse using precomputed symmetric factorization
- [ ] `?hetrf` - Bunch-Kaufman factorization of Hermitian indefinite matrix
- [ ] `?hetrs` - solve system using precomputed Hermitian factorization
- [ ] `?hetri` - matrix inverse using precomputed Hermitian factorization
- [ ] `?sptrf` - Bunch-Kaufman factorization of symmetric indefinite matrix in packed storage
- [ ] `?sptrs` - solve system using precomputed packed symmetric factorization
- [ ] `?sptri` - matrix inverse using precomputed packed symmetric factorization
- [ ] `?hptrf` - Bunch-Kaufman factorization of Hermitian indefinite matrix in packed storage
- [ ] `?hptrs` - solve system using precomputed packed Hermitian factorization
- [ ] `?hptri` - matrix inverse using precomputed packed Hermitian factorization
- [ ] `?trtrs` - solve triangular system
- [ ] `?trtri` - triangular matrix inverse

#### LAPACK least squares (driver routines)

- [ ] `?gels` - solve overdetermined or underdetermined system using QR or LQ factorization
- [ ] `?gelsy` - solve using complete orthogonal factorization with column pivoting
- [ ] `?gelss` - solve using SVD
- [ ] `?gelsd` - solve using SVD with divide-and-conquer

#### LAPACK least squares (computational routines)

- [ ] `?geqrf` - QR factorization
- [ ] `?geqp3` - QR factorization with column pivoting
- [ ] `?orgqr` - generate orthogonal matrix Q from QR factorization
- [ ] `?ormqr` - multiply by orthogonal matrix Q from QR factorization
- [ ] `?ungqr` - generate unitary matrix Q from QR factorization (complex)
- [ ] `?unmqr` - multiply by unitary matrix Q from QR factorization (complex)
- [ ] `?gelqf` - LQ factorization
- [ ] `?orglq` - generate orthogonal matrix Q from LQ factorization
- [ ] `?ormlq` - multiply by orthogonal matrix Q from LQ factorization
- [ ] `?unglq` - generate unitary matrix Q from LQ factorization (complex)
- [ ] `?unmlq` - multiply by unitary matrix Q from LQ factorization (complex)
- [ ] `?geqlf` - QL factorization
- [ ] `?orgql` - generate orthogonal matrix Q from QL factorization
- [ ] `?ormql` - multiply by orthogonal matrix Q from QL factorization
- [ ] `?ungql` - generate unitary matrix Q from QL factorization (complex)
- [ ] `?unmql` - multiply by unitary matrix Q from QL factorization (complex)
- [ ] `?gerqf` - RQ factorization
- [ ] `?orgrq` - generate orthogonal matrix Q from RQ factorization
- [ ] `?ormrq` - multiply by orthogonal matrix Q from RQ factorization
- [ ] `?ungrq` - generate unitary matrix Q from RQ factorization (complex)
- [ ] `?unmrq` - multiply by unitary matrix Q from RQ factorization (complex)
- [ ] `?tzrzf` - RZ factorization of trapezoidal matrix
- [ ] `?ormrz` - multiply by orthogonal matrix from RZ factorization
- [ ] `?unmrz` - multiply by unitary matrix from RZ factorization (complex)

#### LAPACK eigenvalue problems (driver routines)

- [ ] `?syev` - eigenvalues and eigenvectors of symmetric matrix
- [ ] `?syevd` - eigenvalues and eigenvectors using divide-and-conquer
- [ ] `?syevx` - selected eigenvalues and eigenvectors of symmetric matrix
- [ ] `?syevr` - eigenvalues and eigenvectors using MRRR algorithm
- [ ] `?heev` - eigenvalues and eigenvectors of Hermitian matrix
- [ ] `?heevd` - eigenvalues and eigenvectors using divide-and-conquer
- [ ] `?heevx` - selected eigenvalues and eigenvectors of Hermitian matrix
- [ ] `?heevr` - eigenvalues and eigenvectors using MRRR algorithm
- [ ] `?spev` - eigenvalues and eigenvectors of symmetric matrix in packed storage
- [ ] `?spevd` - eigenvalues and eigenvectors using divide-and-conquer
- [ ] `?spevx` - selected eigenvalues and eigenvectors in packed storage
- [ ] `?hpev` - eigenvalues and eigenvectors of Hermitian matrix in packed storage
- [ ] `?hpevd` - eigenvalues and eigenvectors using divide-and-conquer
- [ ] `?hpevx` - selected eigenvalues and eigenvectors in packed storage
- [ ] `?sbev` - eigenvalues and eigenvectors of symmetric band matrix
- [ ] `?sbevd` - eigenvalues and eigenvectors using divide-and-conquer
- [ ] `?sbevx` - selected eigenvalues and eigenvectors of band matrix
- [ ] `?hbev` - eigenvalues and eigenvectors of Hermitian band matrix
- [ ] `?hbevd` - eigenvalues and eigenvectors using divide-and-conquer
- [ ] `?hbevx` - selected eigenvalues and eigenvectors of band matrix
- [ ] `?stev` - eigenvalues and eigenvectors of symmetric tridiagonal matrix
- [ ] `?stevd` - eigenvalues and eigenvectors using divide-and-conquer
- [ ] `?stevx` - selected eigenvalues and eigenvectors of tridiagonal matrix
- [ ] `?stevr` - eigenvalues and eigenvectors using MRRR algorithm
- [ ] `?geev` - eigenvalues and eigenvectors of general matrix
- [ ] `?geevx` - eigenvalues, eigenvectors, and condition numbers
- [ ] `?ggev` - generalized eigenvalue problem for pair of general matrices
- [ ] `?ggevx` - generalized eigenvalue problem with condition numbers
- [ ] `?sygv` - generalized eigenvalue problem for symmetric matrices
- [ ] `?sygvd` - generalized eigenvalue problem using divide-and-conquer
- [ ] `?sygvx` - selected generalized eigenvalues for symmetric matrices
- [ ] `?hegv` - generalized eigenvalue problem for Hermitian matrices
- [ ] `?hegvd` - generalized eigenvalue problem using divide-and-conquer
- [ ] `?hegvx` - selected generalized eigenvalues for Hermitian matrices
- [ ] `?spgv` - generalized eigenvalue problem for symmetric matrices in packed storage
- [ ] `?spgvd` - generalized eigenvalue problem using divide-and-conquer
- [ ] `?spgvx` - selected generalized eigenvalues in packed storage
- [ ] `?hpgv` - generalized eigenvalue problem for Hermitian matrices in packed storage
- [ ] `?hpgvd` - generalized eigenvalue problem using divide-and-conquer
- [ ] `?hpgvx` - selected generalized eigenvalues in packed storage
- [ ] `?sbgv` - generalized eigenvalue problem for symmetric band matrices
- [ ] `?sbgvd` - generalized eigenvalue problem using divide-and-conquer
- [ ] `?sbgvx` - selected generalized eigenvalues for band matrices
- [ ] `?hbgv` - generalized eigenvalue problem for Hermitian band matrices
- [ ] `?hbgvd` - generalized eigenvalue problem using divide-and-conquer
- [ ] `?hbgvx` - selected generalized eigenvalues for band matrices

#### LAPACK eigenvalue problems (computational routines)

- [ ] `?sytrd` - reduce symmetric matrix to tridiagonal form
- [ ] `?hetrd` - reduce Hermitian matrix to tridiagonal form
- [ ] `?sptrd` - reduce symmetric matrix in packed storage to tridiagonal form
- [ ] `?hptrd` - reduce Hermitian matrix in packed storage to tridiagonal form
- [ ] `?sbtrd` - reduce symmetric band matrix to tridiagonal form
- [ ] `?hbtrd` - reduce Hermitian band matrix to tridiagonal form
- [ ] `?sterf` - eigenvalues of symmetric tridiagonal matrix using QR algorithm
- [ ] `?steqr` - eigenvalues and eigenvectors of symmetric tridiagonal matrix using QR algorithm
- [ ] `?stebz` - selected eigenvalues of symmetric tridiagonal matrix using bisection
- [ ] `?stein` - eigenvectors of symmetric tridiagonal matrix using inverse iteration
- [ ] `?stemr` - eigenvalues and eigenvectors using MRRR algorithm
- [ ] `?orgtr` - generate orthogonal matrix from symmetric reduction
- [ ] `?ormtr` - multiply by orthogonal matrix from symmetric reduction
- [ ] `?ungtr` - generate unitary matrix from Hermitian reduction
- [ ] `?unmtr` - multiply by unitary matrix from Hermitian reduction
- [ ] `?opgtr` - generate orthogonal matrix from packed symmetric reduction
- [ ] `?opmtr` - multiply by orthogonal matrix from packed symmetric reduction
- [ ] `?upgtr` - generate unitary matrix from packed Hermitian reduction
- [ ] `?upmtr` - multiply by unitary matrix from packed Hermitian reduction
- [ ] `?gehrd` - reduce general matrix to upper Hessenberg form
- [ ] `?orghr` - generate orthogonal matrix from Hessenberg reduction
- [ ] `?ormhr` - multiply by orthogonal matrix from Hessenberg reduction
- [ ] `?unghr` - generate unitary matrix from Hessenberg reduction
- [ ] `?unmhr` - multiply by unitary matrix from Hessenberg reduction
- [ ] `?gees` - Schur factorization of general matrix
- [ ] `?geesx` - Schur factorization with condition numbers
- [ ] `?trexc` - reorder Schur factorization
- [ ] `?trsen` - reorder Schur factorization and compute condition numbers
- [ ] `?trsyl` - solve Sylvester equation $AX + XB = C$
- [ ] `?ggees` - generalized Schur factorization
- [ ] `?gges` - generalized Schur factorization
- [ ] `?ggeesx` - generalized Schur factorization with condition numbers
- [ ] `?ggesx` - generalized Schur factorization with condition numbers
- [ ] `?tgevc` - eigenvectors of generalized eigenvalue problem
- [ ] `?tgexc` - reorder generalized Schur factorization
- [ ] `?tgsen` - reorder generalized Schur factorization and compute condition numbers
- [ ] `?tgsyl` - solve generalized Sylvester equation

#### LAPACK singular value decomposition (SVD)

- [ ] `?gesvd` - singular value decomposition
- [ ] `?gesdd` - singular value decomposition using divide-and-conquer
- [ ] `?gejsv` - singular value decomposition with high accuracy
- [ ] `?gesvj` - singular value decomposition using Jacobi method
- [ ] `?ggsvd` - generalized singular value decomposition
- [ ] `?ggsvd3` - generalized singular value decomposition (improved)

#### Auxiliary routines

- [ ] `?lacon` - estimate matrix 1-norm
- [ ] `?lacon` - estimate matrix 1-norm (improved)
- [ ] `?lange` - matrix norm (general matrix)
- [ ] `?lansy` - matrix norm (symmetric matrix)
- [ ] `?lanhe` - matrix norm (Hermitian matrix)
- [ ] `?lantr` - matrix norm (triangular matrix)
- [ ] `?lansp` - matrix norm (symmetric packed matrix)
- [ ] `?lanhp` - matrix norm (Hermitian packed matrix)
- [ ] `?lansb` - matrix norm (symmetric band matrix)
- [ ] `?lanhb` - matrix norm (Hermitian band matrix)
- [ ] `?lanst` - matrix norm (symmetric tridiagonal matrix)
- [ ] `?langt` - matrix norm (general tridiagonal matrix)
- [ ] `?langb` - matrix norm (general band matrix)
- [ ] `?gecon` - estimate condition number using LU factorization
- [ ] `?gbcon` - estimate condition number using band LU factorization
- [ ] `?gtcon` - estimate condition number using tridiagonal LU factorization
- [ ] `?pocon` - estimate condition number using Cholesky factorization
- [ ] `?pbcon` - estimate condition number using band Cholesky factorization
- [ ] `?ptcon` - estimate condition number using tridiagonal factorization
- [ ] `?sycon` - estimate condition number using symmetric factorization
- [ ] `?hecon` - estimate condition number using Hermitian factorization
- [ ] `?spcon` - estimate condition number using packed symmetric factorization
- [ ] `?hpcon` - estimate condition number using packed Hermitian factorization
- [ ] `?trcon` - estimate condition number of triangular matrix
- [ ] `?gerfs` - iterative refinement for general linear systems
- [ ] `?gbrfs` - iterative refinement for band linear systems
- [ ] `?gtrfs` - iterative refinement for tridiagonal linear systems
- [ ] `?porfs` - iterative refinement for symmetric positive definite systems
- [ ] `?pbrfs` - iterative refinement for band symmetric positive definite systems
- [ ] `?ptrfs` - iterative refinement for tridiagonal symmetric positive definite systems
- [ ] `?syrfs` - iterative refinement for symmetric indefinite systems
- [ ] `?herfs` - iterative refinement for Hermitian indefinite systems
- [ ] `?sprfs` - iterative refinement for packed symmetric indefinite systems
- [ ] `?hprfs` - iterative refinement for packed Hermitian indefinite systems
- [ ] `?trrfs` - iterative refinement for triangular systems
- [ ] `?geequ` - row and column scaling factors for general matrix
- [ ] `?gbequ` - row and column scaling factors for band matrix
- [ ] `?poequ` - scaling factors for symmetric positive definite matrix
- [ ] `?pbequ` - scaling factors for symmetric positive definite band matrix
- [ ] `?syequ` - scaling factors for symmetric indefinite matrix
- [ ] `?heequ` - scaling factors for Hermitian indefinite matrix
- [ ] `?laqtr` - solve quasi-triangular system
- [ ] `?laqr0` - eigenvalues using multishift QR algorithm
- [ ] `?laqr1` - single-shift QR step
- [ ] `?laqr2` - aggressive early deflation
- [ ] `?laqr3` - aggressive early deflation
- [ ] `?laqr4` - eigenvalues using multishift QR algorithm (improved)
- [ ] `?laqr5` - multishift QR sweep
