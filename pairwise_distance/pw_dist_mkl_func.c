#include <stdio.h>
#include "util.h"
#include "pw_dist_mkl_func.h"
#include "mkl.h"
#include <math.h>
#include <immintrin.h>
// this computation way is for column-major
//#define C(i,j) C[(i)+(j)*LDC]

//#define TIME_COUNT 1        // if you want to count time for performance profiling, uncomment this

// Version 4: 1.simulate MATLAB with x' * y, thus, A^T. For A^T, each row is one observation.(one point in a k-dimension space), For B, cloumn.
//            2.use mkl lib
//            3.use timer and make this as a function
//            4.solve the problem of matrix C's non-zero malloc by set all numbers from 3 matrix to zero after allocate space 
void pw_dist_mkl(double *A, double *B, double *C, int M, int N, int K){    
    // pairwise distance:
    // ||x-y||^2 = |x|^2 + |y|^2 - 2 * x * y

    //int LDC = M;
    double *ptr_A = A;
    double sqrt_val_Va;  
    double ones[M];
    double *Vx = (double*)malloc(sizeof(double) * M);
    double *ptr_B = B;
    double sqrt_val_Vb; 
    double *Vy = (double*)malloc(sizeof(double) * N);
    double value_y;
    double *ptr_C = C;
    // in Mac, comment all "printf"
    // type left click and scan one "print", then right click, 
    // then click "Change All Occurrences", then click "<-" on keyboard,
    // then type "//" to comment all "print"

    /*
    printf("in function, matrix A is:\n");
    print_matrix(A, K, M);
    printf("in function, matrix B is:\n");
    print_matrix(B, K, N);
    printf("in function, matrix C is:\n");
    print_matrix(C, M, N);
    */
#ifdef TIME_COUNT
    double t0, t1;
    double t_dnrm2_xy = 0.,t_daxpy_xy = 0., t_total = 0., t_dgemm = 0., t_sqrt = 0.;
    t0 = get_sec();
#endif
    // |x'|^2 (x' is a M * K matrix) -> |x'|^2 (M * 1)
    for (int i = 0; i < M; i++)
    {
        sqrt_val_Va = cblas_dnrm2(K, ptr_A, 1); 
        // sqrt_val_Va: the result is a square root of one column of matrix A: sqrt(a_1^2 + a_2^2 + ... + a_k^2)
        ptr_A += K;
        Vx[i] = sqrt_val_Va * sqrt_val_Va; // recover the result of one column of matrix A: a_1^2 + a_2^2 + ... + a_k^2
        ones[i] = 1;
    }
    /*
    printf("Vx (M * 1) is:\n");
    print_array(Vx, M);
    printf("ones (M * 1) is:\n");
    print_array(ones, M);
    */

    // |y|^2 (y is a K * N matrix) -> |y|^2 (1 * N)
    for (int j = 0; j < N; j++)
    {
        sqrt_val_Vb = cblas_dnrm2(K, ptr_B, 1);
        ptr_B += K;
        Vy[j] = sqrt_val_Vb * sqrt_val_Vb;
    }
    /*
    printf("Vy (1 * N) is:\n");
    print_array(Vy, N);
    */
#ifdef TIME_COUNT
    t1 = get_sec();
    t_dnrm2_xy = (t1 - t0);
    t_total += (t1 - t0);
    t0 = t1;
#endif 
    // repmat: M * 1 -> M * N (copy |x'|^2 N times)
    // repmat: 1 * N -> M * N (copy |y|^2 M times)
    for (int i = 0; i < N; i++)
    {
        value_y = Vy[i];
        cblas_daxpy(M, 1., Vx, 1, ptr_C, 1);
        // M elements in this vector operation, 1.0 * Vx[0...M-1] + ptr_C[0...M-1]  
        // add elements of Vector x to the first column of matrix C.
        cblas_daxpy(M, value_y, ones, 1, ptr_C, 1);
        // M elements in this vector operation, value_y * ones[0...M-1] + ptr_C[0...M-1]  
        // add elements of Vector ones[y[i] y[i] ... y[i]] to the first column of matrix C.
        ptr_C += M; // switch to the next column of matrix C (column-major)
    }
    /*
    printf("stretch matrix |y|^2 to M * N and add with stretched matrix |x'|^2 (M * N):\n");
    print_matrix(C, M, N);
    */
#ifdef TIME_COUNT
    t1 = get_sec();
    t_daxpy_xy = (t1 - t0);
    t_total += (t1 - t0);
    t0 = t1;
#endif
    // - 2 * x' * y (x -> the transpose of x': M * K)

    // first parameter decides the pointer of matrix goes in column-major(CblasColMajor) or row-major(CblasRowMajor)
    // second parameter and third parameter determines whether the first input matrix or the second input matrix needs to be transposed for the following matrix multiplication: CblasNoTrans, CblasTrans
    // Here, "CblasTrans, CblasNoTrans" means matrix A need to be transposed to A^T, B not need.

    // "M,N,K" here represents that in the real matrix multiplication, real_A,real_B,real_C are M*K, K*N, M*N 
    // M is rows of matrix real_A(here is transposed A: A^T) and rows of matrix real_C(here is C), 
    // N is columns of matrix real_B(here is B) and columns of matrix real_C(here is C)
    // K is columns of matrix real_A(here is transposed A: A^T) and rows of matrix real_B(here is B)
    // https://software.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/blas-and-sparse-blas-routines/blas-routines/blas-level-3-routines/cblas-gemm.html

    // -2.0 * matrix A * matrix B (-2xy) + 1.0 * matrix C (stretched result got before: x^2 + y^2)
    // parameters order: A,K,B,K,...,C,M means that input matrix A's leading dimension is K, due to column-major here, when we turn to next column, we add K(row numbers) to the pointer of matrix 
    // We can consider "leading dimension" as the number when we decide to turn to next line of matrix(different between col-major and row-major). So, same to B and C, their row numbers are K and M. Thus, K,...,K,...,M
    // For these parameters, we don't consider about the transpose case. For example, A needs to be transposed here, but we input K(row numbers) as the original version of A: K*M, not M*K.
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, M, N, K, -2., A, K, B, K, 1., C, M);
    /*
    printf("After dgemm, the matrix C is:\n");
    print_matrix(C, M, N);
    */
#ifdef TIME_COUNT
    t1 = get_sec();
    t_dgemm = (t1 - t0);
    t_total += (t1 - t0);
    t0 = t1;
#endif
    // euclidean distance = |x - y|^2 = |x|^2 + |y|^2 - 2 * x * y
    // get the square root |x - y|
    /*
    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            C(i,j) = sqrt(C(i,j));
        }
    }
    //*/
    ///*
    
    int i, MN = M * N;
    int MN4 = MN & -4;
    //printf("MN = %d\n", MN);
    //printf("MN4 = %d\n", MN4);
    for (i = 0; i < MN4; i += 4)
    {
        _mm256_storeu_pd(C + i, _mm256_sqrt_pd(_mm256_loadu_pd(C + i)));
    }
    //printf("i = %d\n", i);
    if (i != MN)
    {
        for (; i < MN; i++)
        {
            C[i] = sqrt(C[i]);
        }
    }
    //*/
#ifdef TIME_COUNT
    t1 = get_sec();
    t_sqrt = (t1 - t0);
    t_total += (t1 - t0);
    t0 = t1;
#endif
    /*
    // print the distance matrix
    printf("The distance matrix is:\n");
    print_matrix(C, M, N);
    //*/
    free(Vx);free(Vy);
#ifdef TIME_COUNT
    t1 = get_sec();
    t_total += (t1 - t0);
    printf("t-dnrm2_xy : %f s\n", t_dnrm2_xy);
    printf("t-daxpy_xy : %f s\n", t_daxpy_xy);
    printf("t-dgemm : %f s\n", t_dgemm);
    printf("t-sqrt : %f s\n", t_sqrt);
    printf("t-total : %f s\n", t_total);
    printf("potential improvement ratio by fusing: %f %%\n", 100. - t_dgemm / t_total * 100);
    printf("\n");
#endif
    return;
}