#include <stdio.h>
#include <stdlib.h>
#include "util.h"
#include "mkl.h"
#include <math.h>
// this computation way is for column-major
#define A(i,j) A[(i)+(j)*LDA]
#define A_T(i,j) A_T[(i)+(j)*LDA_T]
#define B(i,j) B[(i)+(j)*LDB]
#define C(i,j) C[(i)+(j)*LDC]
#define M 3
#define N 2
#define K 4

// Version 3: 1.simulate MATLAB with x' * y, thus, A^T. For A^T, each row is one observation.(one point in a k-dimension space), For B, cloumn.
//            2.use mkl lib
int main(int argc, char *argv[]){
    double *A, *B, *C, *A_T;
    int LDA = K, LDB = K, LDC = M, LDA_T = M;
    A = (double *)malloc(sizeof(double) * K * M);
    A_T = (double *)malloc(sizeof(double) * M * K);
    B = (double *)malloc(sizeof(double) * K * N);
    C = (double *)malloc(sizeof(double) * M * N);
    
    // Initialization
    printf("matrix x is initialized as:\n");
    sequential_initialize_matrix(A, K, M);
    print_matrix(A, K, M);
    print_array(A, K * M);

    printf("matrix y is initialized as:\n");
    sequential_initialize_matrix(B, K, N);
    print_matrix(B, K, N);
    print_array(B, K * N);

    //random_initialize_matrix(A, K, M);
    //random_initialize_matrix(B, K, N);

    
    // pairwise distance:
    // ||x-y||^2 = |x|^2 + |y|^2 - 2 * x * y

    // x (K * M) -> the transpose of x: x' (M * K)
    for (int k = 0; k < K; k++){
        for (int i = 0; i < M; i++){
            //printf("%5.2f ", A(k,i));
            A_T(i,k) = A(k,i); 
            //printf("%5.2f\n", A_T(i,k));      
        }
    }
    printf("the transpose matrix x' is:\n");
    print_matrix(A_T, M, K);
    print_array(A_T, M * K);

    // |x'|^2 (x' is a M * K matrix) -> |x'|^2 (M * 1)
    double *ptr_A = A;
    double sqrt_val_Va;  
    double ones[M];
    double *Vx = (double*)malloc(sizeof(double) * M);
    for (int i = 0; i < M; i++)
    {
        sqrt_val_Va = cblas_dnrm2(K, ptr_A, 1); 
        // sqrt_val_Va: the result is a square root of one column of matrix A: sqrt(a_1^2 + a_2^2 + ... + a_k^2)
        ptr_A += K;
        Vx[i] = sqrt_val_Va * sqrt_val_Va; // recover the result of one column of matrix A: a_1^2 + a_2^2 + ... + a_k^2
        ones[i] = 1;
    }
    printf("Vx (M * 1) is:\n");
    print_array(Vx, M);
    printf("ones (M * 1) is:\n");
    print_array(ones, M);

    // |y|^2 (y is a K * N matrix) -> |y|^2 (1 * N)
    double *ptr_B = B;
    double sqrt_val_Vb; 
    double *Vy = (double*)malloc(sizeof(double) * N);
    for (int j = 0; j < N; j++)
    {
        sqrt_val_Vb = cblas_dnrm2(K, ptr_B, 1);
        ptr_B += K;
        Vy[j] = sqrt_val_Vb * sqrt_val_Vb;
    }
    printf("Vy (1 * N) is:\n");
    print_array(Vy, N);

    // repmat: M * 1 -> M * N (copy |x'|^2 N times)
    // repmat: 1 * N -> M * N (copy |y|^2 M times)
    double value_y;
    double *ptr_C = C;
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
    printf("stretch matrix |y|^2 to M * N and add with stretched matrix |x'|^2 (M * N):\n");
    print_matrix(C, M, N);

    free(Vx);free(Vy);

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
    printf("After dgemm, the matrix C is:\n");
    print_matrix(C, M, N);

    // euclidean distance = |x - y|^2 = |x|^2 + |y|^2 - 2 * x * y
    // get the square root |x - y|
    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            C(i,j) = sqrt(C(i,j));
        }
    }

    // print the distance matrix
    printf("The distance matrix is:\n");
    print_matrix(C, M, N);

    free(A);free(B);free(C);free(A_T);
    return 0;
}