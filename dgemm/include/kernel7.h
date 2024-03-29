#include "immintrin.h"
#define A(i,j) A[(i)+(j)*LDA]
#define B(i,j) B[(i)+(j)*LDB]
#define C(i,j) C[(i)+(j)*LDC]
// 8*4 register blocking with AVX2 + loop unrolling * 4

// 8*4 register blocking, still can be improved by adding one 4*2 register blocking

// C := alpha * op(A) * op(B) + beta * C
void pzyscale_C_k7(int M, int N, double beta, double *C, int LDC){
    int i,j;
    for (i = 0; i < M; i++){
        for (j = 0; j < N; j++){
            C(i,j) *= beta; 
        }
    }
}

void pzydgemm_cpu_opt_k7(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    int i,j,k;
    if (beta != 1.0) pzyscale_C_k7(M,N,beta,C,LDC);
    for (i = 0; i < M; i++){
        for (j = 0; j < N; j++){
            // add register reuse on C(i,j) to reduce the times of accessing C(i,j) in memory
            double tmp = C(i,j);
            for (k = 0; k < K; k++){
                tmp += alpha * A(i,k) * B(k,j); 
            }
            C(i,j) = tmp;
        }
    }
}

// ax0: a00,a10,a20,a30; loadu: unaligned, not need to be aligned 
// ay0: a40,a50,a60,a70; loadu: unaligned, not need to be aligned 
#define KERNEL_K1_8x4_avx2_intrinsics\
    ax0 = _mm256_mul_pd(valpha, _mm256_loadu_pd(&A(i,k)));\
    ay0 = _mm256_mul_pd(valpha, _mm256_loadu_pd(&A(i+4,k)));\
    b00 = _mm256_broadcast_sd(&B(k,j));\
    b01 = _mm256_broadcast_sd(&B(k,j+1));\
    b02 = _mm256_broadcast_sd(&B(k,j+2));\
    b03 = _mm256_broadcast_sd(&B(k,j+3));\
    cx0 = _mm256_fmadd_pd(ax0,b00,cx0);\
    cx1 = _mm256_fmadd_pd(ax0,b01,cx1);\
    cx2 = _mm256_fmadd_pd(ax0,b02,cx2);\
    cx3 = _mm256_fmadd_pd(ax0,b03,cx3);\
    cy0 = _mm256_fmadd_pd(ay0,b00,cy0);\
    cy1 = _mm256_fmadd_pd(ay0,b01,cy1);\
    cy2 = _mm256_fmadd_pd(ay0,b02,cy2);\
    cy3 = _mm256_fmadd_pd(ay0,b03,cy3);\
    k++;

void pzydgemm_cpu_v7(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    int i,j,k;
    if (beta != 1.0) pzyscale_C_k7(M,N,beta,C,LDC);
    // get an integer which is divisible by 2(eg: 16 divided by 2 is 8)
    int M8=M&-8, N4=N&-4, K4= K&-4; // // return maximal numbers which is divisible by 8/4/4
    __m256d valpha = _mm256_set1_pd(alpha); // broadcast alpha to a 256-bit vector, input is a double value
    __m256d ax0, ay0, b00, b01, b02, b03;
    for (i = 0; i < M8; i+=8){
        // each time for specific i, j loop means to compute all 8*4 matrix in same row on matrix C.
        for (j = 0; j < N4; j+=4){ 
            // add register reuse on C(i,j) to reduce the times of accessing C(i,j) in memory
            __m256d cx0 = _mm256_setzero_pd(); // cx0: c00,c10,c20,c30
            __m256d cx1 = _mm256_setzero_pd();
            __m256d cx2 = _mm256_setzero_pd();
            __m256d cx3 = _mm256_setzero_pd();
            __m256d cy0 = _mm256_setzero_pd(); // cy0: c40,c50,c60,c70
            __m256d cy1 = _mm256_setzero_pd();
            __m256d cy2 = _mm256_setzero_pd();
            __m256d cy3 = _mm256_setzero_pd();
            // unroll the loop by 4 times
            for (k = 0; k < K4;){ // since we have "k++" in the following "#define" part, we don't need "k+=4" here
                KERNEL_K1_8x4_avx2_intrinsics
                KERNEL_K1_8x4_avx2_intrinsics
                KERNEL_K1_8x4_avx2_intrinsics
                KERNEL_K1_8x4_avx2_intrinsics
            }
            // deal with the edge case for K
            for (k = K4; k < K;){ // since we have "k++" in the following "#define" part, we don't need "k++" here
                KERNEL_K1_8x4_avx2_intrinsics
            }
            _mm256_storeu_pd(&C(i,j), _mm256_add_pd(cx0, _mm256_loadu_pd(&C(i,j))));
            _mm256_storeu_pd(&C(i,j+1), _mm256_add_pd(cx1, _mm256_loadu_pd(&C(i,j+1))));
            _mm256_storeu_pd(&C(i,j+2), _mm256_add_pd(cx2, _mm256_loadu_pd(&C(i,j+2))));
            _mm256_storeu_pd(&C(i,j+3), _mm256_add_pd(cx3, _mm256_loadu_pd(&C(i,j+3))));
            _mm256_storeu_pd(&C(i+4,j), _mm256_add_pd(cy0, _mm256_loadu_pd(&C(i+4,j))));
            _mm256_storeu_pd(&C(i+4,j+1), _mm256_add_pd(cy1, _mm256_loadu_pd(&C(i+4,j+1))));
            _mm256_storeu_pd(&C(i+4,j+2), _mm256_add_pd(cy2, _mm256_loadu_pd(&C(i+4,j+2))));
            _mm256_storeu_pd(&C(i+4,j+3), _mm256_add_pd(cy3, _mm256_loadu_pd(&C(i+4,j+3))));
        }
    }
    if(M8 == M && N4 == N) return;
    // boundary conditions
    if (M8 != M) pzydgemm_cpu_opt_k7(M - M8, N, K, alpha, A + M8, LDA, B, LDB, 1.0, &C(M8, 0), LDC); // A+M8 move to M8 row, because it's column major
    if (N4 != N) pzydgemm_cpu_opt_k7(M8, N - N4, K, alpha, A, LDA, &B(0, N4), LDB, 1.0, &C(0, N4), LDC);
}