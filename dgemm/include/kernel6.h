#include "immintrin.h"
#define A(i,j) A[(i)+(j)*LDA]
#define B(i,j) B[(i)+(j)*LDB]
#define C(i,j) C[(i)+(j)*LDC]
// 4*4 register blocking with AVX2 + loop unrolling * 4

// 4*4 register blocking, still can be improved by adding one 2*2 register blocking

// C := alpha * op(A) * op(B) + beta * C
void pzyscale_C_k6(int M, int N, double beta, double *C, int LDC){
    int i,j;
    for (i = 0; i < M; i++){
        for (j = 0; j < N; j++){
            C(i,j) *= beta; 
        }
    }
}

void pzydgemm_cpu_opt_k6(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    int i,j,k;
    if (beta != 1.0) pzyscale_C_k6(M,N,beta,C,LDC);
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

// 为什么这里不需要 __m256声明？ 因为函数中事先统一声明过了
// ax0: a00,a10,a20,a30; loadu 不需要对齐256
// 注意：注释不能写在宏定义中，否则会报错

#define KERNEL_K1_4x4_avx2_intrinsics\
    ax0 = _mm256_mul_pd(valpha, _mm256_loadu_pd(&A(i,k)));\
    b00 = _mm256_broadcast_sd(&B(k,j));\
    b01 = _mm256_broadcast_sd(&B(k,j+1));\
    b02 = _mm256_broadcast_sd(&B(k,j+2));\
    b03 = _mm256_broadcast_sd(&B(k,j+3));\
    cx0 = _mm256_fmadd_pd(ax0,b00,cx0);\
    cx1 = _mm256_fmadd_pd(ax0,b01,cx1);\
    cx2 = _mm256_fmadd_pd(ax0,b02,cx2);\
    cx3 = _mm256_fmadd_pd(ax0,b03,cx3);\
    k++;

// 4 * 1 算子(kernel) 
#define KERNEL_K2_4x1_avx2_intrinsics\
    ax0 = _mm256_mul_pd(valpha, _mm256_loadu_pd(&A(i,k)));\
    b00 = _mm256_broadcast_sd(&B(k,j));\
    cx0 = _mm256_fmadd_pd(ax0,b00,cx0);\
    k++;

void pzydgemm_cpu_v6(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    int i,j,k;
    if (beta != 1.0) pzyscale_C_k6(M,N,beta,C,LDC);
    // get an integer which is divisible by 2(eg: 16 divided by 2 is 8)
    int M4=M&-4, N4=N&-4, K4= K&-4; // 返回整除4的最大整数
    __m256d valpha = _mm256_set1_pd(alpha); // broadcast alpha to a 256-bit vector, input is a double value
    __m256d ax0, b00, b01, b02, b03;
    for (i = 0; i < M4; i+=4){
        for (j = 0; j < N4; j+=4){
            // add register reuse on C(i,j) to reduce the times of accessing C(i,j) in memory
            __m256d cx0 = _mm256_setzero_pd(); // cx0: c00,c10,c20,c30
            __m256d cx1 = _mm256_setzero_pd();
            __m256d cx2 = _mm256_setzero_pd();
            __m256d cx3 = _mm256_setzero_pd();
            // unroll the loop by 4 times
            for (k = 0; k < K4;){ // 由于在宏定义中有k++，因此此处不需要再写
                KERNEL_K1_4x4_avx2_intrinsics
                KERNEL_K1_4x4_avx2_intrinsics
                KERNEL_K1_4x4_avx2_intrinsics
                KERNEL_K1_4x4_avx2_intrinsics
            }
            // deal with the edge case for K
            for (k = K4; k < K;){ // 由于在宏定义中有k++，因此此处不需要再写
                KERNEL_K1_4x4_avx2_intrinsics
            }
            _mm256_storeu_pd(&C(i,j), _mm256_add_pd(cx0, _mm256_loadu_pd(&C(i,j))));
            _mm256_storeu_pd(&C(i,j+1), _mm256_add_pd(cx1, _mm256_loadu_pd(&C(i,j+1))));
            _mm256_storeu_pd(&C(i,j+2), _mm256_add_pd(cx2, _mm256_loadu_pd(&C(i,j+2))));
            _mm256_storeu_pd(&C(i,j+3), _mm256_add_pd(cx3, _mm256_loadu_pd(&C(i,j+3))));
        }
    }
    if(M4 == M && N4 == N) return;
    // boundary conditions
    if (M4 != M) pzydgemm_cpu_opt_k6(M - M4, N, K, alpha, A + M4, LDA, B, LDB, 1.0, &C(M4, 0), LDC); // A+M4 move to M4 row, because it's column major
    // 疑问：C语言是怎么知道这是column major的？为什么A+M4就能正确移动到第M4行第一列的位置？
    if (N4 != N) pzydgemm_cpu_opt_k6(M4, N - N4, K, alpha, A, LDA, &B(0, N4), LDB, 1.0, &C(0, N4), LDC);
}