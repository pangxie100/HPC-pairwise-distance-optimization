#include "immintrin.h"
#define A(i,j) A[(i)+(j)*LDA]
#define B(i,j) B[(i)+(j)*LDB]
#define C(i,j) C[(i)+(j)*LDC]
// 4*4 register blocking with AVX2

// 4*4 register blocking, still can be improved by adding one 2*2 register blocking

// C := alpha * op(A) * op(B) + beta * C
void pzyscale_C_k5(int M, int N, double beta, double *C, int LDC){
    int i,j;
    for (i = 0; i < M; i++){
        for (j = 0; j < N; j++){
            C(i,j) *= beta; 
        }
    }
}

void pzydgemm_cpu_opt_k5(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    int i,j,k;
    if (beta != 1.0) pzyscale_C_k5(M,N,beta,C,LDC);
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

void pzydgemm_cpu_v5(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    int i,j,k;
    if (beta != 1.0) pzyscale_C_k5(M,N,beta,C,LDC);
    // get an integer which is divisible by 2(eg: 16 divided by 2 is 8)
    int M4=M&-4, N4=N&-4;
    __m256d valpha = _mm256_set1_pd(alpha); // broadcast alpha to a 256-bit vector, input is a double value
    for (i = 0; i < M4; i+=4){
        for (j = 0; j < N4; j+=4){
            // add register reuse on C(i,j) to reduce the times of accessing C(i,j) in memory
            __m256d cx0 = _mm256_setzero_pd(); // cx0: c00,c10,c20,c30
            __m256d cx1 = _mm256_setzero_pd();
            __m256d cx2 = _mm256_setzero_pd();
            __m256d cx3 = _mm256_setzero_pd();
            for (k = 0; k < K; k++){
                __m256d ax0 = _mm256_mul_pd(valpha, _mm256_loadu_pd(&A(i,k))); // ax0: a00,a10,a20,a30; loadu 不需要对齐256
                __m256d b00 = _mm256_broadcast_sd(&B(k,j)); // b00: b00,b00,b00,b00
                __m256d b01 = _mm256_broadcast_sd(&B(k,j+1));
                __m256d b02 = _mm256_broadcast_sd(&B(k,j+2));
                __m256d b03 = _mm256_broadcast_sd(&B(k,j+3));
                cx0 = _mm256_fmadd_pd(ax0,b00,cx0); // cx0: partial c00(adding a00 * b00), partial c10(adding a10 * b00), partial c20(adding a20 * b00), partial c30(adding a30 * b00)
                cx1 = _mm256_fmadd_pd(ax0,b01,cx1);
                cx2 = _mm256_fmadd_pd(ax0,b02,cx2);
                cx3 = _mm256_fmadd_pd(ax0,b03,cx3);
                // column [a00,a10,a20,a30] * row [b00,b01,b02,b03] = partial matrix (4 columns)[c00,c10,c20,c30][c01,c11,c21,c31][c02,c12,c22,c32][c03,c13,c23,c33]
            }
            // after k iterations, matrix [cx0,cx1,cx2,cx3] is a complete matrix result = matrix (k columns)[ax0,ax1,...,ax(k-1)] * matrix (k rows)[b0x,b1x,...,b(k-1)x] 
            
            // then, add this multiplication result matrix to matrix C
            _mm256_storeu_pd(&C(i,j), _mm256_add_pd(cx0, _mm256_loadu_pd(&C(i,j)))); // Cx0 = Cx0 + cx0
            _mm256_storeu_pd(&C(i,j+1), _mm256_add_pd(cx1, _mm256_loadu_pd(&C(i,j+1))));
            _mm256_storeu_pd(&C(i,j+2), _mm256_add_pd(cx2, _mm256_loadu_pd(&C(i,j+2))));
            _mm256_storeu_pd(&C(i,j+3), _mm256_add_pd(cx3, _mm256_loadu_pd(&C(i,j+3))));
        }
    }
    if(M4 == M && N4 == N) return;
    // boundary conditions
    if (M4 != M) pzydgemm_cpu_opt_k5(M - M4, N, K, alpha, A + M4, LDA, B, LDB, 1.0, &C(M4, 0), LDC); // A+M4 move to M4 row, because it's column major
    // 疑问：C语言是怎么知道这是column major的？为什么A+M4就能正确移动到第M4行第一列的位置？
    if (N4 != N) pzydgemm_cpu_opt_k5(M4, N - N4, K, alpha, A, LDA, &B(0, N4), LDB, 1.0, &C(0, N4), LDC);
}