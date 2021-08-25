#include "immintrin.h"
#define A(i,j) A[(i)+(j)*LDA]
#define B(i,j) B[(i)+(j)*LDB]
#define C(i,j) C[(i)+(j)*LDC]
// cache blocking + 8*4 register blocking with AVX2 + loop unrolling * 4

// 8*4 register blocking, still can be improved by adding one 4*2 register blocking

// C := alpha * op(A) * op(B) + beta * C
void pzyscale_C_k8(int M, int N, double beta, double *C, int LDC){
    int i,j;
    for (i = 0; i < M; i++){
        for (j = 0; j < N; j++){
            C(i,j) *= beta; 
        }
    }
}

void pzydgemm_cpu_opt_k8(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    int i,j,k;
    if (beta != 1.0) pzyscale_C_k8(M,N,beta,C,LDC);
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

// 为什么这里不需要 __m256声明？ 是因为使用_mm256命令会自动声明__m256吗？不是，因为函数中事先统一声明过了
// ax0: a00,a10,a20,a30; loadu 不需要对齐256
// ay0: a40,a50,a60,a70; loadu 不需要对齐256
// 学长这里的表达方式没看懂，学长的意思是说乘出来是一个4*8的部分结果C矩阵吗？不应该是8*4吗？
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

// K4 is "k_inc & -4", is not a index, should add to inner_k_count: inner_k_count + K4
#define macro_kernel_8xkx4_avx2\
    cx0 = _mm256_setzero_pd();\
    cx1 = _mm256_setzero_pd();\
    cx2 = _mm256_setzero_pd();\
    cx3 = _mm256_setzero_pd();\
    cy0 = _mm256_setzero_pd();\
    cy1 = _mm256_setzero_pd();\
    cy2 = _mm256_setzero_pd();\
    cy3 = _mm256_setzero_pd();\
    for (k = inner_k_count; k < inner_k_count + K4;){\
        KERNEL_K1_8x4_avx2_intrinsics\
        KERNEL_K1_8x4_avx2_intrinsics\
        KERNEL_K1_8x4_avx2_intrinsics\
        KERNEL_K1_8x4_avx2_intrinsics\
    }\
    for (k = inner_k_count + K4; k < inner_k_end;){\
        KERNEL_K1_8x4_avx2_intrinsics\
    }\
    _mm256_storeu_pd(&C(i,j), _mm256_add_pd(cx0, _mm256_loadu_pd(&C(i,j))));\
    _mm256_storeu_pd(&C(i,j+1), _mm256_add_pd(cx1, _mm256_loadu_pd(&C(i,j+1))));\
    _mm256_storeu_pd(&C(i,j+2), _mm256_add_pd(cx2, _mm256_loadu_pd(&C(i,j+2))));\
    _mm256_storeu_pd(&C(i,j+3), _mm256_add_pd(cx3, _mm256_loadu_pd(&C(i,j+3))));\
    _mm256_storeu_pd(&C(i+4,j), _mm256_add_pd(cy0, _mm256_loadu_pd(&C(i+4,j))));\
    _mm256_storeu_pd(&C(i+4,j+1), _mm256_add_pd(cy1, _mm256_loadu_pd(&C(i+4,j+1))));\
    _mm256_storeu_pd(&C(i+4,j+2), _mm256_add_pd(cy2, _mm256_loadu_pd(&C(i+4,j+2))));\
    _mm256_storeu_pd(&C(i+4,j+3), _mm256_add_pd(cy3, _mm256_loadu_pd(&C(i+4,j+3))));

///*
#define M_BLOCKING 192
#define N_BLOCKING 112 // very big 2048
#define K_BLOCKING 384
//*/

/*
// test
#define M_BLOCKING 16
#define N_BLOCKING 8
#define K_BLOCKING 32
//*/

void pzydgemm_cpu_v8(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    if (beta != 1.0) pzyscale_C_k8(M, N, beta, C, LDC);
    __m256d valpha = _mm256_set1_pd(alpha); // broadcast alpha to a 256-bit vector, the input is a double value
    __m256d ax0, ay0, b00, b01, b02, b03;
    __m256d cx0, cx1, cx2, cx3, cy0, cy1, cy2, cy3;

    int n_count = 0, k_count = 0, m_count = 0;
    int n_inc = 0, k_inc = 0, m_inc = 0;

    // get an integer which is divisible by 8, 4, 4(eg: 16 divided by 8 is 2)
    int M8 = 0, N4 = 0; // 8 * 4 kernel
    int K4 = 0; // loop unrolling 4 times
    int i = 0, j = 0, k = 0;
    int inner_m_count = 0, inner_n_count = 0, inner_k_count = 0, inner_k_end = 0;

    for (n_count = 0; n_count < N; n_count += n_inc){
        n_inc = (N - n_count > N_BLOCKING) ? N_BLOCKING : N - n_count;
        //n_block_start = n_block_start + n_inc;

        for (k_count = 0; k_count < K; k_count += k_inc){
                k_inc = K - k_count > K_BLOCKING ? K_BLOCKING : K - k_count;
                //k_block_start = k_block_start + k_inc;

            for (m_count = 0; m_count < M; m_count += m_inc){
                //printf("m_count = %d\n", m_count);
                m_inc = M - m_count > M_BLOCKING ? M_BLOCKING : M - m_count;
                //m_block_start = m_block_start + m_inc;

                N4 = n_inc & -4;
                M8 = m_inc & -8;
                K4 = k_inc & -4;
                //printf("K4 = %d\n", K4);
                for (inner_m_count = 0; inner_m_count < M8; inner_m_count += 8){
                    i = m_count + inner_m_count;
                    //printf("i = %d\n", i);
                    // each time for specific i, j loop means to compute all 8*4 matrix in same row on matrix C.
                    for (inner_n_count = 0; inner_n_count < N4; inner_n_count += 4){ 
                        j = n_count + inner_n_count;
                        //printf("j = %d\n", j);
                        inner_k_count = k_count;
                        //printf("inner_k_count = %d\n", inner_k_count);
                        inner_k_end = inner_k_count + k_inc;
                        //printf("inner_k_end = %d\n", inner_k_end);

                        // add register reuse on C(i,j) to reduce the times of accessing C(i,j) in memory
                        macro_kernel_8xkx4_avx2
                        /*
                        // print matrix
                        printf("matrix A:\n");
                        for (int m = 0; m < M; m++){
                            for (int n = 0; n < K; n++){
                                printf("%5.2f ", A[m + n * M]);    
                            }
                            printf("\n");    
                        }
                        printf("\n");
                        // print matrix
                        printf("matrix B:\n");
                        for (int m = 0; m < K; m++){
                            for (int n = 0; n < N; n++){
                                printf("%5.2f ", B[m + n * K]);    
                            }
                            printf("\n");    
                        }
                        printf("\n");
                        */
                        /*
                        // print matrix
                        printf("matrix C:\n");
                        for (int m = 0; m < M; m++){
                            for (int n = 0; n < N; n++){
                                printf("%5.2f ", C[m + n * M]);    
                            }
                            printf("\n");    
                        }
                        printf("\n");
                        //*/
                    }
                }

                //printf("M8 = %d, m_inc = %d, N4 = %d, n_inc = %d\n", M8, m_inc, N4, n_inc);
                
                // here is not the end of the function, so shouldn't use "return"
                //if (M8 == m_inc && N4 == n_inc) return; 
                
                // Attention! I didn't check the edge case, so the following codes are not sure for correctness
                // boundary conditions
                if (M8 != m_inc) {
                    printf("enter edge case for m_inc. \n");
                    pzydgemm_cpu_opt_k8(m_inc - M8, n_inc, k_inc, alpha, &A(M8, 0), LDA, B, LDB, 1.0, &C(M8, 0), LDC); // A+M8 move to M8 row, because it's column major
                }
                if (N4 != n_inc) {
                    printf("enter edge case for n_inc. \n");
                    pzydgemm_cpu_opt_k8(M8, n_inc - N4, k_inc, alpha, A, LDA, &B(0, N4), LDB, 1.0, &C(0, N4), LDC);
                }
            }
            //printf("m_count = %d\n", m_count);

            // edge case for M
            if (m_inc != M_BLOCKING && m_count != 0){
                printf("enter edge case for M. \n");
                pzydgemm_cpu_opt_k8(m_inc, N_BLOCKING, K, alpha, &A(m_count, 0), LDA, &B(0, n_count), LDB, 1.0, &C(m_count, n_count), LDC);
            }

        }

        // edge case for N
        if (n_inc != N_BLOCKING && n_count != 0){
            printf("enter edge case for N. \n");
            pzydgemm_cpu_opt_k8(M, n_inc, K, alpha, A, LDA, &B(0, n_count), LDB, 1.0, &C(0, n_count), LDC);
        }
    }
    
}