#include "immintrin.h"
#define A(i,j) A[(i)+(j)*LDA]
#define B(i,j) B[(i)+(j)*LDB]
#define C(i,j) C[(i)+(j)*LDC]
// packing + cache blocking + 8*4 register blocking with AVX2 + loop unrolling * 4
// The least number of AVX2 register needed is 8 + 2 + 1 = 11
// Maybe we can try 8 * 6 kernel, which needs 12 + 2 + 1 = 15 (max is 16)

// 8*4 register blocking, still can be improved by adding one 4*2 register blocking or 4*4

// C := alpha * op(A) * op(B) + beta * C
void pzyscale_C_k9(int M, int N, double beta, double *C, int LDC){
    int i, j;
    for (i = 0; i < M; i++){
        for (j = 0; j < N; j++){
            C(i,j) *= beta; 
        }
    }
}

void pzydgemm_cpu_opt_k9(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    int i, j, k;
    if (beta != 1.0) pzyscale_C_k9(M,N,beta,C,LDC);
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

#define KERNEL_K1_8x4_avx2_intrinsics_packing\
    ax0 = _mm256_mul_pd(valpha, _mm256_loadu_pd(ptr_packing_a));\
    ay0 = _mm256_mul_pd(valpha, _mm256_loadu_pd(ptr_packing_a + 4));\
    b00 = _mm256_broadcast_sd(ptr_packing_b);\
    b01 = _mm256_broadcast_sd(ptr_packing_b + 1);\
    b02 = _mm256_broadcast_sd(ptr_packing_b + 2);\
    b03 = _mm256_broadcast_sd(ptr_packing_b + 3);\
    cx0 = _mm256_fmadd_pd(ax0,b00,cx0);\
    cx1 = _mm256_fmadd_pd(ax0,b01,cx1);\
    cx2 = _mm256_fmadd_pd(ax0,b02,cx2);\
    cx3 = _mm256_fmadd_pd(ax0,b03,cx3);\
    cy0 = _mm256_fmadd_pd(ay0,b00,cy0);\
    cy1 = _mm256_fmadd_pd(ay0,b01,cy1);\
    cy2 = _mm256_fmadd_pd(ay0,b02,cy2);\
    cy3 = _mm256_fmadd_pd(ay0,b03,cy3);\
    ptr_packing_a += 8;\
    ptr_packing_b += 4;\
    k++;

// K4 is "k_inc & -4", is not a index, should add to inner_k_count: inner_k_count + K4
#define macro_kernel_8xkx4_avx2_packing\
    cx0 = _mm256_setzero_pd();\
    cx1 = _mm256_setzero_pd();\
    cx2 = _mm256_setzero_pd();\
    cx3 = _mm256_setzero_pd();\
    cy0 = _mm256_setzero_pd();\
    cy1 = _mm256_setzero_pd();\
    cy2 = _mm256_setzero_pd();\
    cy3 = _mm256_setzero_pd();\
    for (k = inner_k_count; k < inner_k_count + K4;){\
        KERNEL_K1_8x4_avx2_intrinsics_packing\
        KERNEL_K1_8x4_avx2_intrinsics_packing\
        KERNEL_K1_8x4_avx2_intrinsics_packing\
        KERNEL_K1_8x4_avx2_intrinsics_packing\
    }\
    for (k = inner_k_count + K4; k < inner_k_end;){\
        KERNEL_K1_8x4_avx2_intrinsics_packing\
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

void pzypacking_b_k9(double *packsrc, double *packdst, int LDB, int dim_k, int dim_n){
    // kernel A * B is 8 * 4, thus we need four pointers to store each number in one row (col-major)
    double *src1, *src2, *src3, *src4, *dst;
    dst = packdst;
    int k, n;
    for (n = 0; n < dim_n; n += 4){
        src1 = packsrc + n * LDB;
        src2 = src1 + LDB;
        src3 = src2 + LDB;
        src4 = src3 + LDB;
        for (k = 0; k < dim_k; k++){
            *dst = *src1; src1++; dst++;
            *dst = *src2; src2++; dst++;
            *dst = *src3; src3++; dst++;
            *dst = *src4; src4++; dst++;
        }
    }
}

void pzypacking_a_k9(double *packsrc, double *packdst, int LDA, int dim_m, int dim_k){
    // kernel A * B is 8 * 4, we can use one pointer to store 8 numbers in one column (col-major)
    double *src, *dst;
    dst = packdst;
    int i, k, remain = dim_m;
    for (i = 0; remain > 7; remain -= 8, i += 8){
        src = packsrc + i;
        for (k = 0; k < dim_k; k++){
            _mm512_store_pd(dst, _mm512_loadu_pd(src));
            src += LDA;
            dst += 8;
        }
    }
    // for the remained part which is smaller than 8, use 4 * 4 kernel style to store
    for (; remain > 3; remain -= 4, i += 4){
        src = packsrc + i;
        for (k = 0; k < dim_k; k++){
            _mm256_store_pd(dst, _mm256_loadu_pd(src));
            src += LDA;
            dst += 4;
        }
    }
}

void pzydgemm_cpu_v9(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    if (beta != 1.0) pzyscale_C_k9(M, N, beta, C, LDC);
    // difference between malloc and aligned_alloc : 
    // https://stackoverflow.com/questions/39677063/difference-between-aligned-malloc-and-standard-malloc
    
    // usage of aligned_alloc: 
    // https://en.cppreference.com/w/c/memory/aligned_alloc
    // https://zhuanlan.zhihu.com/p/111780698
    // it says that "K_BLOCKING * N_BLOCKING * sizeof(double)" or "K_BLOCKING * M_BLOCKING * sizeof(double)" should be an integral multiple of 4096 here
    double *b_buffer = (double *)aligned_alloc(4096,K_BLOCKING*N_BLOCKING*sizeof(double));
    double *a_buffer = (double *)aligned_alloc(4096,K_BLOCKING*M_BLOCKING*sizeof(double));
    double *ptr_packing_a, *ptr_packing_b;
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

        for (k_count = 0; k_count < K; k_count += k_inc){
                k_inc = K - k_count > K_BLOCKING ? K_BLOCKING : K - k_count;
                pzypacking_b_k9(B + k_count + n_count * LDB, b_buffer, LDB, k_inc, n_inc);

            for (m_count = 0; m_count < M; m_count += m_inc){
                //printf("m_count = %d\n", m_count);
                m_inc = M - m_count > M_BLOCKING ? M_BLOCKING : M - m_count;
                pzypacking_a_k9(A + m_count + k_count * LDA, a_buffer, LDA, m_inc, k_inc);

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

                        ptr_packing_a = a_buffer + inner_m_count * k_inc;
                        ptr_packing_b = b_buffer + k_inc * inner_n_count;

                        // add register reuse on C(i,j) to reduce the times of accessing C(i,j) in memory
                        macro_kernel_8xkx4_avx2_packing
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
                    pzydgemm_cpu_opt_k9(m_inc - M8, n_inc, k_inc, alpha, &A(M8, 0), LDA, B, LDB, 1.0, &C(M8, 0), LDC); // A+M8 move to M8 row, because it's column major
                }
                if (N4 != n_inc) {
                    printf("enter edge case for n_inc. \n");
                    pzydgemm_cpu_opt_k9(M8, n_inc - N4, k_inc, alpha, A, LDA, &B(0, N4), LDB, 1.0, &C(0, N4), LDC);
                }
            }
            //printf("m_count = %d\n", m_count);

            // edge case for M
            if (m_inc != M_BLOCKING && m_count != 0){
                printf("enter edge case for M. \n");
                pzydgemm_cpu_opt_k9(m_inc, N_BLOCKING, K, alpha, &A(m_count, 0), LDA, &B(0, n_count), LDB, 1.0, &C(m_count, n_count), LDC);
            }

        }

        // edge case for N
        if (n_inc != N_BLOCKING && n_count != 0){
            printf("enter edge case for N. \n");
            pzydgemm_cpu_opt_k9(M, n_inc, K, alpha, A, LDA, &B(0, n_count), LDB, 1.0, &C(0, n_count), LDC);
        }
    }
    free(a_buffer);free(b_buffer);
}