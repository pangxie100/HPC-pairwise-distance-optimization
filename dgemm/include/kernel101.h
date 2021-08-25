#include "immintrin.h"
#define A(i,j) A[(i)+(j)*LDA]
#define B(i,j) B[(i)+(j)*LDB]
#define C(i,j) C[(i)+(j)*LDC]
// packing + cache blocking + 16 * 14 register blocking with AVX512 + loop unrolling * 4
// The least number of AVX512 register needed is 28 + 2 + 1 = 31 (max is 32)

// 8*4 register blocking, still can be improved by adding one 4*2 register blocking or 4*4

// C := alpha * op(A) * op(B) + beta * C
void pzyscale_C_k101(int M, int N, double beta, double *C, int LDC){
    int i, j;
    for (i = 0; i < M; i++){
        for (j = 0; j < N; j++){
            C(i,j) *= beta; 
        }
    }
}

void pzydgemm_cpu_opt_k101(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    int i, j, k;
    if (beta != 1.0) pzyscale_C_k101(M, N, beta, C, LDC);
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

// _mm512_loadu_pd(ptr_packing_a) use pointer as its input becasue it needs “void const* mem_addr" which is an address
// _mm512_set1_pd(*ptr_packing_b) use "*ptr_packing_b" because it needs a double-type number
// Later, we can compare the performance difference between _mm512_set1_pd() and _mm512_broadcastsd_pd() (if this broadcast function can be used)
// Here, we use 28(for C) + 2(for A) + 2(for B) = 32 AVX512 registers. In fact, we can give only 1 register to B.

// test and debug 
// printf("ax0[0] = %5.2f, ay0[0] = %5.2f, az0[0] = %5.2f\n", ax0[0], ay0[0], az0[0]);
// printf("b00[0] = %5.2f, b01[0] = %5.2f\n", b00[0], b01[0]);
// printf("cx0[0] = %5.2f, cy0[0] = %5.2f, cz0[0] = %5.2f\n", cx0[0], cy0[0], cz0[0]);
#define KERNEL_K1_16x14_avx512_intrinsics_packing\
    ax0 = _mm512_mul_pd(valpha, _mm512_loadu_pd(ptr_packing_a));\
    ay0 = _mm512_mul_pd(valpha, _mm512_loadu_pd(ptr_packing_a + 8));\
    b00 = _mm512_set1_pd(*ptr_packing_b);\
    b01 = _mm512_set1_pd(*(ptr_packing_b + 1));\
    cx0 = _mm512_fmadd_pd(ax0,b00,cx0);\
    cx1 = _mm512_fmadd_pd(ax0,b01,cx1);\
    cy0 = _mm512_fmadd_pd(ay0,b00,cy0);\
    cy1 = _mm512_fmadd_pd(ay0,b01,cy1);\
    b00 = _mm512_set1_pd(*(ptr_packing_b + 2));\
    b01 = _mm512_set1_pd(*(ptr_packing_b + 3));\
    cx2 = _mm512_fmadd_pd(ax0,b00,cx2);\
    cx3 = _mm512_fmadd_pd(ax0,b01,cx3);\
    cy2 = _mm512_fmadd_pd(ay0,b00,cy2);\
    cy3 = _mm512_fmadd_pd(ay0,b01,cy3);\
    b00 = _mm512_set1_pd(*(ptr_packing_b + 4));\
    b01 = _mm512_set1_pd(*(ptr_packing_b + 5));\
    cx4 = _mm512_fmadd_pd(ax0,b00,cx4);\
    cx5 = _mm512_fmadd_pd(ax0,b01,cx5);\
    cy4 = _mm512_fmadd_pd(ay0,b00,cy4);\
    cy5 = _mm512_fmadd_pd(ay0,b01,cy5);\
    b00 = _mm512_set1_pd(*(ptr_packing_b + 6));\
    b01 = _mm512_set1_pd(*(ptr_packing_b + 7));\
    cx6 = _mm512_fmadd_pd(ax0,b00,cx6);\
    cx7 = _mm512_fmadd_pd(ax0,b01,cx7);\
    cy6 = _mm512_fmadd_pd(ay0,b00,cy6);\
    cy7 = _mm512_fmadd_pd(ay0,b01,cy7);\
    b00 = _mm512_set1_pd(*(ptr_packing_b + 8));\
    b01 = _mm512_set1_pd(*(ptr_packing_b + 9));\
    cx8 = _mm512_fmadd_pd(ax0,b00,cx8);\
    cx9 = _mm512_fmadd_pd(ax0,b01,cx9);\
    cy8 = _mm512_fmadd_pd(ay0,b00,cy8);\
    cy9 = _mm512_fmadd_pd(ay0,b01,cy9);\
    b00 = _mm512_set1_pd(*(ptr_packing_b + 10));\
    b01 = _mm512_set1_pd(*(ptr_packing_b + 11));\
    cx10 = _mm512_fmadd_pd(ax0,b00,cx10);\
    cx11 = _mm512_fmadd_pd(ax0,b01,cx11);\
    cy10 = _mm512_fmadd_pd(ay0,b00,cy10);\
    cy11 = _mm512_fmadd_pd(ay0,b01,cy11);\
    b00 = _mm512_set1_pd(*(ptr_packing_b + 12));\
    b01 = _mm512_set1_pd(*(ptr_packing_b + 13));\
    cx12 = _mm512_fmadd_pd(ax0,b00,cx12);\
    cx13 = _mm512_fmadd_pd(ax0,b01,cx13);\
    cy12 = _mm512_fmadd_pd(ay0,b00,cy12);\
    cy13 = _mm512_fmadd_pd(ay0,b01,cy13);\
    ptr_packing_a += 16;\
    ptr_packing_b += 14;\
    k++;

// K4 is "k_inc & -4", is not a index, should add to inner_k_count: inner_k_count + K4

// test and debug
// printf("cx0[0] = %5.2f, cy0[0] = %5.2f, cz0[0] = %5.2f\n", cx0[0], cy0[0], cz0[0]);
#define macro_kernel_16xkx14_avx512_packing\
    cx0 = _mm512_setzero_pd();\
    cx1 = _mm512_setzero_pd();\
    cx2 = _mm512_setzero_pd();\
    cx3 = _mm512_setzero_pd();\
    cx4 = _mm512_setzero_pd();\
    cx5 = _mm512_setzero_pd();\
    cx6 = _mm512_setzero_pd();\
    cx7 = _mm512_setzero_pd();\
    cx8 = _mm512_setzero_pd();\
    cx9 = _mm512_setzero_pd();\
    cx10 = _mm512_setzero_pd();\
    cx11 = _mm512_setzero_pd();\
    cx12 = _mm512_setzero_pd();\
    cx13 = _mm512_setzero_pd();\
    cy0 = _mm512_setzero_pd();\
    cy1 = _mm512_setzero_pd();\
    cy2 = _mm512_setzero_pd();\
    cy3 = _mm512_setzero_pd();\
    cy4 = _mm512_setzero_pd();\
    cy5 = _mm512_setzero_pd();\
    cy6 = _mm512_setzero_pd();\
    cy7 = _mm512_setzero_pd();\
    cy8 = _mm512_setzero_pd();\
    cy9 = _mm512_setzero_pd();\
    cy10 = _mm512_setzero_pd();\
    cy11 = _mm512_setzero_pd();\
    cy12 = _mm512_setzero_pd();\
    cy13 = _mm512_setzero_pd();\
    for (k = inner_k_count; k < inner_k_count + K4;){\
        KERNEL_K1_16x14_avx512_intrinsics_packing\
        KERNEL_K1_16x14_avx512_intrinsics_packing\
        KERNEL_K1_16x14_avx512_intrinsics_packing\
        KERNEL_K1_16x14_avx512_intrinsics_packing\
    }\
    for (k = inner_k_count + K4; k < inner_k_end;){\
        KERNEL_K1_16x14_avx512_intrinsics_packing\
    }\
    _mm512_storeu_pd(&C(i,j), _mm512_add_pd(cx0, _mm512_loadu_pd(&C(i,j))));\
    _mm512_storeu_pd(&C(i,j+1), _mm512_add_pd(cx1, _mm512_loadu_pd(&C(i,j+1))));\
    _mm512_storeu_pd(&C(i,j+2), _mm512_add_pd(cx2, _mm512_loadu_pd(&C(i,j+2))));\
    _mm512_storeu_pd(&C(i,j+3), _mm512_add_pd(cx3, _mm512_loadu_pd(&C(i,j+3))));\
    _mm512_storeu_pd(&C(i,j+4), _mm512_add_pd(cx4, _mm512_loadu_pd(&C(i,j+4))));\
    _mm512_storeu_pd(&C(i,j+5), _mm512_add_pd(cx5, _mm512_loadu_pd(&C(i,j+5))));\
    _mm512_storeu_pd(&C(i,j+6), _mm512_add_pd(cx6, _mm512_loadu_pd(&C(i,j+6))));\
    _mm512_storeu_pd(&C(i,j+7), _mm512_add_pd(cx7, _mm512_loadu_pd(&C(i,j+7))));\
    _mm512_storeu_pd(&C(i,j+8), _mm512_add_pd(cx8, _mm512_loadu_pd(&C(i,j+8))));\
    _mm512_storeu_pd(&C(i,j+9), _mm512_add_pd(cx9, _mm512_loadu_pd(&C(i,j+9))));\
    _mm512_storeu_pd(&C(i,j+10), _mm512_add_pd(cx10, _mm512_loadu_pd(&C(i,j+10))));\
    _mm512_storeu_pd(&C(i,j+11), _mm512_add_pd(cx11, _mm512_loadu_pd(&C(i,j+11))));\
    _mm512_storeu_pd(&C(i,j+12), _mm512_add_pd(cx12, _mm512_loadu_pd(&C(i,j+12))));\
    _mm512_storeu_pd(&C(i,j+13), _mm512_add_pd(cx13, _mm512_loadu_pd(&C(i,j+13))));\
    _mm512_storeu_pd(&C(i+8,j), _mm512_add_pd(cy0, _mm512_loadu_pd(&C(i+8,j))));\
    _mm512_storeu_pd(&C(i+8,j+1), _mm512_add_pd(cy1, _mm512_loadu_pd(&C(i+8,j+1))));\
    _mm512_storeu_pd(&C(i+8,j+2), _mm512_add_pd(cy2, _mm512_loadu_pd(&C(i+8,j+2))));\
    _mm512_storeu_pd(&C(i+8,j+3), _mm512_add_pd(cy3, _mm512_loadu_pd(&C(i+8,j+3))));\
    _mm512_storeu_pd(&C(i+8,j+4), _mm512_add_pd(cy4, _mm512_loadu_pd(&C(i+8,j+4))));\
    _mm512_storeu_pd(&C(i+8,j+5), _mm512_add_pd(cy5, _mm512_loadu_pd(&C(i+8,j+5))));\
    _mm512_storeu_pd(&C(i+8,j+6), _mm512_add_pd(cy6, _mm512_loadu_pd(&C(i+8,j+6))));\
    _mm512_storeu_pd(&C(i+8,j+7), _mm512_add_pd(cy7, _mm512_loadu_pd(&C(i+8,j+7))));\
    _mm512_storeu_pd(&C(i+8,j+8), _mm512_add_pd(cy8, _mm512_loadu_pd(&C(i+8,j+8))));\
    _mm512_storeu_pd(&C(i+8,j+9), _mm512_add_pd(cy9, _mm512_loadu_pd(&C(i+8,j+9))));\
    _mm512_storeu_pd(&C(i+8,j+10), _mm512_add_pd(cy10, _mm512_loadu_pd(&C(i+8,j+10))));\
    _mm512_storeu_pd(&C(i+8,j+11), _mm512_add_pd(cy11, _mm512_loadu_pd(&C(i+8,j+11))));\
    _mm512_storeu_pd(&C(i+8,j+12), _mm512_add_pd(cy12, _mm512_loadu_pd(&C(i+8,j+12))));\
    _mm512_storeu_pd(&C(i+8,j+13), _mm512_add_pd(cy13, _mm512_loadu_pd(&C(i+8,j+13))));

/*
#define M_BLOCKING 192
#define N_BLOCKING 96 // very big 2048
#define K_BLOCKING 384
//*/

///*
#define M_BLOCKING 192
#define N_BLOCKING 112 
#define K_BLOCKING 384
//*/

/*
#define M_BLOCKING 48
#define N_BLOCKING 56
#define K_BLOCKING 32
//*/

/*
// test
#define M_BLOCKING 16
#define N_BLOCKING 14
#define K_BLOCKING 32
//*/

void pzypacking_b_k101(double *packsrc, double *packdst, int LDB, int dim_k, int dim_n){
    // kernel A * B is 16 * 14, thus we need 14 pointers to store each number in one row (col-major)
    double *src1, *src2, *src3, *src4, *src5, *src6, *src7, *src8, *src9, *src10, *src11, *src12, *src13, *src14, *dst;
    dst = packdst;
    int k, n;
    for (n = 0; n < dim_n; n += 14){
        src1 = packsrc + n * LDB;
        src2 = src1 + LDB;
        src3 = src2 + LDB;
        src4 = src3 + LDB;
        src5 = src4 + LDB;
        src6 = src5 + LDB;
        src7 = src6 + LDB;
        src8 = src7 + LDB;
        src9 = src8 + LDB;
        src10 = src9 + LDB;
        src11 = src10 + LDB;
        src12 = src11 + LDB;
        src13 = src12 + LDB;
        src14 = src13 + LDB;
        for (k = 0; k < dim_k; k++){
            *dst = *src1; src1++; dst++;
            *dst = *src2; src2++; dst++;
            *dst = *src3; src3++; dst++;
            *dst = *src4; src4++; dst++;
            *dst = *src5; src5++; dst++;
            *dst = *src6; src6++; dst++;
            *dst = *src7; src7++; dst++;
            *dst = *src8; src8++; dst++;
            *dst = *src9; src9++; dst++;
            *dst = *src10; src10++; dst++;
            *dst = *src11; src11++; dst++;
            *dst = *src12; src12++; dst++;
            *dst = *src13; src13++; dst++;
            *dst = *src14; src14++; dst++;
        }
    }
}

void pzypacking_a_k101(double *packsrc, double *packdst, int LDA, int dim_m, int dim_k){
    // kernel A * B is 16 * 14, we can use one pointer to store 16 numbers in one column (col-major)
    double *src, *dst;
    dst = packdst;
    //int i, k;
    //for (i = 0; i < dim_m, i += 16){
    int i, k, remain = dim_m;
    for (i = 0; remain > 15; remain -= 16, i += 16){
        src = packsrc + i;
        for (k = 0; k < dim_k; k++){
            _mm512_store_pd(dst, _mm512_loadu_pd(src));
            // we need to store the following 8 number to the following place,
            // so, the destination shoud also +8: dst + 8
            _mm512_store_pd(dst + 8, _mm512_loadu_pd(src + 8));
            src += LDA;
            dst += 16;
        }
    }
}

void pzydgemm_cpu_v101(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    if (beta != 1.0) pzyscale_C_k101(M, N, beta, C, LDC);
    // difference between malloc and aligned_alloc : 
    // https://stackoverflow.com/questions/39677063/difference-between-aligned-malloc-and-standard-malloc
    
    // usage of aligned_alloc: (the address is aligned)
    // https://en.cppreference.com/w/c/memory/aligned_alloc
    // https://zhuanlan.zhihu.com/p/111780698
    // it says that "K_BLOCKING * N_BLOCKING * sizeof(double)" or "K_BLOCKING * M_BLOCKING * sizeof(double)" should be an integral multiple of 4096 here
    double *b_buffer = (double *)aligned_alloc(4096,K_BLOCKING*N_BLOCKING*sizeof(double));
    double *a_buffer = (double *)aligned_alloc(4096,K_BLOCKING*M_BLOCKING*sizeof(double));
    double *ptr_packing_a, *ptr_packing_b;
    __m512d valpha = _mm512_set1_pd(alpha); // broadcast alpha to a 512-bit vector, the input is a double value
    __m512d ax0, ay0, b00, b01;
    __m512d cx0, cx1, cx2, cx3, cx4, cx5, cx6, cx7, cx8, cx9, cx10, cx11, cx12, cx13, cy0, cy1, cy2, cy3, cy4, cy5, cy6, cy7, cy8, cy9, cy10, cy11, cy12, cy13;

    int n_count = 0, k_count = 0, m_count = 0;
    int n_inc = 0, k_inc = 0, m_inc = 0;

    // get an integer which is divisible by 16, 14, 4(eg: 48 divided by 16 is 3)
    int M16 = 0, N14 = 0; // 16 * 14 kernel
    int K4 = 0; // loop unrolling 4 times
    int i = 0, j = 0, k = 0;
    int inner_m_count = 0, inner_n_count = 0, inner_k_count = 0, inner_k_end = 0;

    for (n_count = 0; n_count < N; n_count += n_inc){
        n_inc = (N - n_count > N_BLOCKING) ? N_BLOCKING : N - n_count;

        for (k_count = 0; k_count < K; k_count += k_inc){
                k_inc = K - k_count > K_BLOCKING ? K_BLOCKING : K - k_count;
                pzypacking_b_k101(B + k_count + n_count * LDB, b_buffer, LDB, k_inc, n_inc);

            for (m_count = 0; m_count < M; m_count += m_inc){
                //printf("m_count = %d\n", m_count);
                m_inc = M - m_count > M_BLOCKING ? M_BLOCKING : M - m_count;
                pzypacking_a_k101(A + m_count + k_count * LDA, a_buffer, LDA, m_inc, k_inc);

                //N14 = n_inc & -14; // this way only for 2^x, 14 is not.
                N14 = (n_inc / 14) * 14;
                M16 = m_inc & -16;
                K4 = k_inc & -4;

                //printf("M16 = %d, N14 = %d, K4 = %d\n", M16, N14, K4);
                for (inner_m_count = 0; inner_m_count < M16; inner_m_count += 16){
                    i = m_count + inner_m_count;
                    //printf("i = %d\n", i);
                    // each time for specific i, j loop means to compute all 16*14 matrix in same row on matrix C.
                    for (inner_n_count = 0; inner_n_count < N14; inner_n_count += 14){ 
                        j = n_count + inner_n_count;
                        //printf("j = %d\n", j);
                        inner_k_count = k_count;
                        //printf("inner_k_count = %d\n", inner_k_count);
                        inner_k_end = inner_k_count + k_inc;
                        //printf("inner_k_end = %d\n", inner_k_end);

                        ptr_packing_a = a_buffer + inner_m_count * k_inc;
                        ptr_packing_b = b_buffer + k_inc * inner_n_count;

                        /*
                        // test the output of a_buffer
                        double *testbuffer;
                        testbuffer = ptr_packing_a;
                        for (int b = 0; b < M_BLOCKING * K_BLOCKING; b++){
                            printf("%5.2f ", *testbuffer);
                            testbuffer++;   
                        }
                        printf("\n");
                        */

                        // add register reuse on C(i,j) to reduce the times of accessing C(i,j) in memory
                        macro_kernel_16xkx14_avx512_packing
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
                        //*/

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

                //printf("M16 = %d, m_inc = %d, N14 = %d, n_inc = %d\n", M16, m_inc, N14, n_inc);
                
                // here is not the end of the function, so shouldn't use "return"
                //if (M16 == m_inc && N14 == n_inc) return; 
                
                // Attention! I didn't check the edge case, so the following codes are not sure for correctness
                // boundary conditions
                if (M16 != m_inc) {
                    printf("enter edge case for m_inc. \n");
                    pzydgemm_cpu_opt_k101(m_inc - M16, n_inc, k_inc, alpha, &A(M16, 0), LDA, B, LDB, 1.0, &C(M16, 0), LDC); // A+M16 move to M16 row, because it's column major
                }
                if (N14 != n_inc) {
                    printf("enter edge case for n_inc. \n");
                    pzydgemm_cpu_opt_k101(M16, n_inc - N14, k_inc, alpha, A, LDA, &B(0, N14), LDB, 1.0, &C(0, N14), LDC);
                }
            }
            //printf("m_count = %d\n", m_count);

            // edge case for M
            if (m_inc != M_BLOCKING && m_count != 0){
                printf("enter edge case for M. \n");
                pzydgemm_cpu_opt_k101(m_inc, N_BLOCKING, K, alpha, &A(m_count, 0), LDA, &B(0, n_count), LDB, 1.0, &C(m_count, n_count), LDC);
            }

        }

        // edge case for N
        if (n_inc != N_BLOCKING && n_count != 0){
            printf("enter edge case for N. \n");
            pzydgemm_cpu_opt_k101(M, n_inc, K, alpha, A, LDA, &B(0, n_count), LDB, 1.0, &C(0, n_count), LDC);
        }
    }
    
}