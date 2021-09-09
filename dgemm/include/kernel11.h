#include "immintrin.h"
#define A(i,j) A[(i)+(j)*LDA]
#define B(i,j) B[(i)+(j)*LDB]
#define C(i,j) C[(i)+(j)*LDC]
// discontinuous packing(2 - 2 packing) + cache blocking + 24*8 register blocking with AVX512 + loop unrolling * 4
// The least number of AVX512 register needed is 24 + 3 +1 = 28 (max is 32)

///*
#define M_BLOCKING 192
//#define N_BLOCKING 112 // very big 2240
//#define N_BLOCKING 2240 // get a higher performance
#define N_BLOCKING 168 // 3360
#define K_BLOCKING 384
//*/

/*
#define M_BLOCKING 192
#define N_BLOCKING 96 // very big 2048
#define K_BLOCKING 384
//*/

/*
// test
#define M_BLOCKING 24
#define N_BLOCKING 8
#define K_BLOCKING 32
//*/

// 24*8 register blocking, still can be improved by adding one 24*4 register blocking or 24*2 ...

// C := alpha * op(A) * op(B) + beta * C
void pzyscale_C_k11(int M, int N, double beta, double *C, int LDC){
    int i, j;
    for (i = 0; i < M; i++){
        for (j = 0; j < N; j++){
            C(i,j) *= beta; 
        }
    }
}

void pzydgemm_cpu_opt_k11(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    int i, j, k;
    if (beta != 1.0) pzyscale_C_k11(M, N, beta, C, LDC);
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

// test and debug 
// printf("ax0[0] = %5.2f, ay0[0] = %5.2f, az0[0] = %5.2f\n", ax0[0], ay0[0], az0[0]);
// printf("b00[0] = %5.2f, b01[0] = %5.2f\n", b00[0], b01[0]);
// printf("cx0[0] = %5.2f, cy0[0] = %5.2f, cz0[0] = %5.2f\n", cx0[0], cy0[0], cz0[0]);

// changed compared with kernel 10
#define KERNEL_K1_24x8_avx512_intrinsics_packing_v2\
    ax0 = _mm512_mul_pd(valpha, _mm512_loadu_pd(ptr_packing_a));\
    ay0 = _mm512_mul_pd(valpha, _mm512_loadu_pd(ptr_packing_a + 8));\
    az0 = _mm512_mul_pd(valpha, _mm512_loadu_pd(ptr_packing_a + 16));\
    b00 = _mm512_set1_pd(*ptr_packing_b0);\
    b01 = _mm512_set1_pd(*(ptr_packing_b0 + 1));\
    cx0 = _mm512_fmadd_pd(ax0,b00,cx0);\
    cx1 = _mm512_fmadd_pd(ax0,b01,cx1);\
    cy0 = _mm512_fmadd_pd(ay0,b00,cy0);\
    cy1 = _mm512_fmadd_pd(ay0,b01,cy1);\
    cz0 = _mm512_fmadd_pd(az0,b00,cz0);\
    cz1 = _mm512_fmadd_pd(az0,b01,cz1);\
    b00 = _mm512_set1_pd(*(ptr_packing_b1));\
    b01 = _mm512_set1_pd(*(ptr_packing_b1 + 1));\
    cx2 = _mm512_fmadd_pd(ax0,b00,cx2);\
    cx3 = _mm512_fmadd_pd(ax0,b01,cx3);\
    cy2 = _mm512_fmadd_pd(ay0,b00,cy2);\
    cy3 = _mm512_fmadd_pd(ay0,b01,cy3);\
    cz2 = _mm512_fmadd_pd(az0,b00,cz2);\
    cz3 = _mm512_fmadd_pd(az0,b01,cz3);\
    b00 = _mm512_set1_pd(*(ptr_packing_b2));\
    b01 = _mm512_set1_pd(*(ptr_packing_b2 + 1));\
    cx4 = _mm512_fmadd_pd(ax0,b00,cx4);\
    cx5 = _mm512_fmadd_pd(ax0,b01,cx5);\
    cy4 = _mm512_fmadd_pd(ay0,b00,cy4);\
    cy5 = _mm512_fmadd_pd(ay0,b01,cy5);\
    cz4 = _mm512_fmadd_pd(az0,b00,cz4);\
    cz5 = _mm512_fmadd_pd(az0,b01,cz5);\
    b00 = _mm512_set1_pd(*(ptr_packing_b3));\
    b01 = _mm512_set1_pd(*(ptr_packing_b3 + 1));\
    cx6 = _mm512_fmadd_pd(ax0,b00,cx6);\
    cx7 = _mm512_fmadd_pd(ax0,b01,cx7);\
    cy6 = _mm512_fmadd_pd(ay0,b00,cy6);\
    cy7 = _mm512_fmadd_pd(ay0,b01,cy7);\
    cz6 = _mm512_fmadd_pd(az0,b00,cz6);\
    cz7 = _mm512_fmadd_pd(az0,b01,cz7);\
    ptr_packing_a += 24;\
    ptr_packing_b0 += 2;\
    ptr_packing_b1 += 2;\
    ptr_packing_b2 += 2;\
    ptr_packing_b3 += 2;\
    k++;

// K4 is "k_inc & -4", is not a index, should add to inner_k_count: inner_k_count + K4

// test and debug
// printf("cx0[0] = %5.2f, cy0[0] = %5.2f, cz0[0] = %5.2f\n", cx0[0], cy0[0], cz0[0]);
#define macro_kernel_24xkx8_avx512_packing_v2\
    cx0 = _mm512_setzero_pd();\
    cx1 = _mm512_setzero_pd();\
    cx2 = _mm512_setzero_pd();\
    cx3 = _mm512_setzero_pd();\
    cx4 = _mm512_setzero_pd();\
    cx5 = _mm512_setzero_pd();\
    cx6 = _mm512_setzero_pd();\
    cx7 = _mm512_setzero_pd();\
    cy0 = _mm512_setzero_pd();\
    cy1 = _mm512_setzero_pd();\
    cy2 = _mm512_setzero_pd();\
    cy3 = _mm512_setzero_pd();\
    cy4 = _mm512_setzero_pd();\
    cy5 = _mm512_setzero_pd();\
    cy6 = _mm512_setzero_pd();\
    cy7 = _mm512_setzero_pd();\
    cz0 = _mm512_setzero_pd();\
    cz1 = _mm512_setzero_pd();\
    cz2 = _mm512_setzero_pd();\
    cz3 = _mm512_setzero_pd();\
    cz4 = _mm512_setzero_pd();\
    cz5 = _mm512_setzero_pd();\
    cz6 = _mm512_setzero_pd();\
    cz7 = _mm512_setzero_pd();\
    for (k = inner_k_count; k < inner_k_count + K4;){\
        KERNEL_K1_24x8_avx512_intrinsics_packing_v2\
        KERNEL_K1_24x8_avx512_intrinsics_packing_v2\
        KERNEL_K1_24x8_avx512_intrinsics_packing_v2\
        KERNEL_K1_24x8_avx512_intrinsics_packing_v2\
    }\
    for (k = inner_k_count + K4; k < inner_k_end;){\
        KERNEL_K1_24x8_avx512_intrinsics_packing_v2\
    }\
    _mm512_storeu_pd(&C(i,j), _mm512_add_pd(cx0, _mm512_loadu_pd(&C(i,j))));\
    _mm512_storeu_pd(&C(i,j+1), _mm512_add_pd(cx1, _mm512_loadu_pd(&C(i,j+1))));\
    _mm512_storeu_pd(&C(i,j+2), _mm512_add_pd(cx2, _mm512_loadu_pd(&C(i,j+2))));\
    _mm512_storeu_pd(&C(i,j+3), _mm512_add_pd(cx3, _mm512_loadu_pd(&C(i,j+3))));\
    _mm512_storeu_pd(&C(i,j+4), _mm512_add_pd(cx4, _mm512_loadu_pd(&C(i,j+4))));\
    _mm512_storeu_pd(&C(i,j+5), _mm512_add_pd(cx5, _mm512_loadu_pd(&C(i,j+5))));\
    _mm512_storeu_pd(&C(i,j+6), _mm512_add_pd(cx6, _mm512_loadu_pd(&C(i,j+6))));\
    _mm512_storeu_pd(&C(i,j+7), _mm512_add_pd(cx7, _mm512_loadu_pd(&C(i,j+7))));\
    _mm512_storeu_pd(&C(i+8,j), _mm512_add_pd(cy0, _mm512_loadu_pd(&C(i+8,j))));\
    _mm512_storeu_pd(&C(i+8,j+1), _mm512_add_pd(cy1, _mm512_loadu_pd(&C(i+8,j+1))));\
    _mm512_storeu_pd(&C(i+8,j+2), _mm512_add_pd(cy2, _mm512_loadu_pd(&C(i+8,j+2))));\
    _mm512_storeu_pd(&C(i+8,j+3), _mm512_add_pd(cy3, _mm512_loadu_pd(&C(i+8,j+3))));\
    _mm512_storeu_pd(&C(i+8,j+4), _mm512_add_pd(cy4, _mm512_loadu_pd(&C(i+8,j+4))));\
    _mm512_storeu_pd(&C(i+8,j+5), _mm512_add_pd(cy5, _mm512_loadu_pd(&C(i+8,j+5))));\
    _mm512_storeu_pd(&C(i+8,j+6), _mm512_add_pd(cy6, _mm512_loadu_pd(&C(i+8,j+6))));\
    _mm512_storeu_pd(&C(i+8,j+7), _mm512_add_pd(cy7, _mm512_loadu_pd(&C(i+8,j+7))));\
    _mm512_storeu_pd(&C(i+16,j), _mm512_add_pd(cz0, _mm512_loadu_pd(&C(i+16,j))));\
    _mm512_storeu_pd(&C(i+16,j+1), _mm512_add_pd(cz1, _mm512_loadu_pd(&C(i+16,j+1))));\
    _mm512_storeu_pd(&C(i+16,j+2), _mm512_add_pd(cz2, _mm512_loadu_pd(&C(i+16,j+2))));\
    _mm512_storeu_pd(&C(i+16,j+3), _mm512_add_pd(cz3, _mm512_loadu_pd(&C(i+16,j+3))));\
    _mm512_storeu_pd(&C(i+16,j+4), _mm512_add_pd(cz4, _mm512_loadu_pd(&C(i+16,j+4))));\
    _mm512_storeu_pd(&C(i+16,j+5), _mm512_add_pd(cz5, _mm512_loadu_pd(&C(i+16,j+5))));\
    _mm512_storeu_pd(&C(i+16,j+6), _mm512_add_pd(cz6, _mm512_loadu_pd(&C(i+16,j+6))));\
    _mm512_storeu_pd(&C(i+16,j+7), _mm512_add_pd(cz7, _mm512_loadu_pd(&C(i+16,j+7))));

// changed compared with kernel 10
// changed from 8 to 2 (compared to kernel 10)
void pzypacking_b_k11(double *packsrc, double *packdst, int LDB, int dim_k, int dim_n){
    // kernel A * B is 24 * 8, thus we need 8 pointers to store each number in one row (col-major)
    // but here, we only use 2 pointers to store
    double *src1, *src2, *dst;
    dst = packdst;
    int k, n;
    for (n = 0; n < dim_n; n += 2){
        src1 = packsrc + n * LDB;
        src2 = src1 + LDB;
        for (k = 0; k < dim_k; k++){
            *dst = *src1; src1++; dst++;
            *dst = *src2; src2++; dst++;
        }
    }
}

void pzypacking_a_k11(double *packsrc, double *packdst, int LDA, int dim_m, int dim_k){
    // kernel A * B is 24 * 8, we can use one pointer to store 24 numbers in one column (col-major)
    double *src, *dst;
    dst = packdst;
    //int i, k;
    //for (i = 0; i < dim_m, i += 24){
    int i, k, remain = dim_m;
    for (i = 0; remain > 23; remain -= 24, i += 24){
        src = packsrc + i;
        for (k = 0; k < dim_k; k++){
            _mm512_store_pd(dst, _mm512_loadu_pd(src));
            // we need to store the following 8 number to the following place,
            // so, the destination shoud also +8: dst + 8; dst + 16
            _mm512_store_pd(dst + 8, _mm512_loadu_pd(src + 8));
            _mm512_store_pd(dst + 16, _mm512_loadu_pd(src + 16));
            src += LDA;
            dst += 24;
        }
    }
}

// changed compared with kernel 10
void pzydgemm_cpu_v11(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    if (beta != 1.0) pzyscale_C_k11(M, N, beta, C, LDC);
    // difference between malloc and aligned_alloc : 
    // https://stackoverflow.com/questions/39677063/difference-between-aligned-malloc-and-standard-malloc
    
    // usage of aligned_alloc: (the address is aligned)
    // https://en.cppreference.com/w/c/memory/aligned_alloc
    // https://zhuanlan.zhihu.com/p/111780698
    // it says that "K_BLOCKING * N_BLOCKING * sizeof(double)" or "K_BLOCKING * M_BLOCKING * sizeof(double)" should be an integral multiple of 4096 here
    double *b_buffer = (double *)aligned_alloc(4096,K_BLOCKING*N_BLOCKING*sizeof(double));
    double *a_buffer = (double *)aligned_alloc(4096,K_BLOCKING*M_BLOCKING*sizeof(double));
    double *ptr_packing_a, *ptr_packing_b0, *ptr_packing_b1, *ptr_packing_b2, *ptr_packing_b3;
    __m512d valpha = _mm512_set1_pd(alpha); // broadcast alpha to a 512-bit vector, the input is a double value
    __m512d ax0, ay0, az0, b00, b01;
    __m512d cx0, cx1, cx2, cx3, cx4, cx5, cx6, cx7, cy0, cy1, cy2, cy3, cy4, cy5, cy6, cy7, cz0, cz1, cz2, cz3, cz4, cz5, cz6, cz7;

    int n_count = 0, k_count = 0, m_count = 0;
    int n_inc = 0, k_inc = 0, m_inc = 0;

    // get an integer which is divisible by 24, 8, 4(eg: 48 divided by 24 is 2)
    int M24 = 0, N8 = 0; // 24 * 8 kernel
    int K4 = 0; // loop unrolling 4 times
    int i = 0, j = 0, k = 0;
    int inner_m_count = 0, inner_n_count = 0, inner_k_count = 0, inner_k_end = 0;

    for (n_count = 0; n_count < N; n_count += n_inc){
        n_inc = (N - n_count > N_BLOCKING) ? N_BLOCKING : N - n_count;

        for (k_count = 0; k_count < K; k_count += k_inc){
                k_inc = K - k_count > K_BLOCKING ? K_BLOCKING : K - k_count;
                //printf("k_inc = %d\n", k_inc);
                //printf("LDB = %d\n", LDB);

                pzypacking_b_k11(B + k_count + n_count * LDB, b_buffer, LDB, k_inc, n_inc);
                /*
                // test the output of b_buffer
                double *testbuffer0, *testbuffer1, *testbuffer2,*testbuffer3;
                testbuffer0 = b_buffer;
                testbuffer1 = b_buffer + 2 * k_inc;
                testbuffer2 = b_buffer + 4 * k_inc;
                testbuffer3 = b_buffer + 6 * k_inc;
                printf("b_buffer:\n");
                for (int b = 0; b < k_inc; b++){
                    printf("%5.2f ", *testbuffer0);
                    testbuffer0++;
                    printf("%5.2f ", *testbuffer0);
                    testbuffer0++; 
                    printf("%5.2f ", *testbuffer1);
                    testbuffer1++;
                    printf("%5.2f ", *testbuffer1);
                    testbuffer1++;
                    printf("%5.2f ", *testbuffer2);
                    testbuffer2++;
                    printf("%5.2f ", *testbuffer2);
                    testbuffer2++;
                    printf("%5.2f ", *testbuffer3);
                    testbuffer3++;
                    printf("%5.2f ", *testbuffer3);
                    testbuffer3++;
                    printf("\n"); 
                }
                printf("\n");
                //*/
            for (m_count = 0; m_count < M; m_count += m_inc){
                //printf("m_count = %d\n", m_count);

                m_inc = M - m_count > M_BLOCKING ? M_BLOCKING : M - m_count;
                pzypacking_a_k11(A + m_count + k_count * LDA, a_buffer, LDA, m_inc, k_inc);

                N8 = n_inc & -8;
                //M24 = m_inc & -24; // this way only for 2^x, 24 is not.
                M24 = (m_inc / 24) * 24;
                K4 = k_inc & -4;

                //printf("M24 = %d, N8 = %d, K4 = %d\n", M24, N8, K4);
                for (inner_m_count = 0; inner_m_count < M24; inner_m_count += 24){
                    //printf("inner_m_count = %d\n", inner_m_count);

                    i = m_count + inner_m_count;
                    //printf("i = %d\n", i);

                    // each time for specific i, j loop means to compute all 24*8 matrix in same row on matrix C.
                    for (inner_n_count = 0; inner_n_count < N8; inner_n_count += 8){ 
                        //printf("inner_n_count = %d\n", inner_n_count);

                        j = n_count + inner_n_count;
                        //printf("j = %d\n", j);

                        inner_k_count = k_count;
                        //printf("inner_k_count = %d\n", inner_k_count);

                        inner_k_end = inner_k_count + k_inc;
                        //printf("inner_k_end = %d\n", inner_k_end);

                        ptr_packing_a = a_buffer + inner_m_count * k_inc;
                        ptr_packing_b0 = b_buffer + k_inc * inner_n_count;
                        /*
                        // in new version from Yujia, it used K, but I think it's wrong
                        // because buffer is used for each block, not the whole matrix, so it should not use K, it should use k_inc
                        ptr_packing_b1 = ptr_packing_b0 + K * 2;
                        ptr_packing_b2 = ptr_packing_b1 + K * 2;
                        ptr_packing_b3 = ptr_packing_b2 + K * 2;
                        */
                        ptr_packing_b1 = ptr_packing_b0 + k_inc * 2;
                        ptr_packing_b2 = ptr_packing_b1 + k_inc * 2;
                        ptr_packing_b3 = ptr_packing_b2 + k_inc * 2;

                        /*
                        // test the output of a_buffer
                        double *testbuffer;
                        testbuffer = ptr_packing_a;
                        for (int b = 0; b < m_inc * k_inc; b++){
                            printf("%5.2f ", *testbuffer);
                            testbuffer++;   
                        }
                        printf("\n");
                        */
                        /*
                        // test the output of b_buffer
                        double *testbuffer;
                        int tmp = 0;
                        testbuffer = ptr_packing_b0;
                        printf("buffer part ptr_packing_b0:\n");
                        for (int b = 0; b < 2 * k_inc; b++){
                            printf("%5.2f ", *testbuffer);
                            testbuffer++; 
                            tmp++;
                            if (tmp > 1 && tmp % 2 == 0){
                                printf("\n");
                            }  
                        }
                        printf("\n");

                        tmp = 0;
                        testbuffer = ptr_packing_b1;
                        printf("buffer part ptr_packing_b1:\n");
                        for (int b = 0; b < 2 * k_inc; b++){
                            printf("%5.2f ", *testbuffer);
                            testbuffer++; 
                            tmp++;
                            if (tmp > 1 && tmp % 2 == 0){
                                printf("\n");
                            }  
                        }
                        printf("\n");

                        tmp = 0;
                        testbuffer = ptr_packing_b2;
                        printf("buffer part ptr_packing_b2:\n");
                        for (int b = 0; b < 2 * k_inc; b++){
                            printf("%5.2f ", *testbuffer);
                            testbuffer++; 
                            tmp++;
                            if (tmp > 1 && tmp % 2 == 0){
                                printf("\n");
                            }  
                        }
                        printf("\n");

                        tmp = 0;
                        testbuffer = ptr_packing_b3;
                        printf("buffer part ptr_packing_b3:\n");
                        for (int b = 0; b < 2 * k_inc; b++){
                            printf("%5.2f ", *testbuffer);
                            testbuffer++; 
                            tmp++;
                            if (tmp > 1 && tmp % 2 == 0){
                                printf("\n");
                            }  
                        }
                        printf("\n");
                        //*/

                        // add register reuse on C(i,j) to reduce the times of accessing C(i,j) in memory
                        macro_kernel_24xkx8_avx512_packing_v2
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

                //printf("M24 = %d, m_inc = %d, N8 = %d, n_inc = %d\n", M24, m_inc, N8, n_inc);
                
                // here is not the end of the function, so shouldn't use "return"
                //if (M24 == m_inc && N8 == n_inc) return; 
                
                // Attention! I didn't check the edge case, so the following codes are not sure for correctness
                // boundary conditions
                if (M24 != m_inc) {
                    printf("enter edge case for m_inc. \n");
                    pzydgemm_cpu_opt_k11(m_inc - M24, n_inc, k_inc, alpha, &A(M24, 0), LDA, B, LDB, 1.0, &C(M24, 0), LDC); // A+M24 move to M24 row, because it's column major
                }
                if (N8 != n_inc) {
                    printf("enter edge case for n_inc. \n");
                    pzydgemm_cpu_opt_k11(M24, n_inc - N8, k_inc, alpha, A, LDA, &B(0, N8), LDB, 1.0, &C(0, N8), LDC);
                }
            }
            //printf("m_count = %d\n", m_count);

            // edge case for M
            if (m_inc != M_BLOCKING && m_count != 0){
                printf("enter edge case for M. \n");
                pzydgemm_cpu_opt_k11(m_inc, N_BLOCKING, K, alpha, &A(m_count, 0), LDA, &B(0, n_count), LDB, 1.0, &C(m_count, n_count), LDC);
            }

        }

        // edge case for N
        if (n_inc != N_BLOCKING && n_count != 0){
            printf("enter edge case for N. \n");
            pzydgemm_cpu_opt_k11(M, n_inc, K, alpha, A, LDA, &B(0, n_count), LDB, 1.0, &C(0, n_count), LDC);
        }
    }
    free(a_buffer);free(b_buffer);
}