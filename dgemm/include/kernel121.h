#include "immintrin.h"
#define A(i,j) A[(i)+(j)*LDA]
#define B(i,j) B[(i)+(j)*LDB]
#define C(i,j) C[(i)+(j)*LDC]
// inline assembling + discontinuous packing + cache blocking + 16 * 14 register blocking with AVX512 + loop unrolling * 4
// The least number of AVX512 register needed is 28 + 2 + 1 = 31 (max is 32)

/*
#define M_BLOCKING 192
#define N_BLOCKING 96 // very big 2048
#define K_BLOCKING 384
//*/

///*
#define M_BLOCKING 192
//#define N_BLOCKING 112 // 2240
#define N_BLOCKING 168 // 3360
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

// C := alpha * op(A) * op(B) + beta * C
void pzyscale_C_k121(int M, int N, double beta, double *C, int LDC){
    int i, j;
    for (i = 0; i < M; i++){
        for (j = 0; j < N; j++){
            C(i,j) *= beta; 
        }
    }
}

void pzydgemm_cpu_opt_k121(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    int i, j, k;
    if (beta != 1.0) pzyscale_C_k121(M, N, beta, C, LDC);
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

void pzypacking_b_k121(double *packsrc, double *packdst, int LDB, int dim_k, int dim_n){
    // kernel A * B is 16 * 14, thus we need 14 pointers to store each number in one row (col-major)
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

// changed vs kernel 111
void pzypacking_a_k121(double alpha, double *packsrc, double *packdst, int LDA, int dim_m, int dim_k){
    // kernel A * B is 16 * 14, we can use one pointer to store 16 numbers in one column (col-major)
    double *src, *dst;
    dst = packdst;
    __m512d valpha=_mm512_set1_pd(alpha); // broadcast alpha to a 512-bit vector, the input is a double value
    //int i, k;
    //for (i = 0; i < dim_m, i += 16){
    int i, k, remain = dim_m;
    for (i = 0; remain > 15; remain -= 16, i += 16){
        src = packsrc + i;
        for (k = 0; k < dim_k; k++){
            _mm512_store_pd(dst, _mm512_mul_pd(_mm512_loadu_pd(src), valpha));
            // we need to store the following 8 number to the following place,
            // so, the destination shoud also +8: dst + 8
            _mm512_store_pd(dst + 8, _mm512_mul_pd(_mm512_loadu_pd(src + 8), valpha));
            src += LDA;
            dst += 16;
        }
    }
}

// changed vs kernel 111
#define INIT_m16n14 \
  "vpxorq %%zmm4,%%zmm4,%%zmm4; vpxorq %%zmm5,%%zmm5,%%zmm5; vpxorq %%zmm6,%%zmm6,%%zmm6; vpxorq %%zmm7,%%zmm7,%%zmm7;"\
  "vpxorq %%zmm8,%%zmm8,%%zmm8; vpxorq %%zmm9,%%zmm9,%%zmm9; vpxorq %%zmm10,%%zmm10,%%zmm10; vpxorq %%zmm11,%%zmm11,%%zmm11;"\
  "vpxorq %%zmm12,%%zmm12,%%zmm12; vpxorq %%zmm13,%%zmm13,%%zmm13; vpxorq %%zmm14,%%zmm14,%%zmm14; vpxorq %%zmm15,%%zmm15,%%zmm15;"\
  "vpxorq %%zmm16,%%zmm16,%%zmm16; vpxorq %%zmm17,%%zmm17,%%zmm17; vpxorq %%zmm18,%%zmm18,%%zmm18; vpxorq %%zmm19,%%zmm19,%%zmm19;"\
  "vpxorq %%zmm20,%%zmm20,%%zmm20; vpxorq %%zmm21,%%zmm21,%%zmm21; vpxorq %%zmm22,%%zmm22,%%zmm22; vpxorq %%zmm23,%%zmm23,%%zmm23;"\
  "vpxorq %%zmm24,%%zmm24,%%zmm24; vpxorq %%zmm25,%%zmm25,%%zmm25; vpxorq %%zmm26,%%zmm26,%%zmm26; vpxorq %%zmm27,%%zmm27,%%zmm27;"\
  "vpxorq %%zmm28,%%zmm28,%%zmm28; vpxorq %%zmm29,%%zmm29,%%zmm29; vpxorq %%zmm30,%%zmm30,%%zmm30; vpxorq %%zmm31,%%zmm31,%%zmm31;"

// zmm0 - zmm1 = ax0, ay0
#define load_a_m16 \
  "vmovaps (%0),%%zmm0; vmovaps 64(%0),%%zmm1; addq $128,%0;"
// zmm2, zmm3 = b00, b01
// zmm4 - zmm17 = cx0 - cx13
// zmm18 - zmm31 = cy0 - cy13
#define kernel_m16n2_1 \
  "vbroadcastsd (%1),%%zmm2; vbroadcastsd 8(%1),%%zmm3;"\
  "vfmadd231pd %%zmm0,%%zmm2,%%zmm4; vfmadd231pd %%zmm0,%%zmm3,%%zmm5;"\
  "vfmadd231pd %%zmm1,%%zmm2,%%zmm18; vfmadd231pd %%zmm1,%%zmm3,%%zmm19;"
#define kernel_m16n2_2 \
  "vbroadcastsd (%1,%%r11,1),%%zmm2; vbroadcastsd 8(%1,%%r11,1),%%zmm3;"\
  "vfmadd231pd %%zmm0,%%zmm2,%%zmm6; vfmadd231pd %%zmm0,%%zmm3,%%zmm7;"\
  "vfmadd231pd %%zmm1,%%zmm2,%%zmm20; vfmadd231pd %%zmm1,%%zmm3,%%zmm21;"
#define kernel_m16n2_3 \
  "vbroadcastsd (%1,%%r11,2),%%zmm2; vbroadcastsd 8(%1,%%r11,2),%%zmm3;"\
  "vfmadd231pd %%zmm0,%%zmm2,%%zmm8; vfmadd231pd %%zmm0,%%zmm3,%%zmm9;"\
  "vfmadd231pd %%zmm1,%%zmm2,%%zmm22; vfmadd231pd %%zmm1,%%zmm3,%%zmm23;"
#define kernel_m16n2_4 \
  "vbroadcastsd (%%r12),%%zmm2; vbroadcastsd 8(%%r12),%%zmm3;"\
  "vfmadd231pd %%zmm0,%%zmm2,%%zmm10; vfmadd231pd %%zmm0,%%zmm3,%%zmm11;"\
  "vfmadd231pd %%zmm1,%%zmm2,%%zmm24; vfmadd231pd %%zmm1,%%zmm3,%%zmm25;"
#define kernel_m16n2_5 \
  "vbroadcastsd (%%r14),%%zmm2; vbroadcastsd 8(%%r14),%%zmm3;"\
  "vfmadd231pd %%zmm0,%%zmm2,%%zmm12; vfmadd231pd %%zmm0,%%zmm3,%%zmm13;"\
  "vfmadd231pd %%zmm1,%%zmm2,%%zmm26; vfmadd231pd %%zmm1,%%zmm3,%%zmm27;"
#define kernel_m16n2_6 \
  "vbroadcastsd (%%r14,%%r11,1),%%zmm2; vbroadcastsd 8(%%r14,%%r11,1),%%zmm3;"\
  "vfmadd231pd %%zmm0,%%zmm2,%%zmm14; vfmadd231pd %%zmm0,%%zmm3,%%zmm15;"\
  "vfmadd231pd %%zmm1,%%zmm2,%%zmm28; vfmadd231pd %%zmm1,%%zmm3,%%zmm29;"
#define kernel_m16n2_7 \
  "vbroadcastsd (%%r14,%%r11,2),%%zmm2; vbroadcastsd 8(%%r14,%%r11,2),%%zmm3;"\
  "vfmadd231pd %%zmm0,%%zmm2,%%zmm16; vfmadd231pd %%zmm0,%%zmm3,%%zmm17;"\
  "vfmadd231pd %%zmm1,%%zmm2,%%zmm30; vfmadd231pd %%zmm1,%%zmm3,%%zmm31;"
#define KERNEL_m16n14 \
  load_a_m16 \
  kernel_m16n2_1 \
  kernel_m16n2_2 \
  kernel_m16n2_3 \
  kernel_m16n2_4 \
  kernel_m16n2_5 \
  kernel_m16n2_6 \
  kernel_m16n2_7 \
  "addq $16,%1; addq $16,%%r12; addq $16,%%r14;"

#define save_m16n2_1 \
  "vaddpd (%2),%%zmm4,%%zmm4; vmovups %%zmm4,(%2);"\
  "vaddpd 64(%2),%%zmm18,%%zmm18; vmovups %%zmm18,64(%2);"\
  "vaddpd (%2,%3,1),%%zmm5,%%zmm5; vmovups %%zmm5,(%2,%3,1);"\
  "vaddpd 64(%2,%3,1),%%zmm19,%%zmm19; vmovups %%zmm19,64(%2,%3,1);"\
  "leaq (%2,%3,2),%2;"
#define save_m16n2_2 \
  "vaddpd (%2),%%zmm6,%%zmm6; vmovups %%zmm6,(%2);"\
  "vaddpd 64(%2),%%zmm20,%%zmm20; vmovups %%zmm20,64(%2);"\
  "vaddpd (%2,%3,1),%%zmm7,%%zmm7; vmovups %%zmm7,(%2,%3,1);"\
  "vaddpd 64(%2,%3,1),%%zmm21,%%zmm21; vmovups %%zmm21,64(%2,%3,1);"\
  "leaq (%2,%3,2),%2;"
#define save_m16n2_3 \
  "vaddpd (%2),%%zmm8,%%zmm8; vmovups %%zmm8,(%2);"\
  "vaddpd 64(%2),%%zmm22,%%zmm22; vmovups %%zmm22,64(%2);"\
  "vaddpd (%2,%3,1),%%zmm9,%%zmm9; vmovups %%zmm9,(%2,%3,1);"\
  "vaddpd 64(%2,%3,1),%%zmm23,%%zmm23; vmovups %%zmm23,64(%2,%3,1);"\
  "leaq (%2,%3,2),%2;"
#define save_m16n2_4 \
  "vaddpd (%2),%%zmm10,%%zmm10; vmovups %%zmm10,(%2);"\
  "vaddpd 64(%2),%%zmm24,%%zmm24; vmovups %%zmm24,64(%2);"\
  "vaddpd (%2,%3,1),%%zmm11,%%zmm11; vmovups %%zmm11,(%2,%3,1);"\
  "vaddpd 64(%2,%3,1),%%zmm25,%%zmm25; vmovups %%zmm25,64(%2,%3,1);"\
  "leaq (%2,%3,2),%2;"
#define save_m16n2_5 \
  "vaddpd (%2),%%zmm12,%%zmm12; vmovups %%zmm12,(%2);"\
  "vaddpd 64(%2),%%zmm26,%%zmm26; vmovups %%zmm26,64(%2);"\
  "vaddpd (%2,%3,1),%%zmm13,%%zmm13; vmovups %%zmm13,(%2,%3,1);"\
  "vaddpd 64(%2,%3,1),%%zmm27,%%zmm27; vmovups %%zmm27,64(%2,%3,1);"\
  "leaq (%2,%3,2),%2;"
#define save_m16n2_6 \
  "vaddpd (%2),%%zmm14,%%zmm14; vmovups %%zmm14,(%2);"\
  "vaddpd 64(%2),%%zmm28,%%zmm28; vmovups %%zmm28,64(%2);"\
  "vaddpd (%2,%3,1),%%zmm15,%%zmm15; vmovups %%zmm15,(%2,%3,1);"\
  "vaddpd 64(%2,%3,1),%%zmm29,%%zmm29; vmovups %%zmm29,64(%2,%3,1);"\
  "leaq (%2,%3,2),%2;"
#define save_m16n2_7 \
  "vaddpd (%2),%%zmm16,%%zmm16; vmovups %%zmm16,(%2);"\
  "vaddpd 64(%2),%%zmm30,%%zmm30; vmovups %%zmm30,64(%2);"\
  "vaddpd (%2,%3,1),%%zmm17,%%zmm17; vmovups %%zmm17,(%2,%3,1);"\
  "vaddpd 64(%2,%3,1),%%zmm31,%%zmm31; vmovups %%zmm31,64(%2,%3,1);"
#define SAVE_m16n14 \
  save_m16n2_1 \
  save_m16n2_2 \
  save_m16n2_3 \
  save_m16n2_4 \
  save_m16n2_5 \
  save_m16n2_6 \
  save_m16n2_7

void macro_kernel_16xkx14_avx512_packing_inlineASM(double *ptr_packing_a, double *ptr_packing_b, double *ptr_c, int64_t K_inc, int64_t LDC_in_bytes){
    __asm__ __volatile__(
        // salq: shift, salq $4,%%r11 => r11 = r11 * 16
        // leaq : "lea" is load effective address, it's not loading value from that address 
        // https://courses.cs.washington.edu/courses/cse351/17wi/lectures/CSE351-L09-x86-II_17wi.pdf
        // salq : 'q' means "quadword"
        "movq %4,%%r11;\n\t"
        "salq $4,%%r11;leaq (%1,%%r11,2),%%r12;addq %%r11,%%r12;movq %4,%%r13;\n\t"
        "leaq (%1,%%r11,4),%%r14; \n\t" // b_ptr + 10 * K
        INIT_m16n14 \
        "cmpq $4,%%r13;jb 7167142f;\n\t" // (only numbers)several numbers with random length
        "7167141:\n\t" // main kernel loop
        KERNEL_m16n14 \
        KERNEL_m16n14 \
        KERNEL_m16n14 \
        KERNEL_m16n14 \
        "subq $4,%%r13;cmpq $4,%%r13;jnb 7167141b;\n\t" // not below: r13 >= 4
        "cmpq $0,%%r13;je 7167143f;\n\t"
        "7167142:\n\t"
        KERNEL_m16n14 \
        "decq %%r13;testq %%r13,%%r13;jnz 7167142b;\n\t" // it skips only when r13 = 0, r13 & r13 = 0
        "7167143:\n\t"
        SAVE_m16n14 \
        // "+" means we will modify the value in this register, tell the compiler to not optimize it
        :"+r"(ptr_packing_a), // %0
         "+r"(ptr_packing_b), // %1
         "+r"(ptr_c),         // %2
         "+r"(LDC_in_bytes)   // %3
        :"m"(K_inc)           // %4
        :"r11","r12","r13","r14","cc","memory"
    );
}

// changed vs kernel 101
void pzydgemm_cpu_v121(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    if (beta != 1.0) pzyscale_C_k121(M, N, beta, C, LDC);
    // difference between malloc and aligned_alloc : 
    // https://stackoverflow.com/questions/39677063/difference-between-aligned-malloc-and-standard-malloc
    
    // usage of aligned_alloc: (the address is aligned)
    // https://en.cppreference.com/w/c/memory/aligned_alloc
    // https://zhuanlan.zhihu.com/p/111780698
    // it says that "K_BLOCKING * N_BLOCKING * sizeof(double)" or "K_BLOCKING * M_BLOCKING * sizeof(double)" should be an integral multiple of 4096 here
    double *b_buffer = (double *)aligned_alloc(4096,K_BLOCKING*N_BLOCKING*sizeof(double));
    double *a_buffer = (double *)aligned_alloc(4096,K_BLOCKING*M_BLOCKING*sizeof(double));
    double *ptr_packing_a, *ptr_packing_b, *ptr_c;

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
                pzypacking_b_k121(B + k_count + n_count * LDB, b_buffer, LDB, k_inc, n_inc);

            for (m_count = 0; m_count < M; m_count += m_inc){
                //printf("m_count = %d\n", m_count);
                m_inc = M - m_count > M_BLOCKING ? M_BLOCKING : M - m_count;
                pzypacking_a_k121(alpha, A + m_count + k_count * LDA, a_buffer, LDA, m_inc, k_inc);

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
                        ptr_c = C + i + j * LDC;
                        int64_t K_inc = (int64_t)k_inc, LDC_in_bytes = (int64_t)LDC * sizeof(double);

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
                        macro_kernel_16xkx14_avx512_packing_inlineASM(ptr_packing_a, ptr_packing_b, ptr_c, K_inc, LDC_in_bytes);
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
                    pzydgemm_cpu_opt_k121(m_inc - M16, n_inc, k_inc, alpha, &A(M16, 0), LDA, B, LDB, 1.0, &C(M16, 0), LDC); // A+M16 move to M16 row, because it's column major
                }
                if (N14 != n_inc) {
                    printf("enter edge case for n_inc. \n");
                    pzydgemm_cpu_opt_k121(M16, n_inc - N14, k_inc, alpha, A, LDA, &B(0, N14), LDB, 1.0, &C(0, N14), LDC);
                }
            }
            //printf("m_count = %d\n", m_count);

            // edge case for M
            if (m_inc != M_BLOCKING && m_count != 0){
                printf("enter edge case for M. \n");
                pzydgemm_cpu_opt_k121(m_inc, N_BLOCKING, K, alpha, &A(m_count, 0), LDA, &B(0, n_count), LDB, 1.0, &C(m_count, n_count), LDC);
            }

        }

        // edge case for N
        if (n_inc != N_BLOCKING && n_count != 0){
            printf("enter edge case for N. \n");
            pzydgemm_cpu_opt_k121(M, n_inc, K, alpha, A, LDA, &B(0, n_count), LDB, 1.0, &C(0, n_count), LDC);
        }
    }
    free(a_buffer);free(b_buffer);
}