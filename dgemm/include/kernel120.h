#include "immintrin.h"
#include <stdint.h> // for int64_t
#define A(i,j) A[(i)+(j)*LDA]
#define B(i,j) B[(i)+(j)*LDB]
#define C(i,j) C[(i)+(j)*LDC]
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

// inline assembling + discontinuous packing(2 - 2 packing) + cache blocking + 24*8 register blocking with AVX512 + loop unrolling * 4
// The least number of AVX512 register needed is 24 + 3 +1 = 28 (max is 32)

// 8*4 register blocking, still can be improved by adding one 4*2 register blocking or 4*4

// C := alpha * op(A) * op(B) + beta * C
void pzyscale_C_k120(int M, int N, double beta, double *C, int LDC){
    int i, j;
    for (i = 0; i < M; i++){
        for (j = 0; j < N; j++){
            C(i,j) *= beta; 
        }
    }
}

void pzydgemm_cpu_opt_k120(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    int i, j, k;
    if (beta != 1.0) pzyscale_C_k120(M, N, beta, C, LDC);
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

void pzypacking_b_k120(double *packsrc, double *packdst, int LDB, int dim_k, int dim_n){
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

// changed vs kernel 11
void pzypacking_a_k120(double alpha, double *packsrc, double *packdst, int LDA, int dim_m, int dim_k){
    // kernel A * B is 24 * 8, we can use one pointer to store 24 numbers in one column (col-major)
    double *src, *dst;
    dst = packdst;
    __m512d valpha=_mm512_set1_pd(alpha); // broadcast alpha to a 512-bit vector, the input is a double value
    //int i, k;
    //for (i = 0; i < dim_m, i += 24){
    int i, k, remain = dim_m;
    for (i = 0; remain > 23; remain -= 24, i += 24){
        src = packsrc + i;
        for (k = 0; k < dim_k; k++){
            _mm512_store_pd(dst, _mm512_mul_pd(_mm512_loadu_pd(src), valpha));
            // we need to store the following 8 number to the following place,
            // so, the destination shoud also +8: dst + 8; dst + 16
            _mm512_store_pd(dst + 8, _mm512_mul_pd(_mm512_loadu_pd(src + 8), valpha));
            _mm512_store_pd(dst + 16, _mm512_mul_pd(_mm512_loadu_pd(src + 16), valpha));
            src += LDA;
            dst += 24;
        }
    }
}

// changed vs kernel 11
#define init_m24n1 \
  "vpxorq %%zmm8,%%zmm8,%%zmm8; vpxorq %%zmm9,%%zmm9,%%zmm9; vpxorq %%zmm10,%%zmm10,%%zmm10;"
#define init_m24n2 \
  init_m24n1 \
  "vpxorq %%zmm11,%%zmm11,%%zmm11; vpxorq %%zmm12,%%zmm12,%%zmm12; vpxorq %%zmm13,%%zmm13,%%zmm13;"
#define init_m24n4 \
  init_m24n2 \
  "vpxorq %%zmm14,%%zmm14,%%zmm14; vpxorq %%zmm15,%%zmm15,%%zmm15; vpxorq %%zmm16,%%zmm16,%%zmm16;"\
  "vpxorq %%zmm17,%%zmm17,%%zmm17; vpxorq %%zmm18,%%zmm18,%%zmm18; vpxorq %%zmm19,%%zmm19,%%zmm19;"
#define init_m24n6 \
  init_m24n4 \
  "vpxorq %%zmm20,%%zmm20,%%zmm20; vpxorq %%zmm21,%%zmm21,%%zmm21; vpxorq %%zmm22,%%zmm22,%%zmm22;"\
  "vpxorq %%zmm23,%%zmm23,%%zmm23; vpxorq %%zmm24,%%zmm24,%%zmm24; vpxorq %%zmm25,%%zmm25,%%zmm25;"
#define init_m24n8 \
  init_m24n6 \
  "vpxorq %%zmm26,%%zmm26,%%zmm26; vpxorq %%zmm27,%%zmm27,%%zmm27; vpxorq %%zmm28,%%zmm28,%%zmm28;"\
  "vpxorq %%zmm29,%%zmm29,%%zmm29; vpxorq %%zmm30,%%zmm30,%%zmm30; vpxorq %%zmm31,%%zmm31,%%zmm31;"

#define LOAD_A_COL_m24 \
  "vmovaps (%0),%%zmm1;vmovaps 64(%0),%%zmm2;vmovaps 128(%0),%%zmm3;addq $192,%0;"
#define kernel_m24n2_1 \
  "vbroadcastsd (%1),%%zmm4; vfmadd231pd %%zmm1,%%zmm4,%%zmm8; vfmadd231pd %%zmm2,%%zmm4,%%zmm9; vfmadd231pd %%zmm3,%%zmm4,%%zmm10;"\
  "vbroadcastsd 8(%1),%%zmm5; vfmadd231pd %%zmm1,%%zmm5,%%zmm11; vfmadd231pd %%zmm2,%%zmm5,%%zmm12; vfmadd231pd %%zmm3,%%zmm5,%%zmm13;"
#define kernel_m24n2_2 \
  "vbroadcastsd (%1,%%r11,1),%%zmm4; vfmadd231pd %%zmm1,%%zmm4,%%zmm14; vfmadd231pd %%zmm2,%%zmm4,%%zmm15; vfmadd231pd %%zmm3,%%zmm4,%%zmm16;"\
  "vbroadcastsd 8(%1,%%r11,1),%%zmm5; vfmadd231pd %%zmm1,%%zmm5,%%zmm17; vfmadd231pd %%zmm2,%%zmm5,%%zmm18; vfmadd231pd %%zmm3,%%zmm5,%%zmm19;"
#define kernel_m24n2_3 \
  "vbroadcastsd (%1,%%r11,2),%%zmm4; vfmadd231pd %%zmm1,%%zmm4,%%zmm20; vfmadd231pd %%zmm2,%%zmm4,%%zmm21; vfmadd231pd %%zmm3,%%zmm4,%%zmm22;"\
  "vbroadcastsd 8(%1,%%r11,2),%%zmm5; vfmadd231pd %%zmm1,%%zmm5,%%zmm23; vfmadd231pd %%zmm2,%%zmm5,%%zmm24; vfmadd231pd %%zmm3,%%zmm5,%%zmm25;"
#define kernel_m24n2_4 \
  "vbroadcastsd (%%r12),%%zmm4; vfmadd231pd %%zmm1,%%zmm4,%%zmm26; vfmadd231pd %%zmm2,%%zmm4,%%zmm27; vfmadd231pd %%zmm3,%%zmm4,%%zmm28;"\
  "vbroadcastsd 8(%%r12),%%zmm5; vfmadd231pd %%zmm1,%%zmm5,%%zmm29; vfmadd231pd %%zmm2,%%zmm5,%%zmm30; vfmadd231pd %%zmm3,%%zmm5,%%zmm31;"
#define KERNEL_m24n8 \
  LOAD_A_COL_m24 \
  kernel_m24n2_1 \
  kernel_m24n2_2 \
  kernel_m24n2_3 \
  kernel_m24n2_4 \
  "addq $16,%1;addq $16,%%r12;"

#define save_m24n2_1 \
  "vaddpd (%2),%%zmm8,%%zmm8;vaddpd 64(%2),%%zmm9,%%zmm9;vaddpd 128(%2),%%zmm10,%%zmm10;"\
  "vmovups %%zmm8,(%2);vmovups %%zmm9,64(%2);vmovups %%zmm10,128(%2);"\
  "vaddpd (%2,%3,1),%%zmm11,%%zmm11;vaddpd 64(%2,%3,1),%%zmm12,%%zmm12;vaddpd 128(%2,%3,1),%%zmm13,%%zmm13;"\
  "vmovups %%zmm11,(%2,%3,1);vmovups %%zmm12,64(%2,%3,1);vmovups %%zmm13,128(%2,%3,1);leaq (%2,%3,2),%2;"
#define save_m24n2_2 \
  "vaddpd (%2),%%zmm14,%%zmm14;vaddpd 64(%2),%%zmm15,%%zmm15;vaddpd 128(%2),%%zmm16,%%zmm16;"\
  "vmovups %%zmm14,(%2);vmovups %%zmm15,64(%2);vmovups %%zmm16,128(%2);"\
  "vaddpd (%2,%3,1),%%zmm17,%%zmm17;vaddpd 64(%2,%3,1),%%zmm18,%%zmm18;vaddpd 128(%2,%3,1),%%zmm19,%%zmm19;"\
  "vmovups %%zmm17,(%2,%3,1);vmovups %%zmm18,64(%2,%3,1);vmovups %%zmm19,128(%2,%3,1);leaq (%2,%3,2),%2;"
#define save_m24n2_3 \
  "vaddpd (%2),%%zmm20,%%zmm20;vaddpd 64(%2),%%zmm21,%%zmm21;vaddpd 128(%2),%%zmm22,%%zmm22;"\
  "vmovups %%zmm20,(%2);vmovups %%zmm21,64(%2);vmovups %%zmm22,128(%2);"\
  "vaddpd (%2,%3,1),%%zmm23,%%zmm23;vaddpd 64(%2,%3,1),%%zmm24,%%zmm24;vaddpd 128(%2,%3,1),%%zmm25,%%zmm25;"\
  "vmovups %%zmm23,(%2,%3,1);vmovups %%zmm24,64(%2,%3,1);vmovups %%zmm25,128(%2,%3,1);leaq (%2,%3,2),%2;"
#define save_m24n2_4 \
  "vaddpd (%2),%%zmm26,%%zmm26;vaddpd 64(%2),%%zmm27,%%zmm27;vaddpd 128(%2),%%zmm28,%%zmm28;"\
  "vmovups %%zmm26,(%2);vmovups %%zmm27,64(%2);vmovups %%zmm28,128(%2);"\
  "vaddpd (%2,%3,1),%%zmm29,%%zmm29;vaddpd 64(%2,%3,1),%%zmm30,%%zmm30;vaddpd 128(%2,%3,1),%%zmm31,%%zmm31;"\
  "vmovups %%zmm29,(%2,%3,1);vmovups %%zmm30,64(%2,%3,1);vmovups %%zmm31,128(%2,%3,1);leaq (%2,%3,2),%2;"
#define SAVE_m24n8 \
  save_m24n2_1 \
  save_m24n2_2 \
  save_m24n2_3 \
  save_m24n2_4

void macro_kernel_24xkx8_avx512_packing_inlineASM_k120(double *ptr_packing_a, double *ptr_packing_b, double *ptr_c, int64_t K_inc, int64_t LDC_in_bytes){
    __asm__ __volatile__(
        // salq: shift, salq $4,%%r11 => r11 = r11 * 16
        // leaq : "lea" is load effective address, it's not loading value from that address 
        // https://courses.cs.washington.edu/courses/cse351/17wi/lectures/CSE351-L09-x86-II_17wi.pdf
        // salq : 'q' means "quadword"
        "movq %4,%%r11;\n\t"
        "salq $4,%%r11;leaq (%1,%%r11,2),%%r12;addq %%r11,%%r12;movq %4,%%r13;\n\t"
        init_m24n8 \
        "cmpq $4,%%r13;jb 724782f;\n\t"
        "724781:\n\t" // main kernel loop
        KERNEL_m24n8 \
        KERNEL_m24n8 \
        KERNEL_m24n8 \
        KERNEL_m24n8 \
        "subq $4,%%r13;cmpq $4,%%r13;jnb 724781b;\n\t" // not below: r13 >= 4
        "cmpq $0,%%r13;je 724783f;\n\t"
        "724782:\n\t"
        KERNEL_m24n8 \
        "decq %%r13;testq %%r13,%%r13;jnz 724782b;\n\t" // it skips only when r13 = 0, r13 & r13 = 0
        "724783:\n\t"
        SAVE_m24n8 \
        // "+" means we will modify the value in this register, tell the compiler to not optimize it
        :"+r"(ptr_packing_a), // %0
         "+r"(ptr_packing_b), // %1
          "+r"(ptr_c),        // %2
          "+r"(LDC_in_bytes)  // %3
        :"m"(K_inc)           // %4
        :"r11","r12","r13","cc","memory"
    );
}

// changed vs kernel 11
void pzydgemm_cpu_v120(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    if (beta != 1.0) pzyscale_C_k120(M, N, beta, C, LDC);
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

                pzypacking_b_k120(B + k_count + n_count * LDB, b_buffer, LDB, k_inc, n_inc);

            for (m_count = 0; m_count < M; m_count += m_inc){
                //printf("m_count = %d\n", m_count);

                m_inc = M - m_count > M_BLOCKING ? M_BLOCKING : M - m_count;
                pzypacking_a_k120(alpha, A + m_count + k_count * LDA, a_buffer, LDA, m_inc, k_inc);

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
                        ptr_packing_b = b_buffer + k_inc * inner_n_count;
                        ptr_c = C + i + j * LDC;
                        int64_t K_inc = (int64_t)k_inc, LDC_in_bytes = (int64_t)LDC * sizeof(double);
                        

                        // add register reuse on C(i,j) to reduce the times of accessing C(i,j) in memory
                        macro_kernel_24xkx8_avx512_packing_inlineASM_k120(ptr_packing_a, ptr_packing_b, ptr_c, K_inc, LDC_in_bytes);
                        
                    }
                }

                //printf("M24 = %d, m_inc = %d, N8 = %d, n_inc = %d\n", M24, m_inc, N8, n_inc);
                
                // here is not the end of the function, so shouldn't use "return"
                //if (M24 == m_inc && N8 == n_inc) return; 
                
                // Attention! I didn't check the edge case, so the following codes are not sure for correctness
                // boundary conditions
                if (M24 != m_inc) {
                    printf("enter edge case for m_inc. \n");
                    pzydgemm_cpu_opt_k120(m_inc - M24, n_inc, k_inc, alpha, &A(M24, 0), LDA, B, LDB, 1.0, &C(M24, 0), LDC); // A+M24 move to M24 row, because it's column major
                }
                if (N8 != n_inc) {
                    printf("enter edge case for n_inc. \n");
                    pzydgemm_cpu_opt_k120(M24, n_inc - N8, k_inc, alpha, A, LDA, &B(0, N8), LDB, 1.0, &C(0, N8), LDC);
                }
            }
            //printf("m_count = %d\n", m_count);

            // edge case for M
            if (m_inc != M_BLOCKING && m_count != 0){
                printf("enter edge case for M. \n");
                pzydgemm_cpu_opt_k120(m_inc, N_BLOCKING, K, alpha, &A(m_count, 0), LDA, &B(0, n_count), LDB, 1.0, &C(m_count, n_count), LDC);
            }

        }

        // edge case for N
        if (n_inc != N_BLOCKING && n_count != 0){
            printf("enter edge case for N. \n");
            pzydgemm_cpu_opt_k120(M, n_inc, K, alpha, A, LDA, &B(0, n_count), LDB, 1.0, &C(0, n_count), LDC);
        }
    }
    free(a_buffer);free(b_buffer);
}