#include "immintrin.h"
#include <stdint.h> // for int64_t
#define A(i,j) A[(i)+(j)*LDA]
#define B(i,j) B[(i)+(j)*LDB]
#define C(i,j) C[(i)+(j)*LDC]
// a bigger region applying inline assembling + discontinuous packing(2 - 2 packing) + cache blocking + 24*8 register blocking with AVX512 + loop unrolling * 4
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

// 8*4 register blocking, still can be improved by adding one 4*2 register blocking or 4*4

// C := alpha * op(A) * op(B) + beta * C
void pzyscale_C_k130(int M, int N, double beta, double *C, int LDC){
    int i, j;
    for (i = 0; i < M; i++){
        for (j = 0; j < N; j++){
            C(i,j) *= beta; 
        }
    }
}

void pzydgemm_cpu_opt_k130(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    int i, j, k;
    if (beta != 1.0) pzyscale_C_k130(M, N, beta, C, LDC);
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

void pzypacking_b_k130(double *packsrc, double *packdst, int LDB, int dim_k, int dim_n){
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

void pzypacking_a_k130(double alpha, double *packsrc, double *packdst, int LDA, int dim_m, int dim_k){
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

#define INIT_m24n8 \
  "vpxorq %%zmm8,%%zmm8,%%zmm8; vpxorq %%zmm9,%%zmm9,%%zmm9; vpxorq %%zmm10,%%zmm10,%%zmm10; vpxorq %%zmm11,%%zmm11,%%zmm11;"\
  "vpxorq %%zmm12,%%zmm12,%%zmm12; vpxorq %%zmm13,%%zmm13,%%zmm13; vpxorq %%zmm14,%%zmm14,%%zmm14; vpxorq %%zmm15,%%zmm15,%%zmm15;"\
  "vpxorq %%zmm16,%%zmm16,%%zmm16; vpxorq %%zmm17,%%zmm17,%%zmm17; vpxorq %%zmm18,%%zmm18,%%zmm18; vpxorq %%zmm19,%%zmm19,%%zmm19;"\
  "vpxorq %%zmm20,%%zmm20,%%zmm20; vpxorq %%zmm21,%%zmm21,%%zmm21; vpxorq %%zmm22,%%zmm22,%%zmm22; vpxorq %%zmm23,%%zmm23,%%zmm23;"\
  "vpxorq %%zmm24,%%zmm24,%%zmm24; vpxorq %%zmm25,%%zmm25,%%zmm25; vpxorq %%zmm26,%%zmm26,%%zmm26; vpxorq %%zmm27,%%zmm27,%%zmm27;"\
  "vpxorq %%zmm28,%%zmm28,%%zmm28; vpxorq %%zmm29,%%zmm29,%%zmm29; vpxorq %%zmm30,%%zmm30,%%zmm30; vpxorq %%zmm31,%%zmm31,%%zmm31;"

// zmm0 - zmm2 = ax0, ay0, az0
// in load_a_m24, vmovups is also fine
#define load_a_m24 \
  "vmovaps (%0),%%zmm0; vmovaps 64(%0),%%zmm1; vmovaps 128(%0),%%zmm2; addq $192,%0;"
// zmm3, zmm4 = b00, b01
// zmm8 - zmm15 = cx0 - cx7
// zmm16 - zmm23 = cy0 - cy7
// zmm24 - zmm31 = cz0 - cz7
#define kernel_m24n2_1 \
  "vbroadcastsd (%1),%%zmm3; vbroadcastsd 8(%1),%%zmm4;"\
  "vfmadd231pd %%zmm0,%%zmm3,%%zmm8; vfmadd231pd %%zmm0,%%zmm4,%%zmm9;"\
  "vfmadd231pd %%zmm1,%%zmm3,%%zmm16; vfmadd231pd %%zmm1,%%zmm4,%%zmm17;"\
  "vfmadd231pd %%zmm2,%%zmm3,%%zmm24; vfmadd231pd %%zmm2,%%zmm4,%%zmm25;"
#define kernel_m24n2_2 \
  "vbroadcastsd (%1,%%r11,1),%%zmm3;vbroadcastsd 8(%1,%%r11,1),%%zmm4;"\
  "vfmadd231pd %%zmm0,%%zmm3,%%zmm10; vfmadd231pd %%zmm0,%%zmm4,%%zmm11;"\
  "vfmadd231pd %%zmm1,%%zmm3,%%zmm18; vfmadd231pd %%zmm1,%%zmm4,%%zmm19;"\
  "vfmadd231pd %%zmm2,%%zmm3,%%zmm26; vfmadd231pd %%zmm2,%%zmm4,%%zmm27;"
#define kernel_m24n2_3 \
  "vbroadcastsd (%1,%%r11,2),%%zmm3;vbroadcastsd 8(%1,%%r11,2),%%zmm4;"\
  "vfmadd231pd %%zmm0,%%zmm3,%%zmm12; vfmadd231pd %%zmm0,%%zmm4,%%zmm13;"\
  "vfmadd231pd %%zmm1,%%zmm3,%%zmm20; vfmadd231pd %%zmm1,%%zmm4,%%zmm21;"\
  "vfmadd231pd %%zmm2,%%zmm3,%%zmm28; vfmadd231pd %%zmm2,%%zmm4,%%zmm29;"
#define kernel_m24n2_4 \
  "vbroadcastsd (%%r12),%%zmm3;vbroadcastsd 8(%%r12),%%zmm4;"\
  "vfmadd231pd %%zmm0,%%zmm3,%%zmm14; vfmadd231pd %%zmm0,%%zmm4,%%zmm15;"\
  "vfmadd231pd %%zmm1,%%zmm3,%%zmm22; vfmadd231pd %%zmm1,%%zmm4,%%zmm23;"\
  "vfmadd231pd %%zmm2,%%zmm3,%%zmm30; vfmadd231pd %%zmm2,%%zmm4,%%zmm31;"
#define KERNEL_m24n8 \
  load_a_m24 \
  kernel_m24n2_1 \
  kernel_m24n2_2 \
  kernel_m24n2_3 \
  kernel_m24n2_4 \
  "addq $16,%1;addq $16,%%r12;"

#define save_m24n2_1 \
  "vaddpd (%3),%%zmm8,%%zmm8; vmovups %%zmm8,(%3);"\
  "vaddpd 64(%3),%%zmm16,%%zmm16; vmovups %%zmm16,64(%3);"\
  "vaddpd 128(%3),%%zmm24,%%zmm24; vmovups %%zmm24,128(%3);"\
  "vaddpd (%3,%4,1),%%zmm9,%%zmm9; vmovups %%zmm9,(%3,%4,1);"\
  "vaddpd 64(%3,%4,1),%%zmm17,%%zmm17; vmovups %%zmm17,64(%3,%4,1);"\
  "vaddpd 128(%3,%4,1),%%zmm25,%%zmm25; vmovups %%zmm25,128(%3,%4,1);"\
  "leaq (%3,%4,2),%3;"
#define save_m24n2_2 \
  "vaddpd (%3),%%zmm10,%%zmm10; vmovups %%zmm10,(%3);"\
  "vaddpd 64(%3),%%zmm18,%%zmm18; vmovups %%zmm18,64(%3);"\
  "vaddpd 128(%3),%%zmm26,%%zmm26; vmovups %%zmm26,128(%3);"\
  "vaddpd (%3,%4,1),%%zmm11,%%zmm11; vmovups %%zmm11,(%3,%4,1);"\
  "vaddpd 64(%3,%4,1),%%zmm19,%%zmm19; vmovups %%zmm19,64(%3,%4,1);"\
  "vaddpd 128(%3,%4,1),%%zmm27,%%zmm27; vmovups %%zmm27,128(%3,%4,1);"\
  "leaq (%3,%4,2),%3;"
#define save_m24n2_3 \
  "vaddpd (%3),%%zmm12,%%zmm12; vmovups %%zmm12,(%3);"\
  "vaddpd 64(%3),%%zmm20,%%zmm20; vmovups %%zmm20,64(%3);"\
  "vaddpd 128(%3),%%zmm28,%%zmm28; vmovups %%zmm28,128(%3);"\
  "vaddpd (%3,%4,1),%%zmm13,%%zmm13; vmovups %%zmm13,(%3,%4,1);"\
  "vaddpd 64(%3,%4,1),%%zmm21,%%zmm21; vmovups %%zmm21,64(%3,%4,1);"\
  "vaddpd 128(%3,%4,1),%%zmm29,%%zmm29; vmovups %%zmm29,128(%3,%4,1);"\
  "leaq (%3,%4,2),%3;"
#define save_m24n2_4 \
  "vaddpd (%3),%%zmm14,%%zmm14; vmovups %%zmm14,(%3);"\
  "vaddpd 64(%3),%%zmm22,%%zmm22; vmovups %%zmm22,64(%3);"\
  "vaddpd 128(%3),%%zmm30,%%zmm30; vmovups %%zmm30,128(%3);"\
  "vaddpd (%3,%4,1),%%zmm15,%%zmm15; vmovups %%zmm15,(%3,%4,1);"\
  "vaddpd 64(%3,%4,1),%%zmm23,%%zmm23; vmovups %%zmm23,64(%3,%4,1);"\
  "vaddpd 128(%3,%4,1),%%zmm31,%%zmm31; vmovups %%zmm31,128(%3,%4,1);"\
  "leaq (%3,%4,2),%3;" //this is not necessary
#define save_init_m24 \
  "movq %2,%3; addq $192,%2;"
#define SAVE_m24n8 \
  save_init_m24 \
  save_m24n2_1 \
  save_m24n2_2 \
  save_m24n2_3 \
  save_m24n2_4

#define COMPUTE_m24n8 \
    "movq %%r14,%1;leaq (%1,%%r11,2),%%r12;addq %%r11,%%r12;movq %5,%%r13;"\
    INIT_m24n8 \
    "cmpq $4,%%r13;jb 724782f;\n\t"\
    "724781:\n\t"\
    KERNEL_m24n8 \
    KERNEL_m24n8 \
    KERNEL_m24n8 "subq $4,%%r13;"\
    KERNEL_m24n8 \
    "cmpq $4,%%r13;jnb 724781b;\n\t"\
    "cmpq $0,%%r13;je 724783f;\n\t"\
    "724782:\n\t"\
    KERNEL_m24n8 \
    "testq %%r13,%%r13;jnz 724782b;\n\t"\
    "724783:\n\t"\
    SAVE_m24n8 

// %0: ptr_packing_a, %1: ptr_packing_b, %2: ptr_c, %3: ptr_c_tmp, %4: LDC_in_bytes, %5: K_inc, %6: M_inc
#define MACRO_KERNEL_n8 {\
  __asm__ __volatile__(\
    "movq %6,%%r15; movq %1,%%r14; movq %5,%%r11; salq $4,%%r11;"\
    "cmpq $24,%%r15; jb 3243831f;"\
    "3243830:\n\t"\
    COMPUTE_m24n8 "subq $24,%%r15; cmpq $24,%%r15; jnb 3243830b;"\
    "3243831:\n\t"\
    "movq %%r14,%1;"\
    :"+r"(ptr_packing_a),"+r"(ptr_packing_b),"+r"(ptr_c),"+r"(ptr_c_tmp),"+r"(LDC_in_bytes)\
    :"m"(K_inc),"m"(M_inc)\
    :"r11","r12","r13","r14","r15","cc","memory",\
    "zmm0","zmm1","zmm2","zmm3","zmm4","zmm5","zmm6","zmm7","zmm8","zmm9","zmm10","zmm11","zmm12","zmm13","zmm14","zmm15",\
    "zmm16","zmm17","zmm18","zmm19","zmm20","zmm21","zmm22","zmm23","zmm24","zmm25","zmm26","zmm27","zmm28","zmm29","zmm30","zmm31");\
  ptr_packing_a -= M_inc * K_inc; ptr_packing_b += 8 * K_inc; ptr_c += 8 * ldc - M_inc;\
}


// changed vs kernel 12
void pzydgemm_cpu_v130(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    if (beta != 1.0) pzyscale_C_k130(M, N, beta, C, LDC);
    // difference between malloc and aligned_alloc : 
    // https://stackoverflow.com/questions/39677063/difference-between-aligned-malloc-and-standard-malloc
    
    // usage of aligned_alloc: (the address is aligned)
    // https://en.cppreference.com/w/c/memory/aligned_alloc
    // https://zhuanlan.zhihu.com/p/111780698
    // it says that "K_BLOCKING * N_BLOCKING * sizeof(double)" or "K_BLOCKING * M_BLOCKING * sizeof(double)" should be an integral multiple of 4096 here
    double *b_buffer = (double *)aligned_alloc(4096,K_BLOCKING*N_BLOCKING*sizeof(double));
    double *a_buffer = (double *)aligned_alloc(4096,K_BLOCKING*M_BLOCKING*sizeof(double));
    double *ptr_packing_a, *ptr_packing_b, *ptr_c = C, *ptr_c_tmp = C;

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

                pzypacking_b_k130(B + k_count + n_count * LDB, b_buffer, LDB, k_inc, n_inc);

            for (m_count = 0; m_count < M; m_count += m_inc){
                //printf("m_count = %d\n", m_count);

                m_inc = M - m_count > M_BLOCKING ? M_BLOCKING : M - m_count;
                pzypacking_a_k130(alpha, A + m_count + k_count * LDA, a_buffer, LDA, m_inc, k_inc);

                ptr_packing_a = a_buffer;
                ptr_packing_b = b_buffer;
                int64_t M_inc = (int64_t)m_inc, K_inc = (int64_t)k_inc, LDC_in_bytes = (int64_t)LDC * sizeof(double), ldc = (int32_t)LDC;

                N8 = n_inc & -8;
                //M24 = m_inc & -24; // this way only for 2^x, 24 is not.
                M24 = (m_inc / 24) * 24;
                K4 = k_inc & -4;

                // each time for specific i, j loop means to compute all 24*8 matrix in same row on matrix C.
                for (inner_n_count = 0; inner_n_count < N8; inner_n_count += 8){ 
                    //printf("inner_n_count = %d\n", inner_n_count);
                    
                    MACRO_KERNEL_n8

                }

                //printf("M24 = %d, m_inc = %d, N8 = %d, n_inc = %d\n", M24, m_inc, N8, n_inc);
                
                // here is not the end of the function, so shouldn't use "return"
                //if (M24 == m_inc && N8 == n_inc) return; 
                
                // Attention! I didn't check the edge case, so the following codes are not sure for correctness
                // boundary conditions
                if (M24 != m_inc) {
                    printf("enter edge case for m_inc. \n");
                    pzydgemm_cpu_opt_k130(m_inc - M24, n_inc, k_inc, alpha, &A(M24, 0), LDA, B, LDB, 1.0, &C(M24, 0), LDC); // A+M24 move to M24 row, because it's column major
                }
                if (N8 != n_inc) {
                    printf("enter edge case for n_inc. \n");
                    pzydgemm_cpu_opt_k130(M24, n_inc - N8, k_inc, alpha, A, LDA, &B(0, N8), LDB, 1.0, &C(0, N8), LDC);
                }
            }
            //printf("m_count = %d\n", m_count);

            // edge case for M
            if (m_inc != M_BLOCKING && m_count != 0){
                printf("enter edge case for M. \n");
                pzydgemm_cpu_opt_k130(m_inc, N_BLOCKING, K, alpha, &A(m_count, 0), LDA, &B(0, n_count), LDB, 1.0, &C(m_count, n_count), LDC);
            }

        }

        // edge case for N
        if (n_inc != N_BLOCKING && n_count != 0){
            printf("enter edge case for N. \n");
            pzydgemm_cpu_opt_k130(M, n_inc, K, alpha, A, LDA, &B(0, n_count), LDB, 1.0, &C(0, n_count), LDC);
        }
    }
    free(a_buffer);free(b_buffer);
}