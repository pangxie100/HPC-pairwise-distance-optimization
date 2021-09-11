#include "immintrin.h"
#define A(i,j) A[(i)+(j)*LDA]
#define B(i,j) B[(i)+(j)*LDB]
#define C(i,j) C[(i)+(j)*LDC]
// discontinuous packing + cache blocking + 8*6 register blocking with AVX2 + loop unrolling * 4
// we can try 8 * 6 kernel, which needs 12 + 2 + 1 = 15 (max is 16)

///*
#define M_BLOCKING 192
//#define N_BLOCKING 112 // very big 2240
//#define N_BLOCKING 2240 // get a higher performance
#define N_BLOCKING 168 // for 4, 6, 8, 14 in N of kernel // 3360
#define K_BLOCKING 384
//*/

/*
// test
#define M_BLOCKING 16
#define N_BLOCKING 8
#define K_BLOCKING 32
//*/

// C := alpha * op(A) * op(B) + beta * C
void pzyscale_C_k122(int M, int N, double beta, double *C, int LDC){
    int i, j;
    for (i = 0; i < M; i++){
        for (j = 0; j < N; j++){
            C(i,j) *= beta; 
        }
    }
}

void pzydgemm_cpu_opt_k122(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    int i, j, k;
    if (beta != 1.0) pzyscale_C_k122(M,N,beta,C,LDC);
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

void pzypacking_b_k122(double *packsrc, double *packdst, int LDB, int dim_k, int dim_n){
    // kernel A * B is 8 * 4, thus we need four pointers to store each number in one row (col-major)
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

// changed vs kernel 112
void pzypacking_a_k122(double alpha, double *packsrc, double *packdst, int LDA, int dim_m, int dim_k){
    // kernel A * B is 8 * 4, we can use one pointer to store 8 numbers in one column (col-major)
    double *src, *dst;
    __m512d valpha1 = _mm512_set1_pd(alpha); // broadcast alpha to a 512-bit vector, the input is a double value
    __m256d valpha2 = _mm256_set1_pd(alpha); // broadcast alpha to a 256-bit vector, the input is a double value
    dst = packdst;
    int i, k, remain = dim_m;
    for (i = 0; remain > 7; remain -= 8, i += 8){
        src = packsrc + i;
        for (k = 0; k < dim_k; k++){
            _mm512_store_pd(dst, _mm512_mul_pd(_mm512_loadu_pd(src), valpha1));
            src += LDA;
            dst += 8;
        }
    }
    // for the remained part which is smaller than 8, use 4 * 4 kernel style to store
    for (; remain > 3; remain -= 4, i += 4){
        src = packsrc + i;
        for (k = 0; k < dim_k; k++){
            _mm256_store_pd(dst, _mm256_mul_pd(_mm256_loadu_pd(src), valpha2));
            src += LDA;
            dst += 4;
        }
    }
}

#define INIT_m8n6 \
  "vpxor %%ymm4,%%ymm4,%%ymm4; vpxor %%ymm5,%%ymm5,%%ymm5; vpxor %%ymm6,%%ymm6,%%ymm6; vpxor %%ymm7,%%ymm7,%%ymm7;"\
  "vpxor %%ymm8,%%ymm8,%%ymm8; vpxor %%ymm9,%%ymm9,%%ymm9; vpxor %%ymm10,%%ymm10,%%ymm10; vpxor %%ymm11,%%ymm11,%%ymm11;"\
  "vpxor %%ymm12,%%ymm12,%%ymm12; vpxor %%ymm13,%%ymm13,%%ymm13; vpxor %%ymm14,%%ymm14,%%ymm14; vpxor %%ymm15,%%ymm15,%%ymm15;"

// ymm0 - ymm1 = ax0, ay0
#define load_a_m8 \
  "vmovapd (%0),%%ymm0; vmovapd 32(%0),%%ymm1; addq $64,%0;"

// ymm2 - ymm3 = b00, b01
// ymm4 - ymm9 = cx0 - cx5
// ymm10 - ymm15 = cy0 - cy5
#define kernel_m8n2_1 \
  "vbroadcastsd (%1),%%ymm2; vbroadcastsd 8(%1),%%ymm3;"\
  "vfmadd231pd %%ymm0,%%ymm2,%%ymm4; vfmadd231pd %%ymm0,%%ymm3,%%ymm5;"\
  "vfmadd231pd %%ymm1,%%ymm2,%%ymm10; vfmadd231pd %%ymm1,%%ymm3,%%ymm11;"
#define kernel_m8n2_2 \
  "vbroadcastsd (%1,%%r11,1),%%ymm2; vbroadcastsd 8(%1,%%r11,1),%%ymm3;"\
  "vfmadd231pd %%ymm0,%%ymm2,%%ymm6; vfmadd231pd %%ymm0,%%ymm3,%%ymm7;"\
  "vfmadd231pd %%ymm1,%%ymm2,%%ymm12; vfmadd231pd %%ymm1,%%ymm3,%%ymm13;"
#define kernel_m8n2_3 \
  "vbroadcastsd (%1,%%r11,2),%%ymm2; vbroadcastsd 8(%1,%%r11,2),%%ymm3;"\
  "vfmadd231pd %%ymm0,%%ymm2,%%ymm8; vfmadd231pd %%ymm0,%%ymm3,%%ymm9;"\
  "vfmadd231pd %%ymm1,%%ymm2,%%ymm14; vfmadd231pd %%ymm1,%%ymm3,%%ymm15;"
#define KERNEL_m8n6 \
  load_a_m8 \
  kernel_m8n2_1 \
  kernel_m8n2_2 \
  kernel_m8n2_3 \
  "addq $16,%1;"

#define save_m8n2_1 \
  "vaddpd (%2),%%ymm4,%%ymm4; vmovupd %%ymm4,(%2);"\
  "vaddpd 32(%2),%%ymm10,%%ymm10; vmovupd %%ymm10,32(%2);"\
  "vaddpd (%2,%3,1),%%ymm5,%%ymm5; vmovupd %%ymm5,(%2,%3,1);"\
  "vaddpd 32(%2,%3,1),%%ymm11,%%ymm11; vmovupd %%ymm11,32(%2,%3,1);"\
  "leaq (%2,%3,2),%2;"
#define save_m8n2_2 \
  "vaddpd (%2),%%ymm6,%%ymm6; vmovupd %%ymm6,(%2);"\
  "vaddpd 32(%2),%%ymm12,%%ymm12; vmovupd %%ymm12,32(%2);"\
  "vaddpd (%2,%3,1),%%ymm7,%%ymm7; vmovupd %%ymm7,(%2,%3,1);"\
  "vaddpd 32(%2,%3,1),%%ymm13,%%ymm13; vmovupd %%ymm13,32(%2,%3,1);"\
  "leaq (%2,%3,2),%2;"
#define save_m8n2_3 \
  "vaddpd (%2),%%ymm8,%%ymm8; vmovupd %%ymm8,(%2);"\
  "vaddpd 32(%2),%%ymm14,%%ymm14; vmovupd %%ymm14,32(%2);"\
  "vaddpd (%2,%3,1),%%ymm9,%%ymm9; vmovupd %%ymm9,(%2,%3,1);"\
  "vaddpd 32(%2,%3,1),%%ymm15,%%ymm15; vmovupd %%ymm15,32(%2,%3,1);"\
  "leaq (%2,%3,2),%2;" // this is not necessary for this kernel
#define SAVE_m8n6 \
  save_m8n2_1 \
  save_m8n2_2 \
  save_m8n2_3 \

void macro_kernel_8xkx6_avx2_packing_inlineASM(double *ptr_packing_a, double *ptr_packing_b, double *ptr_c, int64_t K_inc, int64_t LDC_in_bytes){
    __asm__ __volatile__(
        // salq: shift, salq $4,%%r11 => r11 = r11 * 16
        // leaq : "lea" is load effective address, it's not loading value from that address 
        // https://courses.cs.washington.edu/courses/cse351/17wi/lectures/CSE351-L09-x86-II_17wi.pdf
        // salq : 'q' means "quadword"
        "movq %4,%%r11; salq $4,%%r11;\n\t"
        "movq %4,%%r12;\n\t"
        INIT_m8n6 \
        "cmpq $4,%%r12; jb 8062f;\n\t" // (only numbers)several numbers with random length: 0 = *, 8 * 6 + 1, 2, 3, ...
        "8061:\n\t" // main kernel loop
        KERNEL_m8n6 \
        KERNEL_m8n6 \
        KERNEL_m8n6 \
        KERNEL_m8n6 \
        "subq $4,%%r12; cmpq $4,%%r12; jnb 8061b;\n\t" // not below: r13 >= 4
        "cmpq $0,%%r12; je 8063f;\n\t"
        "8062:\n\t"
        KERNEL_m8n6 \
        "decq %%r12; testq %%r12,%%r12; jnz 8062b;\n\t" // it skips only when r13 = 0, r13 & r13 = 0
        "8063:\n\t"
        SAVE_m8n6 \
        // "+" means we will modify the value in this register, tell the compiler to not optimize it
        :"+r"(ptr_packing_a), // %0
         "+r"(ptr_packing_b), // %1
         "+r"(ptr_c),         // %2
         "+r"(LDC_in_bytes)   // %3
        :"m"(K_inc)           // %4
        :"r11","r12","cc","memory"
    );
}

// changed vs kernel 112
void pzydgemm_cpu_v122(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    if (beta != 1.0) pzyscale_C_k122(M, N, beta, C, LDC);
    // difference between malloc and aligned_alloc : 
    // https://stackoverflow.com/questions/39677063/difference-between-aligned-malloc-and-standard-malloc
    
    // usage of aligned_alloc: 
    // https://en.cppreference.com/w/c/memory/aligned_alloc
    // https://zhuanlan.zhihu.com/p/111780698
    // it says that "K_BLOCKING * N_BLOCKING * sizeof(double)" or "K_BLOCKING * M_BLOCKING * sizeof(double)" should be an integral multiple of 4096 here
    double *b_buffer = (double *)aligned_alloc(4096,K_BLOCKING*N_BLOCKING*sizeof(double));
    double *a_buffer = (double *)aligned_alloc(4096,K_BLOCKING*M_BLOCKING*sizeof(double));
    double *ptr_packing_a, *ptr_packing_b, *ptr_c;

    __m256d ax0, ay0, b00, b01;
    __m256d cx0, cx1, cx2, cx3, cx4, cx5, cy0, cy1, cy2, cy3, cy4, cy5;

    int n_count = 0, k_count = 0, m_count = 0;
    int n_inc = 0, k_inc = 0, m_inc = 0;

    // get an integer which is divisible by 8, 6, 4(eg: 16 divided by 8 is 2)
    int M8 = 0, N6 = 0; // 8 * 6 kernel
    int K4 = 0; // loop unrolling 4 times
    int i = 0, j = 0, k = 0;
    int inner_m_count = 0, inner_n_count = 0, inner_k_count = 0, inner_k_end = 0;

    for (n_count = 0; n_count < N; n_count += n_inc){
        n_inc = (N - n_count > N_BLOCKING) ? N_BLOCKING : N - n_count;

        for (k_count = 0; k_count < K; k_count += k_inc){
                k_inc = K - k_count > K_BLOCKING ? K_BLOCKING : K - k_count;
                pzypacking_b_k122(B + k_count + n_count * LDB, b_buffer, LDB, k_inc, n_inc);

            for (m_count = 0; m_count < M; m_count += m_inc){
                //printf("m_count = %d\n", m_count);
                m_inc = M - m_count > M_BLOCKING ? M_BLOCKING : M - m_count;
                pzypacking_a_k122(alpha, A + m_count + k_count * LDA, a_buffer, LDA, m_inc, k_inc);

                N6 = (n_inc / 6) * 6;
                M8 = m_inc & -8;
                K4 = k_inc & -4;

                //printf("K4 = %d\n", K4);
                for (inner_m_count = 0; inner_m_count < M8; inner_m_count += 8){
                    i = m_count + inner_m_count;
                    //printf("i = %d\n", i);
                    // each time for specific i, j loop means to compute all 8*6 matrix in same row on matrix C.
                    for (inner_n_count = 0; inner_n_count < N6; inner_n_count += 6){ 
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
                        macro_kernel_8xkx6_avx2_packing_inlineASM(ptr_packing_a, ptr_packing_b, ptr_c, K_inc, LDC_in_bytes);
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
                    pzydgemm_cpu_opt_k122(m_inc - M8, n_inc, k_inc, alpha, &A(M8, 0), LDA, B, LDB, 1.0, &C(M8, 0), LDC); // A+M8 move to M8 row, because it's column major
                }
                if (N6 != n_inc) {
                    printf("enter edge case for n_inc. \n");
                    pzydgemm_cpu_opt_k122(M8, n_inc - N6, k_inc, alpha, A, LDA, &B(0, N6), LDB, 1.0, &C(0, N6), LDC);
                }
            }
            //printf("m_count = %d\n", m_count);

            // edge case for M
            if (m_inc != M_BLOCKING && m_count != 0){
                printf("enter edge case for M. \n");
                pzydgemm_cpu_opt_k122(m_inc, N_BLOCKING, K, alpha, &A(m_count, 0), LDA, &B(0, n_count), LDB, 1.0, &C(m_count, n_count), LDC);
            }

        }

        // edge case for N
        if (n_inc != N_BLOCKING && n_count != 0){
            printf("enter edge case for N. \n");
            pzydgemm_cpu_opt_k122(M, n_inc, K, alpha, A, LDA, &B(0, n_count), LDB, 1.0, &C(0, n_count), LDC);
        }
    }
    free(a_buffer);free(b_buffer);
}