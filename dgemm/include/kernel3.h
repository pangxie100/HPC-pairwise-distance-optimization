#define A(i,j) A[(i)+(j)*LDA]
#define B(i,j) B[(i)+(j)*LDB]
#define C(i,j) C[(i)+(j)*LDC]
// 2*2 register blocking

// C := alpha * op(A) * op(B) + beta * C
void pzyscale_C_k3(int M, int N, double beta, double *C, int LDC){
    int i,j;
    for (i = 0; i < M; i++){
        for (j = 0; j < N; j++){
            C(i,j) *= beta; 
        }
    }
}

void pzydgemm_cpu_opt_k3(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    int i,j,k;
    if (beta != 1.0) pzyscale_C_k3(M,N,beta,C,LDC);
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

void pzydgemm_cpu_v3(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    int i,j,k;
    if (beta != 1.0) pzyscale_C_k3(M,N,beta,C,LDC);
    // get an integer which is divisible by 2(eg: 16 divided by 2 is 8)
    int M2=M&-2, N2=N&-2;
    for (i = 0; i < M2; i+=2){
        for (j = 0; j < N2; j+=2){
            // add register reuse on C(i,j) to reduce the times of accessing C(i,j) in memory
            double c00 = C(i,j);
            double c01 = C(i,j+1);
            double c10 = C(i+1,j);
            double c11 = C(i+1,j+1);
            for (k = 0; k < K; k++){
                double a00 = alpha * A(i,k);
                double a10 = alpha * A(i+1,k);
                double b00 = B(k,j);
                double b01 = B(k,j+1);
                c00 += a00 * b00;
                c01 += a00 * b01;
                c10 += a10 * b00;
                c11 += a10 * b01;
            }
            C(i,j) = c00;
            C(i,j+1) = c01;
            C(i+1,j) = c10;
            C(i+1,j+1) = c11;
        }
    }
    if(M2 == M && N2 == N) return;
    // boundary conditions
    if (M2 != M) pzydgemm_cpu_opt_k3(M - M2, N, K, alpha, A + M2, LDA, B, LDB, 1.0, &C(M2, 0), LDC);
    if (N2 != N) pzydgemm_cpu_opt_k3(M2, N - N2, K, alpha, A, LDA, &B(0, N2), LDB, 1.0, &C(0, N2), LDC);
}