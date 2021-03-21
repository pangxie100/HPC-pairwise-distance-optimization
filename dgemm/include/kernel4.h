#define A(i,j) A[(i)+(j)*LDA]
#define B(i,j) B[(i)+(j)*LDB]
#define C(i,j) C[(i)+(j)*LDC]

// 4*4 register blocking, still can be improved by adding one 2*2 register blocking

// C := alpha * op(A) * op(B) + beta * C
void pzyscale_C_k4(int M, int N, double beta, double *C, int LDC){
    int i,j;
    for (i = 0; i < M; i++){
        for (j = 0; j < N; j++){
            C(i,j) *= beta; 
        }
    }
}

void pzydgemm_cpu_opt_k4(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    int i,j,k;
    if (beta != 1.0) pzyscale_C_k4(M,N,beta,C,LDC);
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

void pzydgemm_cpu_v4(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    int i,j,k;
    if (beta != 1.0) pzyscale_C_k4(M,N,beta,C,LDC);
    // get an integer which is divisible by 2(eg: 16 divided by 2 is 8)
    int M4=M&-4, N4=N&-4;
    for (i = 0; i < M4; i+=4){
        for (j = 0; j < N4; j+=4){
            // add register reuse on C(i,j) to reduce the times of accessing C(i,j) in memory
            double c00 = C(i,j);
            double c01 = C(i,j+1);
            double c02 = C(i,j+2);
            double c03 = C(i,j+3);
            double c10 = C(i+1,j);
            double c11 = C(i+1,j+1);
            double c12 = C(i+1,j+2);
            double c13 = C(i+1,j+3);
            double c20 = C(i+2,j);
            double c21 = C(i+2,j+1);
            double c22 = C(i+2,j+2);
            double c23 = C(i+2,j+3);
            double c30 = C(i+3,j);
            double c31 = C(i+3,j+1);
            double c32 = C(i+3,j+2);
            double c33 = C(i+3,j+3);
            for (k = 0; k < K; k++){
                double a00 = alpha * A(i,k);
                double a10 = alpha * A(i+1,k);
                double a20 = alpha * A(i+2,k);
                double a30 = alpha * A(i+3,k);
                double b00 = B(k,j);
                double b01 = B(k,j+1);
                double b02 = B(k,j+2);
                double b03 = B(k,j+3);
                c00 += a00 * b00;
                c01 += a00 * b01;
                c02 += a00 * b02;
                c03 += a00 * b03;
                c10 += a10 * b00;
                c11 += a10 * b01;
                c12 += a10 * b02;
                c13 += a10 * b03;
                c20 += a20 * b00;
                c21 += a20 * b01;
                c22 += a20 * b02;
                c23 += a20 * b03;
                c30 += a30 * b00;
                c31 += a30 * b01;
                c32 += a30 * b02;
                c33 += a30 * b03;
            }
            C(i,j) = c00;
            C(i,j+1) = c01;
            C(i,j+2) = c02;
            C(i,j+3) = c03;
            C(i+1,j) = c10;
            C(i+1,j+1) = c11;
            C(i+1,j+2) = c12;
            C(i+1,j+3) = c13;
            C(i+2,j) = c20;
            C(i+2,j+1) = c21;
            C(i+2,j+2) = c22;
            C(i+2,j+3) = c23;
            C(i+3,j) = c30;
            C(i+3,j+1) = c31;
            C(i+3,j+2) = c32;
            C(i+3,j+3) = c33;
        }
    }
    if(M4 == M && N4 == N) return;
    // boundary conditions
    if (M4 != M) mydgemm_cpu_opt_k4(M-M4,N,K,alpha,A+M4,LDA,B,LDB,1.0,&C(M4,0),LDC);
    if (N4 != N) mydgemm_cpu_opt_k4(M4,N-N4,K,alpha,A,LDA,&B(0,N4),LDB,1.0,&C(0,N4),LDC);
}