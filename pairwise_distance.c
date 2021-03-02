#include <stdio.h>
#include <stdlib.h>
#include "util.h"
#include <math.h>
#define A(i,j) A[(i)+(j)*LDA]
#define A_T(i,j) A_T[(i)+(j)*LDA_T]
#define B(i,j) B[(i)+(j)*LDB]
#define C(i,j) C[(i)+(j)*LDC]
#define M 3
#define N 2
#define K 4

// Version 1: this is -2 * x' * y matrix multiplication
int main(int argc, char *argv[]){
    double *A, *B, *C, *A_T;
    int LDA = K, LDB = K, LDC = M, LDA_T = M;
    A = (double *)malloc(sizeof(double) * K * M);
    A_T = (double *)malloc(sizeof(double) * M * K);
    B = (double *)malloc(sizeof(double) * K * N);
    C = (double *)malloc(sizeof(double) * M * N);
    
    // Initialization
    double num = 1.0;
    printf("matrix x is initialized as:\n");
    for (int k = 0; k < K; k++){
        for (int i = 0; i < M; i++){
            A(k,i) = num;
            //printf("%f ", A(k,i));
            num++;
        }
        //printf("\n");
    }
    //printf("\n");
    print_matrix(A, K, M);
    num = 1.0;
    printf("matrix y is initialized as:\n");
    for (int k = 0; k < K; k++){
        for (int j = 0; j < N; j++){
            B(k,j) = num;
            //printf("%f ", B(k,j));
            num++;
        }
        //printf("\n");
    }
    //printf("\n");
    print_matrix(B, K, N);
    //randomize_matrix(A,max_size,max_size);
    //randomize_matrix(B,max_size,max_size);
    
    // pairwise distance:
    // ||x-y||^2 = |x|^2 + |y|^2 - 2 * x * y

    // x (K * M) -> the transpose of x: x' (M * K)
    for (int k = 0; k < K; k++){
        for (int i = 0; i < M; i++){
            //printf("%5.2f ", A(k,i));
            A_T(i,k) = A(k,i); 
            //printf("%5.2f\n", A_T(i,k));      
        }
    }
    printf("the transpose matrix x' is:\n");
    print_matrix(A_T, M, K);
    print_array(A_T, M * K);

    // x (K * M) -> the transpose of x: x' (M * K)
    int temp = 0;
    for (int k = 0; k < K; k++){
        for (int i = 0; i < M; i++){
            //printf("%5.2f ", A(k,i));
            A_T(i,k) = A(k,i); 
            //printf("%5.2f\n", A_T(i,k));      
        }
    }
    printf("the transpose matrix x' is:\n");
    print_matrix(A_T, M, K);

    // |x'|^2 (x' is a M * K matrix)
    for (int i = 0; i < M; i++){
        for (int k = 0; k < K; k++){
            C(i,0) += A_T(i,k) * A_T(i,k); 
        }
    }
    printf("|x'|^2 (M * 1) is:\n");
    print_matrix(C, M, N);

    // repmat: M * 1 -> M * N (copy |x'|^2 n times)
    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            C(i,j) = C(i,0);
        }
    }
    printf("stretch matrix |x'|^2 to M * N:\n");
    print_matrix(C, M, N);

    double arr[N] = {0}; 
    // |y|^2 (y is a K * N matrix)
    for (int k = 0; k < K; k++){
        for (int j = 0; j < N; j++){
            arr[j] += B(k,j) * B(k,j);
        }
    }

    // print |y|^2 (1 * N)
    printf("|y|^2 (1 * N) is:\n");
    for (int j = 0; j < N; j++){
        printf("%5.2f ", arr[j]);
    }
    printf("\n");
    printf("\n");

    // repmat: 1 * N -> M * N (copy |y|^2 m times)
    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            C(i,j) += arr[j];
        }
    }
    printf("stretch matrix |y'|^2 to M * N and add with stretched matrix |x'|^2 (M * N):\n");
    print_matrix(C, M, N);

    // - 2 * x' * y (x -> the transpose of x': M * K)
    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            for (int k = 0; k < K; k++){
                C(i,j) -= 2 * A_T(i,k) * B(k,j);
            }
        }
    }

    // euclidean distance = |x - y|^2 = |x|^2 + |y|^2 - 2 * x * y
    // get the square root 
    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            C(i,j) = sqrt(C(i,j));
        }
    }

    // print the distance matrix
    printf("The distance matrix is:\n");
    print_matrix(C, M, N);

    free(A);free(B);free(C);free(A_T);
    return 0;
}