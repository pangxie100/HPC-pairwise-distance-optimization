#include <stdio.h>
#include <stdlib.h>
#include "util.h"
#include "mkl.h"
#include <math.h>
// this computation way is for column-major
#define A(i,j) A[(i)+(j)*LDA]
#define B(i,j) B[(i)+(j)*LDB]
#define B_T(i,j) B_T[(i)+(j)*LDB_T]
#define C(i,j) C[(i)+(j)*LDC]
#define M 3
#define N 2
#define K 4

// Version 2: base on MATLAB, x * y', thus, B_T. For A and B, each row is one observation.(one point in a k-dimension space).
int main(int argc, char *argv[]){
    double *A, *B, *C, *B_T;
    int LDA = M, LDB = N, LDC = M, LDB_T = K;
    A = (double *)malloc(sizeof(double) * M * K);
    B = (double *)malloc(sizeof(double) * N * K);
    B_T = (double *)malloc(sizeof(double) * K * N);
    C = (double *)malloc(sizeof(double) * M * N);
    
    // Initialization
    // 1.Initialize X matrix
    double num = 1.0;
    printf("matrix x is initialized as:\n");
    /*
    // way 1:
    // [1  2  3  4]
    // [5  6  7  8]
    // [9 10 11 12]
    for (int i = 0; i < M; i++){
        for (int k = 0; k < K; k++){
            A(i,k) = num;
            //printf("%f ", A(i,k));
            num++;
        }
        //printf("\n");
    }
    //printf("\n");
    */
    // way 2:
    // [1 4 7 10]
    // [2 5 8 11]
    // [3 6 9 12]
    for (int k = 0; k < K; k++){
        for (int i = 0; i < M; i++){
            A(i,k) = num;
            //printf("%f ", A(i,k));
            num++;
        }
        //printf("\n");
    }
    //printf("\n");
    print_matrix(A, M, K);
    // 2.Initialize Y matrix
    num = 1.0;
    printf("matrix y is initialized as:\n");
    /*
    // way 1:
    // [1 2 3 4]
    // [5 6 7 8]
    for (int j = 0; j < N; j++){
        for (int k = 0; k < K; k++){
            B(j,k) = num;
            //printf("%f ", B(j,k));
            num++;
        }
        //printf("\n");
    }
    //printf("\n");
    */
    // way 2:
    // [1 3 5 7]
    // [2 4 6 8]
    for (int k = 0; k < K; k++){
        for (int j = 0; j < N; j++){
            B(j,k) = num;
            //printf("%f ", B(j,k));
            num++;
        }
        //printf("\n");
    }
    //printf("\n");
    print_matrix(B, N, K);
    //randomize_matrix(A,max_size,max_size);
    //randomize_matrix(B,max_size,max_size);
    
    // pairwise distance:
    // ||x-y||^2 = |x|^2 + |y|^2 - 2 * x * y

    // y (N * K) -> the transpose of y: y' (K * N)
    for (int j = 0; j < N; j++){
        for (int k = 0; k < K; k++){
            //printf("%5.2f ", A(j,k));
            B_T(k,j) = B(j,k); 
            //printf("%5.2f\n", A_T(k,j));      
        }
    }
    printf("the transpose matrix y' is:\n");
    print_matrix(B_T, K, N);

    // |x|^2 (x is a M * K matrix)
    for (int i = 0; i < M; i++){
        for (int k = 0; k < K; k++){
            C(i,0) += A(i,k) * A(i,k); 
        }
    }
    printf("|x|^2 (M * 1) is:\n");
    print_matrix(C, M, N);

    // repmat: M * 1 -> M * N (copy |x|^2 n times)
    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            C(i,j) = C(i,0);
        }
    }
    printf("stretch matrix |x|^2 to M * N:\n");
    print_matrix(C, M, N);

    double arr[N] = {0}; 
    // |y'|^2 (y' is a K * N matrix)
    for (int k = 0; k < K; k++){
        for (int j = 0; j < N; j++){
            arr[j] += B_T(k,j) * B_T(k,j);
        }
    }

    // print |y'|^2 (1 * N)
    printf("|y|^2 (1 * N) is:\n");
    for (int j = 0; j < N; j++){
        printf("%5.2f ", arr[j]);
    }
    printf("\n");
    printf("\n");

    // repmat: 1 * N -> M * N (copy |y'|^2 m times)
    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            C(i,j) += arr[j];
        }
    }
    printf("stretch matrix |y'|^2 to M * N and add with stretched matrix |x|^2 (M * N):\n");
    print_matrix(C, M, N);

    // - 2 * x * y' (y' -> the transpose of y: K * N)
    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            for (int k = 0; k < K; k++){
                C(i,j) -= 2 * A(i,k) * B_T(k,j);
            }
        }
    }

    // euclidean distance = |x - y|^2 = |x|^2 + |y'|'^2 - 2 * x * y'
    // get the square root 
    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            C(i,j) = sqrt(C(i,j));
        }
    }

    // print the distance matrix
    printf("The distance matrix is:\n");
    print_matrix(C, M, N);

    // test randomized matrix, result: fixed random
    double *R1;
    R1 = (double *)malloc(sizeof(double) * M * K);
    random_initialize_matrix(R1, M, K);
    printf("The Rand 1 matrix is:\n");
    print_matrix(R1, M, K);
    double *R2;
    R2 = (double *)malloc(sizeof(double) * M * K);
    random_initialize_matrix(R2, M, K);
    printf("The Rand 2 matrix is:\n");
    print_matrix(R2, M, K);
    free(R1);free(R2);

    free(A);free(B);free(C);free(B_T);
    return 0;
}