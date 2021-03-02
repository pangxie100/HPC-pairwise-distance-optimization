#include <stdio.h>
#include <stdlib.h>
#include "util.h"
#include <time.h>

// col-major
void sequential_initialize_matrix(double *A, int M, int N){
    double num = 1.0;
    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            A[i + j * M] = num;
            num++;
        }
    }
}

// col-major
void print_matrix(double *A, int M, int N){
    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            printf("%5.2f ", A[i + j * M]);    
        }
        printf("\n");    
    }
    printf("\n"); 
}

// col-major
void print_array(double *A, int N){
    for (int i = 0 ; i < N; i++){
        printf("%f ", A[i]);
    }
    printf("\n");
    printf("\n");
}

// col-major
void random_initialize_matrix(double *A, int M, int N)
{
    srand(time(NULL));
    int i, j;
    for (i = 0; i < M; i++)
    {
        for (j = 0; j < N; j++)
        {
            A[i + j * M] = (double)(rand() % 100) + 0.01 * (rand() % 100);
            if (rand() % 2 == 0)
            {
                A[i + j * M] *= -1.0;
            }
        }
    }
}