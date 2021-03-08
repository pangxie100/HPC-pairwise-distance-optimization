#include <stdio.h>
#include <stdlib.h>
#include "util.h"
#include <time.h>
#include <sys/time.h>

void zero_initialize_matrix_float(float *A, int M, int N){
    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            A[i + j * M] = 0.0f;
        }
    }
}

// col-major
void sequential_initialize_matrix_float(float *A, int M, int N){
    float num = 1.0f;
    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            A[i + j * M] = num;
            num++;
        }
    }
}

// col-major
void print_matrix_float(float *A, int M, int N){
    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            printf("%5.2f ", A[i + j * M]);    
        }
        printf("\n");    
    }
    printf("\n"); 
}

// col-major
void print_array_float(float *A, int N){
    for (int i = 0 ; i < N; i++){
        printf("%f ", A[i]);
    }
    printf("\n");
    printf("\n");
}

// col-major
void random_initialize_matrix_float(float *A, int M, int N){
    srand(time(NULL));  // from time.h
    int i, j;
    for (i = 0; i < M; i++)
    {
        for (j = 0; j < N; j++)
        {
            A[i + j * M] = (float)(rand() % 100) + 0.01 * (rand() % 100);
            if (rand() % 2 == 0)
            {
                A[i + j * M] *= -1.0;
            }
        }
    }
}

void zero_initialize_matrix(double *A, int M, int N){
    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            A[i + j * M] = 0.;
        }
    }
}

double get_sec(){
    struct timeval time;
    gettimeofday(&time, NULL); // from sys/time.h
    return (time.tv_sec + 1e-6 * time.tv_usec); // micro second, 1s = 1 * 10^6 micro seconds
}

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
void random_initialize_matrix(double *A, int M, int N){
    srand(time(NULL));  // from time.h
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