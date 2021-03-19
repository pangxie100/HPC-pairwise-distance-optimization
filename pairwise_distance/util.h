#ifndef _UTIL_H
#define _UTIL_H
//#include "sys/time.h"  // or #include <sys/time.h> in util.c

void zero_initialize_matrix_float(float *A, int M, int N);
void sequential_initialize_matrix_float(float *A, int M, int N);
void print_matrix_float(float *A, int M, int N);
void print_array_float(float *A, int N);
void random_initialize_matrix_float(float *A, int M, int N);
void zero_initialize_matrix(double *A, int M, int N);
double get_sec();
void sequential_initialize_matrix(double *A, int M, int N);
void print_matrix(double *A, int M, int N);
void print_array(double *A, int N);
void random_initialize_matrix(double *A, int M, int N);

#endif 