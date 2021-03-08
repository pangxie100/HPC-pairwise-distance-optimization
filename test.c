#include <stdio.h>
#include <stdlib.h>

#include "util.h"
#include "pw_dist_mkl_func.h"
#define times 100

int main(int argc, char* argv[])
{
    int M = 3;
    int N = 2;
    int K = 4;
    if (argc != 4) 
    {
        printf("./my_pw_dist_perf [M] [N] [K]\n");
        printf("Wrong input, exit.\n");
        exit(-1);
    }

    M = atoi(argv[1]); // argv[0] is program name
    N = atoi(argv[2]);
    K = atoi(argv[3]);

    if (M == 0 || N == 0 || K == 0) {
        printf("Matrix dimension cannot be ZERO, exit.\n");
        exit(-2);
    }

    double *A, *B, *C;
    double t0, t1, t_total = 0.;

    A = (double *)malloc(sizeof(double) * K * M);
    B = (double *)malloc(sizeof(double) * K * N);
    C = (double *)malloc(sizeof(double) * M * N);

    // Initialization
    sequential_initialize_matrix(A, K, M);
    sequential_initialize_matrix(B, K, N);
    zero_initialize_matrix(C, M, N);

    /*
    printf("matrix A is initialized as:\n");
    print_matrix(A, K, M);
    print_array(A, K * M);

    printf("matrix B is initialized as:\n");
    print_matrix(B, K, N);
    print_array(B, K * N);

    printf("matrix C is initialized as:\n");
    print_matrix(C, M, N);
    print_array(C, M * N);
    */
    printf("Start to compute pairwise distance\n");
    printf("Matrix type: double\n");
    printf("Matrix size:\n");
    printf("A: %d x %d\nB: %d x %d\nC: %d x %d\n", K, M, K, N, M, N);
    printf("We provide the same output as pdist2(A', B', 'euclidean') in MATLAB\n");
    for (int i = 0; i < times; i++)
    {
	    t0 = get_sec();
        pw_dist_mkl(A, B, C, M, N, K);
        t1 = get_sec();
        t_total += (t1 - t0);
        zero_initialize_matrix(C, M, N);
        //printf("after setting zero, matrix C is:\n");
        //print_matrix(C, M, N);
    }
    printf("pw_dist_mkl : Average elapsed time: %8.6fs, Perf: %8.6f GFLOPS\n", t_total / times, \
        2 * (M / 1000.) * (N / 1000.) * (K / 1000.) * times / t_total);

    free(A);free(B);free(C);
    return 0;
}