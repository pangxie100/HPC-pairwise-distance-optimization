#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "utils.h"
#include "mkl.h"
//#define verbose 1

#define MYDGEMM dgemm_asm

int main(int argc, char *argv[]){
    if (argc != 2) {
        printf("Please select a kernel (range 0 - 19, here 0 is for Intel MKL).\n");
        exit(-1);
    }

    /*
    int SIZE[30]={100,200,300,400,500,600,700,800,900,1000,1100,\
                1200,1300,1400,1500,1600,1700,1800,1900,2000,\
                2100,2200,2300,2400,2500,2600,2700,2800,2900,3000};//testing 100-3000 square matrices
    */
    // my test style
    // test m = 192 * size, n = 96 * size, k = 384 * size 
    /*
    int SIZE[30] = {1,2,3,4,5,6,7,8,9,10,\
                    11,12,13,14,15,16,17,18,19,20,\
                    21,22,23,24,25,26,27,28,29,30};
    */
    int SIZE[20] = {1,2,3,4,5,6,7,8,9,10,\
                    11,12,13,14,15,16,17,18,19,20};

    int kernel_num=atoi(argv[1]);
    if (kernel_num<0||kernel_num>19) {
        printf("Please enter a valid kernel number (0-19).\n");
        exit(-2);
    }
    //int m, n, k, max_size = 3000;
    int m, n, k, max_m = 192 * 20, max_n = 96 * 20, max_k = 384 * 20;
    int n_count, N=3, upper_limit;
    if (kernel_num <= 4 && kernel_num != 0) upper_limit = 10;
    else upper_limit = 20;

    double *A=NULL,*B=NULL,*C=NULL,*C_ref=NULL;

    //double alpha = 2.0, beta = -1.5; // two arbitary input parameters
    double alpha = 1.0, beta = 1.0;

    double t0,t1;

    /*
    A=(double *)malloc(sizeof(double)*max_size*max_size);
    B=(double *)malloc(sizeof(double)*max_size*max_size);
    C=(double *)malloc(sizeof(double)*max_size*max_size);
    C_ref=(double *)malloc(sizeof(double)*max_size*max_size);
    if (A == NULL || B == NULL || C == NULL || C_ref == NULL){
        printf("Malloc function failed to allocate space\n");
        exit(-3);
    }

    randomize_matrix(A,max_size,max_size);
    randomize_matrix(B,max_size,max_size);
    randomize_matrix(C,max_size,max_size);
 
    // test
    all_one_matrix(A,max_size,max_size);
    all_one_matrix(B,max_size,max_size);
    all_one_matrix(C,max_size,max_size);

    copy_matrix(C,C_ref,max_size*max_size);
    */

    A=(double *)malloc(sizeof(double) * max_m * max_k);
    B=(double *)malloc(sizeof(double) * max_k * max_n);
    C=(double *)malloc(sizeof(double) * max_m * max_n);
    C_ref=(double *)malloc(sizeof(double) * max_m * max_n);
    if (A == NULL || B == NULL || C == NULL || C_ref == NULL){
        printf("Malloc function failed to allocate space\n");
        exit(-3);
    }

    randomize_matrix(A, max_m, max_k);
    randomize_matrix(B, max_k, max_n);
    randomize_matrix(C, max_m, max_n);
    
    /*
    // test
    all_one_matrix(A, max_m, max_k);
    all_one_matrix(B, max_k, max_n);
    all_one_matrix(C, max_m, max_n);
    //*/

    copy_matrix(C, C_ref, max_m * max_n);

    for (int i_count=0;i_count<upper_limit;i_count++){

        //m=n=k=SIZE[i_count];
        //printf("\nM=N=K=%d:\n",m);

        // my style
        ///*
        m = 192 * SIZE[i_count];
        n = 96 * SIZE[i_count];
        k = 384 * SIZE[i_count];
        //*/

        /*
        // small test case
        m = 24;
        n = 8;
        k = 32;
        //*/

        /*
        m = 24 * 3;
        n = 8 * 3;
        k = 32 * 3;
        //*/
        printf("\nM = %d, ", m);
        printf("N = %d, ", n);
        printf("K = %d\n", k);
        if (kernel_num != 0){//not an MKL implementation
            test_kernel(kernel_num,m,n,k,alpha,A,B,beta,C);
            cblas_dgemm(CblasColMajor, CblasNoTrans,CblasNoTrans,m,n,k,alpha,A,m,B,k,beta,C_ref,m);
            
            // add
            //printf("C[0][0] = %5.2f\n",C[0]);

            if (!verify_matrix(C_ref,C,m*n)) {
                printf("Failed to pass the correctness verification against Intel MKL. Exited.\n");
                print_matrix(C, m, n);
                exit(-3);
            }
        }
        t0=get_sec();
        for (n_count=0;n_count<N;n_count++){
            test_kernel(kernel_num,m,n,k,alpha,A,B,beta,C);
        }
        t1=get_sec();
        printf("Average elasped time: %f second, performance: %f GFLOPS.\n", (t1-t0)/N,2.*1e-9*N*m*n*k/(t1-t0));
        copy_matrix(C_ref,C,m*n);//sync C with Intel MKl to prepare for the next run
    }
    free(A);free(B);free(C);free(C_ref);
    return 0;
}