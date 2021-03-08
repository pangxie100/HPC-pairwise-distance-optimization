# HPC-pairwise-distance-optimization

## Overview
This is a col-major pairwise distance optimization prototype implementaion. The main part to optimize is matrix multiplication.
The benchmark of the pairwise distance computation is completed by applying MKL libraay implemented by the Intel.

## Syntax
```
Input   
        int M
        int N
        int K
        double *A : K - by - M
        double *B : K - by - N
        double *C : M - by - N

Output
        C : M - by - N
```

## Prerequisite
* Unix-like system (Mac OS or Linux - CentOS/RedHat/Ubuntu)

## How to run
```
$ https://github.com/pangxie100/HPC-pairwise-distance-optimization.git
cd HPC-pairwise-distance-optimization
```

*Mac OS*
Run the following command:
```
$ ./run_MacOS.sh
```
If you get "Permission denied" error, please run this command first:
```
$ sudo chmod +x run_MacOS.sh
```

*Linux OS*
Run the following command:
```
$ ./run_Linux.sh
```
If you get "Permission denied" error, please run this command first:
```
$ sudo chmod +x run_Linux.sh
```

In "run_Linux.sh" and "run_MacOS.sh",
both
```
$ cd mkldnn
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`
cd ..
cd build
```
and 
```
$ cd mkldnn
$ cd lib
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`
cd ..
cd ..
cd build
```
are ok for running the program here.

## Run it with different matrix size
After running the above command, it will automatically run a default test case. Then, each time you set a new size matrix or want to run it again, you can just run this:
```
$ ./recompile_run.sh
```
If you get "Permission denied" error, please run this command first:
```
$ sudo chmod +x recompile_run.sh
```

## Test its average performance
Run the following command to test the performance of double type matrix:
```
$ ./test_perf.sh
```

Run the following command to test the performance of float type matrix:
```
$ ./test_float_perf.sh
```

The default run times of pairwise distance computing is 100.
The default size of test matrix in these ".sh" file is 10000 20000 3

## Performance profiling
Performance profiling can see each part's elapsed time in pairwise distance computing.
For double type matrix computing, uncomment "//#define TIME_COUNT 1" in "pw_dist_mkl_func.c" and run:
```
$ ./test_perf.sh
```

For float type matrix computing, uncomment "//#define TIME_COUNT 1" in "pdist_mkl_float_func.c" and run:
```
$ ./test_float_perf.sh
``` 
