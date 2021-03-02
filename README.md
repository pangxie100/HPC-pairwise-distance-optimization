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
```
$ ./run_MacOS.sh
```

*Linux OS*
```
$ ./run_Linux.sh
```

After running the above command, it will automatically run a default case. Then, each time you set a new size matrix or want to run it again, you can just run this:
```
$ recompile_run.sh
```

