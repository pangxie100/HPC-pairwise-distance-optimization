BINARY_NAME = dgemm_x86
CC			= /opt/rh/devtoolset-7/root/usr/bin/gcc
CFLAGS		= -O3 -march=native -w
MKLPATH		= /home/yzhai015/intel/mkl
LDFLAGS		= -L$(MKLPATH)/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -DMKL_ILP64 -m64 -lmkl_avx512
INCFLAGS	= -I$(MKLPATH)/include


SRC			= $(wildcard *.c)
build : $(BINARY_NAME)

$(BINARY_NAME): $(SRC)
	$(CC) $(CFLAGS) $(LDFLAGS) $(INCFLAGS) $(SRC) -o $(BINARY_NAME)

clean:
	rm $(BINARY_NAME)
