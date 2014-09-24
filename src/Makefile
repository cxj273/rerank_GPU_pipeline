CC = g++
NVCC = nvcc
CFLAGS = -O3 -msse3 -ffast-math -fomit-frame-pointer -Wall -fopenmp
NVCC_FLAGS = -O3
INDENT = astyle

INDENT_OPTS=-n --style=allman --indent-classes --indent-switches --indent-cases --indent-namespaces --indent-labels -Y --min-conditional-indent=2 --max-instatement-indent=40 --pad-oper --unpad-paren --mode=c --add-brackets --break-closing-brackets --align-pointer=type

INCLUDES=-I./lib/cula/include
LIBPATH32=-L./lib/cula/lib
LIBPATH64=-L./lib/cula/lib64
LIBS=-lcula_lapack -lcublas -liomp5

all: rerank_train check_kernel compare_kernel

rerank_train: rerank_train.o KR.o crossValidationKR.o
	$(NVCC) -o rerank_train rerank_train.o KR.o crossValidationKR.o -O3 $(INCLUDES) $(LIBPATH64) $(LIBS)

rerank_train.o: rerank_train.cu
	$(NVCC) rerank_train.cu -c -O3 $(INCLUDES) $(LIBPATH64) $(LIBS)

KR.o: KR.cu KR.h 
	$(NVCC) -c KR.cu -O3 $(INCLUDES) $(LIBPATH64) $(LIBS)

crossValidationKR.o: crossValidationKR.cu crossValidationKR.h KR.h 
	$(NVCC) -c crossValidationKR.cu -O3 $(INCLUDES) $(LIBPATH64) $(LIBS)

check_kernel: check_kernel.cpp
	$(CC) check_kernel.cpp -O3 -o check_kernel

compare_kernel: compare_kernel.cpp
	$(CC) compare_kernel.cpp -O3 -o compare_kernel

clean:
	rm rerank_train check_kernel compare_kernel *.o

indent:
	$(INDENT) $(INDENT_OPTS) *.cu *.cpp
