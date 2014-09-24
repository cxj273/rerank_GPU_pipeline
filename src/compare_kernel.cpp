#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <vector>
#include <string>
#include <cassert>
#include <cmath>
#include <errno.h>

#define MINIMUM (0.0001)

using namespace std;


int main(int argc, char* argv[])
{
    if(argc != 4)
    {
        fprintf(stderr, "Usage: ./compare_kernel kernel1 kernel2 size\n");
        return 1;
    }

    FILE* kernel1_file = fopen(argv[1], "r");
    FILE* kernel2_file = fopen(argv[2], "r");
    int kernel_size = atoi(argv[3]);

    float* kernel1 = (float*) malloc(sizeof(float) * kernel_size * kernel_size);
    assert(kernel1 != NULL);
    float* kernel2 = (float*) malloc(sizeof(float) * kernel_size * kernel_size);
    assert(kernel2 != NULL);

    int idx;
    float value;
    int pos = 0;
    fprintf(stderr, "Start loading kernel1\n");
    while(fscanf(kernel1_file, "%d:%f", &idx, &value) != EOF)
    {
        if(idx != 0)
        {
            assert(pos < kernel_size * kernel_size);
            kernel1[pos] = value;
            pos ++;
        }
    }

    assert(pos == kernel_size * kernel_size);
    fprintf(stderr, "Loading kernel1 Done\n");
    fclose(kernel1_file);

    pos = 0;
    fprintf(stderr, "Start loading kernel2\n");
    while(fscanf(kernel2_file, "%d:%f", &idx, &value) != EOF)
    {
        if(idx != 0)
        {
            assert(pos < kernel_size * kernel_size);
            kernel2[pos] = value;
            pos ++;
        }
    }

    assert(pos == kernel_size * kernel_size);
    fprintf(stderr, "Loading kernel2 Done\n");
    fclose(kernel2_file);

    for(int r = 0; r < kernel_size; ++r)
    {
        for(int c = 0; c < kernel_size; ++c)
        {
            float diff1 = kernel1[r * kernel_size + c] - kernel1[c * kernel_size + r];
            float diff2 = kernel2[r * kernel_size + c] - kernel2[c * kernel_size + r];
            float diff3 = kernel1[r * kernel_size + c] - kernel2[r * kernel_size + c];
            if(!(diff1 < MINIMUM && diff1 > - MINIMUM))
            {
                printf("diff1: %f\n", diff1);
            }
            if(!(diff2 < MINIMUM && diff2 > - MINIMUM))
            {
                printf("diff2: %f\n", diff2);
            }
            if(!(diff3 < MINIMUM && diff3 > - MINIMUM))
            {
                printf("diff3: %f\n", diff3);
            }
        }
    }

    free(kernel1);
    free(kernel2);

    return 0;
}
