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
    if(argc != 3)
    {
        fprintf(stderr, "Usage: ./check_kernel kernel_file size\n");
        return 1;
    }

    FILE* kernel_file = fopen(argv[1], "r");
    int kernel_size = atoi(argv[2]);

    float* kernel = (float*) malloc(sizeof(float) * kernel_size * kernel_size);
    assert(kernel != NULL);
    int idx;
    float value;
    int pos = 0;
    fprintf(stderr, "Start loading kernel\n");
    while(fscanf(kernel_file, "%d:%f", &idx, &value) != EOF)
    {
        if(idx != 0)
        {
            assert(pos < kernel_size * kernel_size);
            kernel[pos] = value;
            pos ++;
        }
    }

    assert(pos == kernel_size * kernel_size);
    fprintf(stderr, "Loading kernel Done\n");
    fclose(kernel_file);

    for(int r = 0; r < kernel_size; ++r)
    {
        for(int c = 0; c < kernel_size; ++c)
        {
            float diff = kernel[r * kernel_size + c] - kernel[c * kernel_size + r];
            assert(diff < MINIMUM && diff > - MINIMUM);
        }
    }

    free(kernel);

    return 0;
}
