#include <cuda_runtime.h>
#include <cublas_v2.h>
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
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/wait.h>
#include "KR.h"
#include "crossValidationKR.h"

#define CHUNK_SIZE (1000)

using namespace std;

// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
void gpu_blas_mmul(const float* A, const float* B, float* C, const int m, const int k, const int n)
{
    int lda = m, ldb = k, ldc = m;
    const float alf = 1;
    const float bet = 0;
    const float* alpha = &alf;
    const float* beta = &bet;
    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    // Do the actual multiplication
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    // Destroy the handle
    cublasDestroy(handle);
}

//Print matrix A(nr_rows_A, nr_cols_A) storage in column-major format
void print_matrix(const float* A, int nr_rows_A, int nr_cols_A)
{
    for(int i = 0; i < nr_rows_A; ++i)
    {
        for(int j = 0; j < nr_cols_A; ++j)
        {
            std::cout << A[j * nr_rows_A + i] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char* argv[])
{
    if(argc != 11)
    {
        fprintf(stderr, "Usage: ./rerank_train original_feat original_kernel top_list num_top EKn folds_n kernel_type label_file model_file gpu_id\n");
        return 1;
    }


    /**READ ORIGINAL FEATURE**/
    fprintf(stderr, "reading original feature %s\n", argv[1]);
    FILE* feat_in = fopen(argv[1], "r");
    int feat_dims = 0, num_train = 0;
    fread(&num_train, sizeof(int), 1, feat_in);
    fread(&feat_dims, sizeof(int), 1, feat_in);
    float* feat = (float*)malloc(sizeof(float) * (size_t)feat_dims * (size_t)num_train);
    assert(feat != NULL);
    fread(feat, sizeof(float), (size_t)feat_dims * (size_t)num_train, feat_in);
    fclose(feat_in);
    fprintf(stderr, "done reading original feature %s\n", argv[1]);

    int gpu_id;
    sscanf(argv[10], "%d", &gpu_id);
    if(cudaSetDevice(gpu_id) != cudaSuccess)
    {
        fprintf(stderr, "Failed to set gpu_id\n");
        return 1;
    }

    int num_top = atoi(argv[4]);

    float* h_AxBT = NULL;
    float* h_BxBT = NULL;
    assert(num_top >= 0);
    if(num_top != 0)
    {
        /**READ TOP FEATURE**/
        FILE* top_list_in = fopen(argv[3], "r");
        float* top_feat = (float*) malloc(sizeof(float) * (size_t) feat_dims * (size_t) num_top);
        char top_path[512];
        int num_top_readed = 0;
        const char zcat_prefix[] = "zcat ";
        while(fscanf(top_list_in, "%s", top_path) != EOF)
        {
            FILE* zcat_in;
            char* zcat_cmd;
            zcat_cmd = (char*) malloc((size_t)(sizeof(zcat_prefix) + strlen(top_path) + 1));
            fprintf(stderr, "reading top feature %s\n", top_path);

            if(strcmp(top_path + strlen(top_path) - 3, ".gz") == 0)
            {
                sprintf(zcat_cmd, "%s%s", zcat_prefix, top_path);
                zcat_in = popen(zcat_cmd, "r");
            }
            else
            {
                zcat_in = fopen(top_path, "r");
            }


            int idx;
            float value;
            while(fscanf(zcat_in, "%d:%f", &idx, &value) != EOF)
            {
                top_feat[num_top_readed * feat_dims + idx - 1 ] = value;
            }
            free(zcat_cmd);
            fclose(zcat_in);
            num_top_readed++;
        }
        assert(num_top_readed == num_top);

        /**MATRIX MULTIPLICATION ON GPU**/
        cublasHandle_t cublas_handle;
        cublasCreate(&cublas_handle);
        //float* d_A;
        float* d_BT;
        //float* d_AxBT;
        float* d_BxBT;

        //cudaMalloc(&d_A, num_train * feat_dims * sizeof(float));
        cudaMalloc(&d_BT, num_top * feat_dims * sizeof(float));
        //cudaMalloc(&d_AxBT, num_train * num_top * sizeof(float));
        cudaMalloc(&d_BxBT, num_top * num_top * sizeof(float));
        //assert(d_A != NULL);
        assert(d_BT != NULL);
        //assert(d_AxBT != NULL);
        assert(d_BxBT != NULL);

        float* h_BT = top_feat;
        h_AxBT = (float*) malloc(num_train * num_top * sizeof(float));
        h_BxBT = (float*) malloc(num_top * num_top * sizeof(float));
        cudaMemcpy(d_BT, h_BT, num_top * feat_dims * sizeof(float), cudaMemcpyHostToDevice);

        //C=alpha*op(A)op(B) + beta * C
        //C(m,n) A(m,k) B(k,n)
        //op ( A ) = A if  transa == CUBLAS_OP_N
        //A T if  transa == CUBLAS_OP_T
        //A H if  transa == CUBLAS_OP_C
        //!!A, B and C are column-major

        //cublasStatus_t cublasSgemm(
        //        cublasHandle_t handle,
        //        cublasOperation_t transa,
        //        cublasOperation_t transb,
        //        int m,
        //        int n,
        //        int k,
        //        const float *alpha,
        //        const float *A,
        //        int lda,
        //        const float *B,
        //        int ldb,
        //        const float *beta,
        //        float *C,
        //        int ldc)

        const float alf = 1;
        const float bet = 0;
        const float* alpha = &alf;
        const float* beta = &bet;
        float* A, * B, * C;
        int m, n, k;
        int lda, ldb, ldc;

        //compute BxBT
        m = num_top;
        n = num_top;
        k = feat_dims;
        lda = feat_dims;
        ldb = feat_dims;
        ldc = num_top;
        A = d_BT;
        B = d_BT;
        C = d_BxBT;

        cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        cudaDeviceSynchronize();
        cudaMemcpy(h_BxBT, d_BxBT, num_top * num_top * sizeof(float), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        cudaFree(d_BxBT);
        cudaDeviceSynchronize();
        //print_matrix(h_BxBT, num_top, num_top);

        //compute AxBT
        //actually compute BxAT to convert the column-major to row-major
        //divide A into chunks (Ac) by row

        float* d_AcT;
        cudaMalloc(&d_AcT, CHUNK_SIZE * feat_dims * sizeof(float));
        assert(d_AcT != NULL);

        float* d_BxAcT;
        cudaMalloc(&d_BxAcT, num_top * CHUNK_SIZE * sizeof(float));
        assert(d_BxAcT != NULL);

        int start_row = 0;
        while(start_row < num_train)
        {
            //Ac(chunk_size x feat_dims)
            int chunk_size = CHUNK_SIZE;
            if(start_row + chunk_size >= num_train)
            {
                chunk_size = num_train - start_row;
            }

            fprintf(stderr, "%d, %d, %d\n", start_row, chunk_size, feat_dims);

            float* h_AcT = feat + start_row * feat_dims;
            float* h_BxAcT = h_AxBT + start_row * num_top;

            cudaMemcpy(d_AcT, h_AcT, chunk_size * feat_dims * sizeof(float), cudaMemcpyHostToDevice);
            //col-major view: d_AcT (feat_dims x chunk_size) d_BT (feat_dims x num_top) d_BxAcT (num_top x chunk_size)
            //row-major view: d_BxAcT (chunck_size x num_top)

            //compute (BT)T(m,k)xAcT(k,n)
            m = num_top;
            n = chunk_size;
            k = feat_dims;

            lda = feat_dims;
            ldb = feat_dims;
            ldc = num_top;
            A = d_BT;
            B = d_AcT;
            C = d_BxAcT;

            cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

            cudaDeviceSynchronize();
            cudaMemcpy(h_BxAcT, d_BxAcT, chunk_size * num_top * sizeof(float), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            /*
                    for( int i = 0; i < chunk_size; ++i)
                    {
                        for( int j = 0; j < num_top; ++j)
                        {
                            h_BxAcT[i * num_top + j] = 0.0;
                            for( int d = 0; d < feat_dims; ++d)
                            {
                                h_BxAcT[i * num_top + j] += h_BT[j * feat_dims + d] * h_AcT[i * feat_dims + d];
                                //fprintf(stderr, " %f", h_BT[j * feat_dims + d]);
                            }
                        }
                    }
            */

            //print_matrix(h_BxAcT, num_top, chunk_size);
            //getchar();
            start_row += CHUNK_SIZE;
        }

        cudaFree(d_AcT);
        cudaFree(d_BxAcT);
        cudaFree(d_BT);
        cublasDestroy(cublas_handle);
        free(feat);
        free(top_feat);
        fclose(top_list_in);
    }

    /**MODIFY KERNEL**/
    //K =>  K       |   AxBT
    //      (AxBT)T |   BxBT
    //A(num_train x feat_dim)
    //B(num_top x feat_dim)


    int dev_dist_size = num_train + num_top;
    float* dev_dist = (float*) malloc(sizeof(float) * (size_t)dev_dist_size * (size_t) dev_dist_size);
    assert(dev_dist != NULL);
    size_t kernel_pos = 0;

    //Add AxBT
    fprintf(stderr, "Start to modify kernel\n");
    FILE* original_kernel_file = fopen(argv[2], "r");
    char* original_kernel_line = NULL;
    size_t len = 0;
    ssize_t read;

    for(int line_count = 0; line_count < num_train; ++ line_count)
    {
        int idx;
        float value;

        for(int j = 0; j <= num_train; ++j)
        {
            fscanf(original_kernel_file, "%d:%f", &idx, &value);

            if(idx != 0)
            {
                dev_dist[kernel_pos] = value;
                kernel_pos ++;
            }
        }

        for(int top_i = 0; top_i < num_top; ++ top_i)
        {
            idx = num_train + top_i + 1;
            value = h_AxBT[line_count * num_top + top_i];
            dev_dist[kernel_pos] = value;
            kernel_pos ++;
        }
    }

    assert(kernel_pos == num_train * (num_train + num_top));

    if(original_kernel_line)
    {
        free(original_kernel_line);
    }

    //Add (AxBT)T and BxBT
    for(int top_i = 0; top_i < num_top; ++ top_i)
    {
        for(int train_j = 0; train_j < num_train + num_top; ++ train_j)
        {
            int idx = train_j + 1;
            float value;
            if(train_j < num_train)
            {
                value = h_AxBT[train_j * num_top + top_i];
            }
            else
            {
                value = h_BxBT[top_i * num_top + train_j - num_train];
            }
            dev_dist[kernel_pos] = value;
            kernel_pos ++;
        }
    }

    assert(kernel_pos == (num_train + num_top) * (num_train + num_top));
    fclose(original_kernel_file);
    free(h_BxBT);
    free(h_AxBT);
    fprintf(stderr, "Modifying kernel done\n");

    char* EKIDX = argv[5];
    int nfolds  = atoi(argv[6]);
    int kernel_type = atoi(argv[7]); //6
    int dist_size = num_top + num_train;
    char* label_file = argv[8];
    char* model_save_file = argv[9];

    /**KERNEL REGRESSION AND CROSS VALIDATION**/
    fprintf(stderr, "EKIDX: %s\n", EKIDX);
    fprintf(stderr, "nflods: %d\n", nfolds);
    fprintf(stderr, "kernel_type: %d\n", kernel_type);
    fprintf(stderr, "dev_dist_size: %d\n", dist_size);
    fprintf(stderr, "dev_dist: %d\n", dev_dist);
    fprintf(stderr, "label_file: %s\n", label_file);
    fprintf(stderr, "model_save_file: %s\n", model_save_file);
    fprintf(stderr, "gpu_id: %d\n", gpu_id);
    crossValidationKR(EKIDX, nfolds, kernel_type, dev_dist_size, dev_dist, label_file, model_save_file, gpu_id);
    free(dev_dist);

    return 0;
}
