#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

int N = 256;
#define BLOCK_SIZE 16

void load_input_data(double *A, double *B, double *x, int n){
    FILE *f;
    f = fopen("input.txt", "r");
    if (f == NULL) {
        printf("Error opening input file.\n");
        exit(1);
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n-1; j++) {
            fscanf(f, "%lf", &A[i*n + j]);
        }
        fscanf(f, "%lf", &B[i]);
        x[i] = 0.0;
    }
    fclose(f);
}

__global__ void jacobi_kernel(double *u, double *f, double *u_new, double h, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < N-1 && j > 0 && j < N-1)
    {
        int index = i*N + j;
        u_new[index] = 0.25 * (u[index-N] + u[index-1] + u[index+1] + u[index+N] - h*h*f[index]);
    }
}

int jacobi_algorithm(int s){
    // size
    N = s;
    // generating matrix
    char command_str[300];
    sprintf(command_str, "python generate_matrix.py %d input.txt", s);
    system(command_str);

    double *u, *f, *u_new;
    double h = 1.0/(N-1);
    double tol = 1e-6;
    int maxiter = 10000;
    int iter = 0;
    size_t size = N*N*sizeof(double);
    cudaEvent_t start, stop;
    float elapsedTime;

    cudaMallocManaged(&u, size);
    cudaMallocManaged(&f, size);
    cudaMallocManaged(&u_new, size);

    // Initialize u and f
    load_input_data(f, u, u_new, N);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N+BLOCK_SIZE-1)/BLOCK_SIZE, (N+BLOCK_SIZE-1)/BLOCK_SIZE);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    // Jacobi iteration
    while (iter < maxiter)
    {
        jacobi_kernel<<<grid, block>>>(u, f, u_new, h, N);

        cudaDeviceSynchronize();

        // Check for convergence
        double diff_norm = 0.0;
        for (int i = 1; i < N-1; i++)
        {
            for (int j = 1; j < N-1; j++)
            {
                int index = i*N + j;
                diff_norm += pow(u_new[index] - u[index], 2);
            }
        }
        diff_norm = sqrt(diff_norm)/(N-2)/(N-2);

        if (diff_norm < tol)
        {
            break;
        }

        // Swap u and u_new
        double *temp = u;
        u = u_new;
        u_new = temp;

        iter++;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time: %f ms\n", elapsedTime);

    FILE *fp;
    fp = fopen("cuda_results.txt", "a");
    if (fp == NULL){
        printf("Error in opening file !!\n");
        return -1;
    }
    fprintf(fp, "cuda %d %f\n", s, elapsedTime);
    fclose(fp);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Free memory
    cudaFree(u);
    cudaFree(f);
    cudaFree(u_new);

    return 0;
}

int main(){
    for(int s = 100; s < 1050; s += 50){
        jacobi_algorithm(s);
    }
}