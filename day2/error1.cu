#include <stdio.h>
#include <cuda_runtime.h>

__global__ void addKernel(int *c, const int *a, const int *b) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main() {
    const int arraySize = 5;
    const int a[arraySize] = {1, 2, 3, 4, 5};
    const int b[arraySize] = {10, 20, 30, 40, 50};
    int c[arraySize] = {0};

    int *d_a, *d_b, *d_c;
    
    // Manual error checking
    cudaError_t err;

    err = cudaMalloc((void**)&d_a, arraySize * sizeof(int));
    if (err != cudaSuccess) {
        printf("CUDA malloc failed for d_a: %s\n", cudaGetErrorString(err));
        return -1;
    }

    err = cudaMalloc((void**)&d_b, arraySize * sizeof(int));
    if (err != cudaSuccess) {
        printf("CUDA malloc failed for d_b: %s\n", cudaGetErrorString(err));
        return -1;
    }

    err = cudaMalloc((void**)&d_c, arraySize * sizeof(int));
    if (err != cudaSuccess) {
        printf("CUDA malloc failed for d_c: %s\n", cudaGetErrorString(err));
        return -1;
    }

    err = cudaMemcpy(d_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CUDA memcpy failed for d_a: %s\n", cudaGetErrorString(err));
        return -1;
    }

    err = cudaMemcpy(d_b, b, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CUDA memcpy failed for d_b: %s\n", cudaGetErrorString(err));
        return -1;
    }

    addKernel<<<1, arraySize>>>(d_c, d_a, d_b);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    err = cudaMemcpy(c, d_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("CUDA memcpy failed for d_c: %s\n", cudaGetErrorString(err));
        return -1;
    }

    printf("Result: ");
    for (int i = 0; i < arraySize; ++i)
        printf("%d ", c[i]);
    printf("\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
