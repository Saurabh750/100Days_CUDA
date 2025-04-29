#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
do {                                                                             \
    cudaError_t err = call;                                                      \
    if (err != cudaSuccess) {                                                    \
        fprintf(stderr, "CUDA error in file '%s' at line %d: %s.\n",             \
                __FILE__, __LINE__, cudaGetErrorString(err));                    \
        exit(EXIT_FAILURE);                                                      \
    }                                                                            \
} while (0)

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

    // out of memory error: I try to allocate 373GB on device
    // size_t hugeSize = (size_t)373 * 1024 * 1024 * 1024;
    // CUDA_CHECK(cudaMalloc((void**)&d_a, hugeSize * sizeof(int)));

    CUDA_CHECK(cudaMalloc((void**)&d_a, arraySize * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_b, arraySize * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_c, arraySize * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, arraySize * sizeof(int), cudaMemcpyHostToDevice));

    addKernel<<<1, arraySize>>>(d_c, d_a, d_b);

    // Always check after kernel launch
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(c, d_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost));

    printf("int size: %zu ; bool size: %zu\n",sizeof(int), sizeof(bool));

    printf("Result: ");
    for (int i = 0; i < arraySize; ++i)
        printf("%d ", c[i]);
    printf("\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
