/////////////////////////////////////////////////////////////////
//
//      Using cudaMalloc !
//      Type  Time(%)      Time     Calls       Avg       Min       Max  Name
//      GPU activities:   
//              50.23%  1.4014ms         3  467.12us  445.31us  504.19us  [CUDA memcpy DtoH]
//              48.08%  1.3412ms         3  447.08us  404.61us  486.21us  [CUDA memcpy HtoD]
//              1.69%  47.167us         1  47.167us  47.167us  47.167us  addVectorGPU(float*, float*, float*, int)
//
//      cudaMalloc kernel is faster than cudaManaged by 1.45sec !!
//
/////////////////////////////////////////////////////////////////

#include <iostream>
using namespace std;

__global__ void addVectorGPU(float* a, float* b,float* c, int n) {
    int step = gridDim.x * blockDim.x;
    int start = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = start;i < n;i += step) {
        c[i] = a[i] + b[i];
    }
}

int main() {

    int N = 1<<20;
    float *a = new float[N], *b = new float[N], *c = new float[N];

    for(int i = 0;i < N;i++){
        a[i] = 1.0f;b[i] = 2.0f;c[i] = 0.0f;
    }

    float *a_d, *b_d, *c_d;
    cudaMalloc(&a_d, N * sizeof(float));
    cudaMalloc(&b_d, N * sizeof(float));
    cudaMalloc(&c_d, N * sizeof(float));

    cudaMemcpy(a_d, a, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(c_d, c, N*sizeof(float), cudaMemcpyHostToDevice);

    int threadCount = 256;
    int blockCount = (N+threadCount-1) / threadCount;

    addVectorGPU<<< blockCount, threadCount >>> (a_d, b_d, c_d, N);
    cudaDeviceSynchronize();

    cudaMemcpy(a, a_d, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(b, b_d, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(c, c_d, N*sizeof(float), cudaMemcpyDeviceToHost);

    int total = 0;
    for(int i = 0;i < N;i++) {
        total += c[i];
    }

    cout<<N<<" "<< total<<endl;

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    delete [] a;
    delete [] b;
    delete [] c;

    return 0;
}