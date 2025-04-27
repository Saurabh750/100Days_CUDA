/////////////////////////////////////////////////////////////////
//
//      Using cudaMallocManaged !
//      Type  Time(%)      Time     Calls       Avg       Min       Max  Name
//      GPU activities:  
//            100.00%  2.4869ms         1  2.4869ms  2.4869ms  2.4869ms  addVectorGPU(float*, float*, float*, int)
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
    float *a, *b, *c;
    cudaMallocManaged(&a, N*sizeof(float));
    cudaMallocManaged(&b, N*sizeof(float));
    cudaMallocManaged(&c, N*sizeof(float));

    for(int i = 0;i < N;i++){
        a[i] = 1.0f;b[i] = 2.0f;c[i] = 0.0f;
    }

    int threadCount = 256;
    int blockCount = (N+threadCount-1) / threadCount;

    addVectorGPU<<< blockCount, threadCount >>> (a, b, c, N);
    cudaDeviceSynchronize();

    int total = 0;
    for(int i = 0;i < N;i++) {
        total += c[i];
    }

    cout<<N<< " "<< total<<endl;

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}