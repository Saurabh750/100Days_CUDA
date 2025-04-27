#include <iostream>

using namespace std;

void addVectorCPU(float* a, float* b, float* c, int n) {
    for(int i = 0;i < n;i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {

    int N = 1<<20;
    float* a = new float[N];
    float* b = new float[N];
    float* c = new float[N];

    for(int i = 0;i < N;i++) {
        a[i] = 1; b[i] = 2; c[i] = 0;
    }

    addVectorCPU(a, b, c, N);

    int total = 0;
    for(int i = 0;i < N;i++)
        total += c[i];

    cout<<N<<" "<<total<<endl;

    delete [] a;
    delete [] b;
    delete [] c;
    return 0;
}

