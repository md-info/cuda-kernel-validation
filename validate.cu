// Reuse the upstream kernel while providing a separate validation entry point.
#define main upstream_sample_main
#include "cpp/0_Introduction/matrixMul/matrixMul.cu"
#undef main
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

template<int Tile> bool check_case(int m, int n, int k) {
    // Signed binary fractions keep these small dot products exactly representable.
    std::vector<float> a(m*k), b(k*n), c(m*n);
    for (int i=0; i<m*k; ++i) a[i]=float((i*7+3)%19-9)/8.f;
    for (int i=0; i<k*n; ++i) b[i]=float((i*11+1)%23-11)/8.f;
    float *da=nullptr, *db=nullptr, *dc=nullptr;
    checkCudaErrors(cudaMalloc(&da, a.size()*sizeof(float)));
    checkCudaErrors(cudaMalloc(&db, b.size()*sizeof(float)));
    checkCudaErrors(cudaMalloc(&dc, c.size()*sizeof(float)));
    checkCudaErrors(cudaMemcpy(da,a.data(),a.size()*sizeof(float),cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(db,b.data(),b.size()*sizeof(float),cudaMemcpyHostToDevice));
    // Unwritten output elements remain NaN and fail the finite-value check.
    checkCudaErrors(cudaMemset(dc,0xff,c.size()*sizeof(float)));
    // Round up the grid to exercise partial output tiles.
    MatrixMulCUDA<Tile><<<dim3((n+Tile-1)/Tile,(m+Tile-1)/Tile),dim3(Tile,Tile)>>>(dc,da,db,k,n,m);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(c.data(),dc,c.size()*sizeof(float),cudaMemcpyDeviceToHost));
    // Compare every output with an independent double-precision CPU reference.
    bool ok=true;
    double max_error=0;
    for (int row=0; row<m; ++row) for (int col=0; col<n; ++col) {
        double ref=0;
        for (int j=0; j<k; ++j) ref+=double(a[row*k+j])*double(b[j*n+col]);
        double error=std::abs(double(c[row*n+col])-ref);
        max_error=std::max(max_error,error);
        // Reject nonfinite values before applying absolute-plus-relative tolerance.
        if (!std::isfinite(c[row*n+col]) || error>1e-5+1e-5*std::abs(ref)) ok=false;
    }
    std::printf("tile=%d M=%d N=%d K=%d max_abs_error=%.9g %s\n",Tile,m,n,k,max_error,ok?"PASS":"FAIL");
    checkCudaErrors(cudaFree(da)); checkCudaErrors(cudaFree(db)); checkCudaErrors(cudaFree(dc));
    return ok;
}

template<int Tile> bool suite() {
    // Cover sub-tile, aligned, rectangular, and partial-reduction dimensions.
    const int shapes[][3]={{1,1,1},{7,5,3},{15,17,31},{16,16,16},{17,15,17},
        {31,33,32},{32,32,32},{33,31,33},{63,65,67},{65,63,64},{97,101,127}};
    bool ok=true;
    // Run every case even if an earlier case fails.
    for (auto &s:shapes) ok=check_case<Tile>(s[0],s[1],s[2]) && ok;
    return ok;
}

int main() {
    cudaDeviceProp p{}; checkCudaErrors(cudaGetDeviceProperties(&p,0));
    int driver=0,runtime=0;
    checkCudaErrors(cudaDriverGetVersion(&driver)); checkCudaErrors(cudaRuntimeGetVersion(&runtime));
    std::printf("GPU=%s sm=%d%d driver=%d runtime=%d\n",p.name,p.major,p.minor,driver,runtime);
    bool a=suite<16>(), b=suite<32>();
    return a && b ? EXIT_SUCCESS : EXIT_FAILURE;
}
