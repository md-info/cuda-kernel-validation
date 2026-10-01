#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <initializer_list>
#include "cutlass/half.h"
#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/predicated_tile_access_iterator.h"

#define CUDA_CHECK(call) do { auto status = (call); if (status != cudaSuccess) { \
  std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(status)); std::exit(2); } } while (0)

// Observe byte displacement without dereferencing a potentially incorrect address.
template <typename Element, int Rank>
__global__ void offsets(Element *data, long long *actual, int extent_c, int extent_s,
                        int move_c, int move_s, bool steady) {
  using Shape = cutlass::layout::PitchLinearShape<32, 4>;
  using Map = cutlass::transform::PitchLinearStripminedThreadMap<Shape, 32>;
  using Iterator = cutlass::transform::threadblock::PredicatedTileAccessIterator<
      Shape, Element, cutlass::layout::PitchLinear, Rank, Map,
      cutlass::AlignedArray<Element, 1>>;
  typename Iterator::Params params(cutlass::layout::PitchLinear(256));
  // Keep backward residual transitions inside the allocated tensor.
  Iterator it(params, data, {extent_c, extent_s}, threadIdx.x, {32, 4});
  // First consume the residual tile to isolate the steady-state branch.
  if (steady) it.add_tile_offset({Rank == 0 ? 1 : 0, Rank == 1 ? 1 : 0});
  auto before = reinterpret_cast<std::intptr_t>(it.get());
  it.add_tile_offset({move_c, move_s});
  actual[threadIdx.x] = static_cast<long long>(reinterpret_cast<std::intptr_t>(it.get()) - before);
}

template <typename Element, int Rank>
int check_type(char const *name) {
  Element *data;
  long long *device_result, result[32];
  CUDA_CHECK(cudaMalloc(&data, 256 * 256 * sizeof(Element)));
  CUDA_CHECK(cudaMalloc(&device_result, sizeof(result)));
  int failed = 0;
  // Aligned and residual extents; zero, advance-only, cross-only and diagonal moves.
  int const moves[][2] = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
  for (bool steady : {false, true}) {
    for (int extent : {64, 67}) {
      for (auto const &move : moves) {
        offsets<Element, Rank><<<1, 32>>>(data, device_result, extent, extent,
                                         move[0], move[1], steady);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(result, device_result, sizeof(result), cudaMemcpyDeviceToHost));
        // The first transition advances by the residual size, then whole tiles.
        int dc = 32 * move[0], ds = 4 * move[1];
        if (!steady) {
          if (Rank == 0) dc += (extent % 32 ? extent % 32 : 32) - 32;
          else ds += (extent % 4 ? extent % 4 : 4) - 4;
        }
        long long expected = (dc + 256LL * ds) * sizeof(Element);
        bool pass = true;
        for (auto value : result) pass = pass && value == expected;
        failed += !pass;
        std::printf("type=%s rank=%d steady=%d extent=%d move=(%d,%d) expected=%lld actual=%lld %s\n",
                    name, Rank, steady, extent, move[0], move[1], expected, result[0], pass ? "PASS" : "FAIL");
      }
    }
  }
  CUDA_CHECK(cudaFree(device_result));
  CUDA_CHECK(cudaFree(data));
  return failed;
}

int main() {
  int failed = check_type<cutlass::half_t, 0>("fp16") + check_type<cutlass::half_t, 1>("fp16")
             + check_type<float, 0>("fp32") + check_type<float, 1>("fp32");
  std::printf("failed=%d total=64\n", failed);
  return failed ? 1 : 0;
}
