#if 0
#define GASSERT_NO_COLOURS
#endif
#include <gassert.hpp>


__device__ void test(int argc)
{
  CHECK_FALSE(argc > 1);
  REQUIRE_FALSE(argc <= 3);
  REQUIRE(argc <= 2);
}

__global__ void  kernel(int argc)
{
  test(argc);
}

int main(int argc, char *argv[])
{
  kernel
#ifdef __CUDACC__
    <<<1,1>>>
#endif
    (argc);
#ifdef __CUDACC__
  cudaDeviceSynchronize();
#endif
}

