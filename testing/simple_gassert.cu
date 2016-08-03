#include <gassert.hpp>


__device__ void test(int argc)
{
  CHECK(argc <= 1);
  REQUIRE_FALSE(argc <= 3);
}

__global__ void  kernel(int argc)
{
  test(argc);
}

int main(int argc, char *argv[])
{
  kernel<<<1,1>>>(argc);
}

