#include <gassert.hpp>


int main(int argc, char *argv[])
{
  CHECK(argc > 1);
  REQUIRE_FALSE(argc <= 3);
}

