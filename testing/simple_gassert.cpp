#include <cstdio>
#include <gassert.hpp>
#include <vector>


int main(int argc, char *argv[])
{
  ASSERT(argc > 3);
  auto& s= gassert::failed_on_host::get_instance();
  fprintf(stderr, " s.failed.size() = %d\n", (int)s.failed.size());
#if 0
  printf("x.filename= %s\n", x.filename.c_str());
  printf("x.liennumber= %s\n", x.linenumber.c_str());
  printf("x.expr= %s\n", x.expr.c_str());
  printf("x.lhs_value= %s\n", x.lhs_value.c_str());
  printf("x.op= %s\n", x.op.c_str());
  printf("x.rhs_value= %s\n", x.rhs_value.c_str());
#endif
}

