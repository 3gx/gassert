#include <cstdio>
#include <gassert.hpp>
#include <vector>


int main(int argc, char *argv[])
{
  ASSERT(argc <= 1);
  auto& s= gassert::failed_on_host::get_instance();
  fprintf(stderr, " s.failed.size() = %d\n", (int)s.failed.size());

  if (!s.empty())
  {
    auto &x = s.failed.front();
  printf("x.filename= %s\n", x.filename.c_str());
  printf("x.liennumber= %s\n", x.linenumber.c_str());
  printf(" %s %s %s\n", x.lhs_expr.c_str(), x.op.c_str(), x.rhs_expr.c_str());
  printf(" %s %s %s\n", x.lhs_value.c_str(), x.op.c_str(), x.rhs_value.c_str());
  }
  
}

