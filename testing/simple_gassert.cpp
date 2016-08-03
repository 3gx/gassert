#include <gassert.hpp>


int main(int argc, char *argv[])
{
  CHECK(argc <= 1);
  REQUIRE_FALSE(argc <= 3);
#if 0
  auto &s = gassert::failed_on_host::get_instance();
  fprintf(stderr, " s.failed.size() = %d\n", (int)s.failed.size());

  for (auto &x : s.failed)
  {
    printf("x.filename= %s\n", x.filename.c_str());
    printf("x.liennumber= %s\n", x.linenumber.c_str());
    printf("FAILED: %s %s %s (%s %s %s)\n", x.lhs_expr.c_str(), x.op.c_str(),
           x.rhs_expr.c_str(), x.lhs_value.c_str(), x.op.c_str(),
           x.rhs_value.c_str());
  }
#endif
}

