#pragma once
#include <cstdio>
#include <cassert>

#ifndef __CUDACC__
#define __host__
#define __device__
#define __global__
#endif

namespace gassert
{

template<int N>
struct static_string
{
private:
  char str_[N+1];
  int size_;
public:

  __host__ __device__
  static_string(const char *str)
  {
    size_ = 0;
    while (str[size_] != 0 && size_ <= N)
    {
      str_[size_] = str[size_];
      ++size_;
    }
    str_[size_] = 0;
  }
  
  __host__ __device__
  static_string(const char *str, int count)
  {
    size_ = 0;
    while (str[size_] != 0 && size_ <= N && count > 0)
    {
      str_[size_] = str[size_];
      ++size_;
      count--;
    }
    str_[size_] = 0;
  }

  __host__ __device__
  char const* c_str() const 
  {
    return str_;
  }
  __host__ __device__
  int size() const 
  {
    return size_;
  }
}; // struct static_string

  __host__ __device__
bool __isalnum(char c)
{
  bool ret = false;
  ret |= c >= '0' && c <= '9';
  ret |= c >= 'a' && c <= 'z';
  ret |= c >= 'A' && c <= 'Z';
  return ret;
}

template<class LHS, class RHS>
struct expression
{
private:
  char const *filename;
  int const linenumber;
  char const *expr;
  LHS const lhs_value;
  RHS const rhs_value;
  bool const result;
  const char *op;
  bool expected;
  
  int  lhs_expr_begin, lhs_expr_end;
  int  rhs_expr_begin, rhs_expr_end;

public:
  __host__ __device__
  expression(char const *filename, int linenumber, char const *expr,
             LHS const &lhs, RHS const &rhs, bool result,
             const char * op)
      : filename(filename), linenumber(linenumber), expr(expr), lhs_value(lhs),
        rhs_value(rhs), result(result),  op(op), expected(true)
  {
    int displ = 0;
    while (expr[displ] && !__isalnum(expr[displ]))
      ++displ;
    lhs_expr_begin = displ;
    while (expr[displ] && __isalnum(expr[displ]))
      ++displ;
    lhs_expr_end = displ;
    
    while (expr[displ] && !__isalnum(expr[displ]))
      ++displ;
    rhs_expr_begin = displ;
    while (expr[displ] && __isalnum(expr[displ]))
      ++displ;
    rhs_expr_end = displ;
  }

  __host__ __device__
  operator bool() const 
  {
    return expected == result;
  }

  __host__ __device__
  void set_expected(bool b) { expected = b; }

  //-----------------------------------
  // Specialize value printing here
  //-----------------------------------

  __host__ __device__
  void print_value(int val) const
  {
    printf("%d",val);
  }
  __host__ __device__
  void print_value(float val)
  {
    printf("%f",val);
  }

  __host__ __device__
  void print(const char *what) const
  {
#ifndef GASSERT_NO_COLOURS
#define GASSERT_RESET      "\033[0m"
#define GASSERT_BOLDRED    "\033[1m\033[31m" 
#define GASSERT_GREEN      "\033[32m"
#define GASSERT_GREY       "\033[90m"
#define GASSERT_BOLDBLACK  "\033[1m\033[30m"
#else
#define GASSERT_RESET      ""
#define GASSERT_BOLDRED    ""
#define GASSERT_GREEN      ""
#define GASSERT_GREY       ""
#define GASSERT_BOLDBLACK  ""
#endif

    printf(GASSERT_BOLDBLACK "\n%s:%d " 
           GASSERT_BOLDRED "[FAILED]" GASSERT_RESET "\n", 
           filename, linenumber);
    const int N = 16;
    typedef static_string<N> expr_t;
    expr_t lhs(expr+lhs_expr_begin, lhs_expr_end-lhs_expr_begin);
    expr_t rhs(expr+rhs_expr_begin, rhs_expr_end-rhs_expr_begin);
    printf(GASSERT_GREY "    %s(", what);
    printf("%s %s %s", lhs.c_str(), op, rhs.c_str());
    printf(")" GASSERT_RESET "\n");
    printf("with expansion:\n");

    printf(GASSERT_GREEN "    %s(", (expected ? "" : "!"));
    print_value(lhs_value);
    printf(" %s ", op);
    print_value(rhs_value);
    printf(")" GASSERT_RESET); 
    printf("\n\n");

#undef GASSERT_RESET
#undef GASSERT_BOLDRED
#undef GASSERT_GREEN
#undef GASSERT_GREY
#undef GASSERT_BOLDBLACK
  }
};

struct eval;

template <class LHS> 
struct comparator
{
private:
  char const *filename;
  int const line_number;
  char const *expr;
  LHS const lhs;

public:

  __host__ __device__
  comparator(char const *filename, int const line_number, char const *expr,
             const LHS &lhs)
      : filename(filename), line_number(line_number), expr(expr), lhs(lhs)
  {
  }

private:

  template<class RHS>
  __host__ __device__
  expression<LHS,RHS> construct_expression(RHS const &rhs, 
                                           bool result, 
                                           const char *op)
  {
    return expression<LHS, RHS>(filename, line_number, expr, lhs, rhs, result,
                                op);
  }
                                           
public:

  //  operator==
  template <class RHS> 
  __host__ __device__
  expression<LHS, RHS> operator==(RHS const &rhs)
  {
    return construct_expression(rhs, lhs == rhs, "==");
  }

  //  operator!= 
  template <class RHS> 
  __host__ __device__
  expression<LHS, RHS> operator!=(RHS const &rhs)
  {
    return construct_expression(rhs, lhs != rhs, "!=");
  }
  
  //  operator>= 
  template<class RHS>
  __host__ __device__
  expression<LHS,RHS> operator>=(RHS const &rhs)
  {
    return construct_expression(rhs, lhs >= rhs, ">=");
  }

  //  operator>
  template<class RHS>
  __host__ __device__
  expression<LHS,RHS> operator>(RHS const &rhs)
  {
    return construct_expression(rhs, lhs > rhs, ">");
  }

  //  operator<= 
  template<class RHS> 
  __host__ __device__
  expression<LHS,RHS> operator<=(RHS const &rhs)
  {
    return construct_expression(rhs, lhs <= rhs, "<=");
  }

  //  operator<
  template<class RHS>
  __host__ __device__
  expression<LHS,RHS> operator<(RHS const &rhs)
  {
    return construct_expression(rhs, lhs < rhs, "<");
  }
}; // struct comparator

struct eval
{
private:
  const char *filename;
  const int linenumber;
  const char *expr;

public:
  __host__ __device__
  eval(const char *filename, int linenumber, const char *expr)
      : filename(filename), linenumber(linenumber), expr(expr)
  {}

  template <class LHS>
  __host__ __device__
  comparator<LHS> operator->*(LHS const &lhs)
  {
    return comparator<LHS>(filename, linenumber, expr, lhs);
  }
}; // eval 

void __host__ __device__
trap() 
{
  assert(0);
}

template<class LHS, class RHS>
void __host__ __device__
require(expression<LHS,RHS> expr)
{
  if (!expr)
  {
    expr.print("ASSERT");
    trap();
  }
}


template<class LHS, class RHS>
__host__ __device__
void require_false(expression<LHS,RHS> expr)
{
  expr.set_expected(false);
  if (!expr)
  {
    expr.print("ASSERT_FALSE");
    trap();
  }
}

template <class LHS, class RHS>
void __host__ __device__ 
check(expression<LHS, RHS> expr) 
{
  if (!expr)
  {
    expr.print("CHECK");
  }
}


template<class LHS, class RHS>
void __host__ __device__
check_false(expression<LHS,RHS> expr)
{
  expr.set_expected(false);
  if (!expr)
  {
    expr.print("CHECK_FALSE");
  }
}


} // namespace gassert
#define REQUIRE(expr) (gassert::require(gassert::eval(__FILE__, __LINE__, #expr)->* expr))
#define REQUIRE_FALSE(expr) (gassert::require_false(gassert::eval(__FILE__, __LINE__, #expr)->* expr))
#define CHECK(expr) (gassert::check(gassert::eval(__FILE__, __LINE__, #expr)->* expr))
#define CHECK_FALSE(expr) (gassert::check_false(gassert::eval(__FILE__, __LINE__, #expr)->* expr))

