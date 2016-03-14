#pragma once

#ifndef _LIBNV_ANNOTATE
#ifdef __CUDACC__
#define _LIBNV_ANNOTATE __host__ __device__
#else
#define _LIBNV_ANNOTATE
#endif
#define _UNDEF_LIBNV_ANNOTATE
#endif

#include <string>
#include <vector>
#include <csetjmp>

namespace gassert
{

template<size_t N>
struct static_string
{
private:
  char str_[N+1];
  size_t size_;
public:
  _LIBNV_ANNOTATE
  static_string(static_string const &) = default;

  _LIBNV_ANNOTATE
  static_string(const char *str)
  {
    size_ = 0;
    while (str[size_] != 0 || size_ <= N)
    {
      str_[size_] = str[size_];
      ++size_;
    }
    str_[size_] = 0;
  }

  _LIBNV_ANNOTATE
  char const* c_str() const 
  {
    return str_;
  }
  _LIBNV_ANNOTATE
  size_t size() const 
  {
    return size_;
  }
}; // struct static_string

template<size_t N>
struct fails
{
  static_string<256> filename;
}; // struct translation_unit

using std::string;
using std::to_string;
using std::vector;

struct expression_string
{
  string filename;
  string linenumber;
  string lhs_expr;
  string rhs_expr;
  string lhs_value;
  string rhs_value;
  static_string<2> op;
};

template<class LHS, class RHS>
struct expression
{
private:
  char const *filename;
  int const linenumber;
  char const *expr;
  LHS const lhs_value;
  RHS const rhs_value;
  bool const expected;
  bool const result;
  static_string<2> const op;
  
  int  lhs_expr_begin, lhs_expr_end;
  int  rhs_expr_begin, rhs_expr_end;

public:
  _LIBNV_ANNOTATE
  expression(char const *filename, int linenumber, char const *expr,
             LHS const &lhs, RHS const &rhs, bool result, bool expected,
             static_string<2> op)
      : filename(filename), linenumber(linenumber), expr(expr), lhs_value(lhs),
        rhs_value(rhs), result(result), expected(expected), op(op)
  {
    int displ = 0;
    while (expr[displ] && !isalnum(expr[displ]))
      ++displ;
    lhs_expr_begin = displ;
    while (expr[displ] && isalnum(expr[displ]))
      ++displ;
    lhs_expr_end = displ;
    
    while (expr[displ] && !isalnum(expr[displ]))
      ++displ;
    rhs_expr_begin = displ;
    while (expr[displ] && isalnum(expr[displ]))
      ++displ;
    rhs_expr_end = displ;
  }

  _LIBNV_ANNOTATE
  operator bool() const 
  {
    return expected == result;
  }

  expression_string to_expr_string() const
  {
    return {
        filename, 
        to_string(linenumber),
        string(expr+lhs_expr_begin, expr+lhs_expr_end),
        string(expr+rhs_expr_begin, expr+rhs_expr_end),
        to_string(lhs_value),
        to_string(rhs_value), 
        op
    };
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
  bool const expected;

  friend eval;

  _LIBNV_ANNOTATE
  comparator(char const *filename, int const line_number, char const *expr,
             const LHS &lhs, const bool expected)
      : filename(filename), line_number(line_number), expr(expr), lhs(lhs),
        expected(expected)
  {
  }

public:
  //  operator==
  template <class RHS> 
  _LIBNV_ANNOTATE
  expression<LHS, RHS> operator==(RHS const &rhs)
  {
    const bool result = lhs == rhs;
    return expression<LHS, RHS>(filename, line_number, expr, lhs, rhs, result,
                                expected, "==");
  }

  //  operator!= 
  template <class RHS> 
  _LIBNV_ANNOTATE
  expression<LHS, RHS> operator!=(RHS const &rhs)
  {
    const bool result = lhs != rhs;
    return expression<LHS, RHS>(filename, line_number, expr, lhs, rhs, result,
                                expected, "!=");
  }
  
  //  operator>= 
  template<class RHS>
  _LIBNV_ANNOTATE
  expression<LHS,RHS> operator>=(RHS const &rhs)
  {
    const bool result = lhs >= rhs;
    return expression<LHS, RHS>(filename, line_number, expr, lhs, rhs, result,
                                expected, ">=");
  }

  //  operator>
  template<class RHS>
  _LIBNV_ANNOTATE
  expression<LHS,RHS> operator>(RHS const &rhs)
  {
    const bool result = lhs > rhs;
    return expression<LHS, RHS>(filename, line_number, expr, lhs, rhs, result,
                                expected, ">");
  }

  //  operator<= 
  template<class RHS> 
  _LIBNV_ANNOTATE
  expression<LHS,RHS> operator<=(RHS const &rhs)
  {
    const bool result = lhs <= rhs;
    return expression<LHS, RHS>(filename, line_number, expr, lhs, rhs, result,
                                expected, "<=");
  }

  //  operator<
  template<class RHS>
  _LIBNV_ANNOTATE
  expression<LHS,RHS> operator<(RHS const &rhs)
  {
    const bool result = lhs < rhs;
    return expression<LHS, RHS>(filename, line_number, expr, lhs, rhs, result,
                                expected, "<");
  }
}; // struct comparator

struct eval
{
private:
  const char *filename;
  const int linenumber;
  const char *expr;
  const bool expected;

public:
  _LIBNV_ANNOTATE
  eval(const char *filename, int linenumber, const char *expr, bool expected)
      : filename(filename), linenumber(linenumber), expr(expr),
        expected(expected)
  {}

  template <class LHS>
  _LIBNV_ANNOTATE
  comparator<LHS> operator->*(LHS const &lhs)
  {
    return comparator<LHS>(filename, linenumber, expr, lhs, expected);
  }
}; // eval 

struct failed_on_host
{
  vector<expression_string> failed;
  private:
    failed_on_host() = default;
    ~failed_on_host() = default;
  public:
    static failed_on_host& get_instance() 
    {
      static failed_on_host s;
      return s;
    }

    void push_back(expression_string expr)
    {
      failed.push_back(expr);
    }

    bool empty() const 
    {
      return failed.size() == 0;
    }
};

struct assert_fail {};

template<class LHS, class RHS>
void eager(expression<LHS,RHS> expr)
{
  if (!expr)
  {
    failed_on_host::get_instance().push_back(expr.to_expr_string());
  }
}


} // namespace assert_expr
#define ASSERT(expr) (eager(gassert::eval(__FILE__, __LINE__, #expr, true)->* expr))


#ifdef _UNDEF_LIBNV_ANNOTATE
#undef _UNDEF_LIBNV_ANNOTATE
#undef _LIBNV_ANNOTATE
#endif
