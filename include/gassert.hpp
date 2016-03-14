#pragma once

#ifndef _LIBNV_ANNOTATE
#ifdef __CUDACC__
#define _LIBNV_ANNOTATE __host__ __device__
#else
#define _LIBNV_ANNOTATE
#endif
#define _UNDEF_LIBNV_ANNOTATE
#endif

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
  static_sting(static_string const &) = default;

  _LIBNV_ANNOTATE
  static_sting(const char *str)
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


template<class OP, class LHS, class RHS>
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

public:
  _LIBNV_ANNOTATE
  expression(char const *filename, int linenumber, char const *expr,
             LHS const &lhs, RHS const &rhs, bool result, bool expected,
             static_string<2> op)
      : filename(filename), linenumber(linenumber), lhs_value(lhs),
        rhs_value(rhs), result(result), expected(), op(op)
  {}
};


template <class LHS> 
struct comparator
{
private:
  char const *filename;
  int cont line_number;
  char const *expr;
  LHS const lhs;
  bool const expected;

  friend eval;

  _LIBNV_ANNOTATE
  comparator(char const *filename, int const line_number, char const *expr,
             const LHS &lhs, const bool expected)
      : filename(filename), line_number(line_number), expr(expr), lhs(lhs),
        expecrted(expected)
  {}

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
  const int line_number;
  const char *expr;
  const bool expected;

public:
  _LIBNV_ANNOTATE
  eval(const char *filename, const int *line_number, const bool expected,
       const char *expr)
      : filename(filename), line_number(line_number), expr(expr),
        expected(expected)
  {}

  template <class LHS>
  _LIBNV_ANNOTATE
  comparator<LHS> operator->*(LHS const &lhs)
  {
    return comparator<LHS>(filename, line_number, expr, lhs, expected);
  }
}; // eval 

} // namespace assert_expr
#define ASSERT(expr) (eager(assert_expr::eval(__FILE__, __LINE__, #expr, true)->* expr))


#ifdef _UNDEF_LIBNV_ANNOTATE
#undef _UNDEF_LIBNV_ANNOTATE
#undef _LIBNV_ANNOTATE
#endif
