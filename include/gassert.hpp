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
#include <sstream>
#include <cassert>

namespace gassert
{

using std::string;
using std::stringstream;
using std::to_string;
using std::vector;
using std::ostream;

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

  string str() const
  {
    return string(str_);
  }

  _LIBNV_ANNOTATE
  size_t size() const 
  {
    return size_;
  }

  _LIBNV_ANNOTATE
  char& operator[](size_t i) 
  {
    return str_[i];
  }
  _LIBNV_ANNOTATE
  const char& operator[](size_t i)  const
  {
    return str_[i];
  }
}; // struct static_string

template<size_t N>
struct fails
{
  static_string<256> filename;
}; // struct translation_unit


#if 0
template<class T, class N>
struct shared_vector
{
  char* data_[N];
  int size_;

  __device__
  void push_back(T const& t) 
  {
    int idx = atomicAdd
  }
}
#endif


struct value
{
  enum
  {
    INT32,
    UINT32,
    INT64,
    UINT64,
    FLOAT,
    DOUBLE,
    POINTER
  } type;

  typedef void *pointer;

  union {
    int32_t int32_;
    uint32_t uint32_;
    int64_t int64_;
    uint64_t uint64_;
    float float_;
    double double_;
    pointer pointer_;
  } element;

  _LIBNV_ANNOTATE
  value(int32_t const &t)
  {
    element.int32_ = t;
    type = INT32;
  }
  _LIBNV_ANNOTATE
  value(uint32_t const &t)
  {
    element.uint32_ = t;
    type = UINT32;
  }
  _LIBNV_ANNOTATE
  value(int64_t const &t)
  {
    element.int64_ = t;
    type = INT64;
  }
  _LIBNV_ANNOTATE
  value(uint64_t const &t)
  {
    element.uint64_ = t;
    type = UINT64;
  }
  _LIBNV_ANNOTATE
  value(float const &t)
  {
    element.float_ = t;
    type = FLOAT;
  }
  _LIBNV_ANNOTATE
  value(double const &t)
  {
    element.double_ = t;
    type = DOUBLE;
  }
  _LIBNV_ANNOTATE
  value(pointer const &t)
  {
    element.pointer_ = t;
    type = POINTER;
  }

  friend ostream &operator<<(ostream &os, value const &v)
  {
    switch (v.type)
    {
    case INT32:
      return os << v.element.int32_;
    case UINT32:
      return os << v.element.uint32_;
    case INT64:
      return os << v.element.int64_;
    case UINT64:
      return os << v.element.uint64_;
    case FLOAT:
      return os << v.element.float_;
    case DOUBLE:
      return os << v.element.double_;
    case POINTER:
      return os << v.element.pointer_;
    default:
      assert(0);
    }
    return os;
  }
}; // struct value;

struct expression_string
{
private:
  uint32_t linenumber_ : 30;
  bool result_ : 1;
  bool expected_ : 1;
  value lhs_value_;
  value rhs_value_;
  static_string<2> op_;
  static_string<256> expr_;

public:
  expression_string(expression_string const &) = default;

  template <class LHS, class RHS>
  _LIBNV_ANNOTATE expression_string(int linenumber, bool result, bool expected,
                                    const LHS &lhs, const RHS &rhs,
                                    const char *expr, const char *op)
      : linenumber_(linenumber), result_(result), expected_(expected),
        lhs_value_(lhs), rhs_value_(rhs), expr_(expr), op_(op)
  {
  }

  int linenumber() const { return linenumber_; }
  bool result() const { return result_; }
  bool expected() const { return expected_; }

  string op() const { return string(op_.c_str()); }
  string expr() const { return string(expr_.c_str()); }
  operator bool() const { return expected_ == result_; }
  string lhs_expr() const
  {
    int displ = 0;
    while (expr_[displ] && !isalnum(expr_[displ]))
      ++displ;
    int lhs_expr_begin = displ;
    while (expr_[displ] && isalnum(expr_[displ]))
      ++displ;
    int lhs_expr_end = displ;

    return string(expr_.c_str() + lhs_expr_begin, expr_.c_str() + lhs_expr_end);
  }
  string rhs_expr() const
  {
    int displ = 0;
    while (expr_[displ] && !isalnum(expr_[displ]))
      ++displ;
    while (expr_[displ] && isalnum(expr_[displ]))
      ++displ;

    while (expr_[displ] && !isalnum(expr_[displ]))
      ++displ;
    int rhs_expr_begin = displ;
    while (expr_[displ] && isalnum(expr_[displ]))
      ++displ;
    int rhs_expr_end = displ;
    return string(expr_.c_str() + rhs_expr_begin, expr_.c_str() + rhs_expr_end);
  }
  string lhs_value() const
  {
    stringstream ss;
    ss << lhs_value_;
    return ss.str();
  }
  string rhs_value() const
  {
    stringstream ss;
    ss << rhs_value_;
    return ss.str();
  }
}; // struct expression_string

template<class LHS, class RHS>
struct expression
{
private:
  char const *filename;
  int const linenumber;
  char const *expr;
  LHS const lhs;
  RHS const rhs;
  bool const expected;
  bool const result;
  static_string<2> const op;

public:
  _LIBNV_ANNOTATE
  expression(char const *filename, int linenumber, char const *expr,
             LHS const &lhs, RHS const &rhs, bool result, bool expected,
             static_string<2> op)
      : filename(filename), linenumber(linenumber), expr(expr), lhs(lhs),
        rhs(rhs), result(result), expected(expected), op(op)
  {
  }

  _LIBNV_ANNOTATE
  operator bool() const { return expected == result; }

  expression_string to_expr_string() const
  {
    return expression_string(linenumber, result, expected, lhs, rhs, expr,
                             op.c_str());
  }
}; // struct expression

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
}; // failed_on_host

struct assert_fail {};

template<class LHS, class RHS>
void eager(expression<LHS,RHS> expr)
{
  if (!expr)
  {
    failed_on_host::get_instance().push_back(expr.to_expr_string());
  }
} // eager


} // namespace assert_expr
#define ASSERT(expr) (eager(gassert::eval(__FILE__, __LINE__, #expr, true)->* expr))
#define ASSERT_FALSE(expr) (eager(gassert::eval(__FILE__, __LINE__, #expr, false)->* expr))


#ifdef _UNDEF_LIBNV_ANNOTATE
#undef _UNDEF_LIBNV_ANNOTATE
#undef _LIBNV_ANNOTATE
#endif
