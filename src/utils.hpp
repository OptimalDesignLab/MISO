#ifndef MACH_UTILS
#define MACH_UTILS

#include <exception>
#include <iostream>
#include "mfem.hpp"

namespace mach
{

/// Perform quadratic interpolation based on (x0,y0,dydx0) and (x1,y1)
/// \param[in] x0 - location of first dependent data point
/// \param[in] y0 - value of function at `x0`
/// \param[in] dydx0 - value of derivative of function at `x0`
/// \param[in] x1 - location of second dependent data point
/// \param[in] y1 - value of function at `y1`
double quadInterp(double x0, double y0, double dydx0, double x1, double y1);

/// Handles (high-level) exceptions in both serial and parallel
class MachException: public std::exception
{
public:
   /// Class constructor.
   /// \param[in] err_msg - the error message to be printed
   MachException(std::string err_msg) : error_msg(err_msg) {}
   
   /// Overwrites inherieted member that returns a c-string.
   virtual const char* what() const noexcept
   {
      return error_msg.c_str();
   }

   /// Use this to print the message; prints only on root for parallel runs.
   void print_message()
   {
      // TODO: handle parallel runs!!!
      std::cerr << error_msg << std::endl;
   }
protected:
   /// message printed to std::cerr
   std::string error_msg;
};

/// performs the Hadamard (elementwise) product: `v(i) = v1(i)*v2(i)`
void multiplyElementwise(const mfem::Vector &v1, const mfem::Vector &v2,
                         mfem::Vector &v);

/// performs the Hadamard (elementwise) product: `a(i) *= b(i)`
void multiplyElementwise(const mfem::Vector &b, mfem::Vector &a);

/// performs an elementwise division: `v(i) = v1(i)/v2(i)`
void divideElementwise(const mfem::Vector &v1, const mfem::Vector &v2,
                       mfem::Vector &v);

/// performs elementwise inversion: `y(i) = 1/x(i)`
void invertElementwise(const mfem::Vector &x, mfem::Vector &y);

/// for performing loop unrolling of dot-products using meta-programming
/// \tparam xdouble - `double` or `adept::adouble`
/// \tparam dim - number of dimensions for array
/// This was adapted from http://www.informit.com/articles/article.aspx?p=30667&seqNum=7
template <typename xdouble, int dim>
class DotProduct {
  public:
    static xdouble result(const xdouble *a, const xdouble *b)
    {
        return *a * *b  +  DotProduct<xdouble,dim-1>::result(a+1,b+1);
    }
};

// partial specialization as end criteria
template <typename xdouble>
class DotProduct<xdouble,1> {
  public:
    static xdouble result(const xdouble *a, const xdouble *b)
    {
        return *a * *b;
    }
};

/// dot product of two arrays that uses an unrolled loop
/// \param[in] a - first vector involved in product
/// \param[in] b - second vector involved in product
/// \tparam xdouble - typically `double` or `adept::adouble`
/// \tparam dim - number of array dimensions
template <typename xdouble, int dim>
inline xdouble dot(const xdouble *a, const xdouble *b)
{
    return DotProduct<xdouble,dim>::result(a,b);
}

/// Handles print in parallel case
template<typename _CharT, typename _Traits>

class basic_oblackholestream
    : virtual public std::basic_ostream<_CharT, _Traits>
{
public:   
  /// called when rank is not root, prints nothing 
    explicit basic_oblackholestream() : std::basic_ostream<_CharT, _Traits>(NULL) {}
}; // end class basic_oblackholestream

using oblackholestream = basic_oblackholestream<char,std::char_traits<char> >;
static oblackholestream obj;

static std::ostream *getOutStream(int rank) 
{
   /// print only on root
   if (0==rank)
   {
      return &std::cout;
   }
   else
   {
      return &obj;
   }
}

/// Implementation of expression SFINAE, used to check if an object has a 
/// method with a specific signature implemented. Usage looks like:
///
/// auto hasEval = is_valid([](auto&& x) -> decltype(x.Eval(trans, ip)) { });
///
/// In above espression, `decltype(x.Eval(trans, ip))` is the return type of
/// the lambda expression. `decltype()` returns the type of the expression
/// passed to it without actually evaluating the expression. This calls
/// `is_valid()` with the return type of `x.Eval(trans, ip)`. If x has an
/// `Eval` method with the same signature, function `testValidity` will return
/// a true_type, otherwise it will return a false type.
/// View MeshDependentCoefficient for complete usage.
/// Adapted from https://jguegant.github.io/blogs/tech/sfinae-introduction.html
// template <typename UnnamedType> struct Container
// {
// public:
//     // A public operator() that accept the argument we wish to test onto the UnnamedType.
//     // Notice that the return type is automatic!
//     template <typename Param> constexpr auto operator()(const Param& p)
//     {
//         // The argument is forwarded to one of the two overloads.
//         // The SFINAE on the 'true_type' will come into play to dispatch.
//         // Once again, we use the int for the precedence.
//         return testValidity<Param>(int());
//     }
// private:
//     // We use std::declval to 'recreate' an object of 'UnnamedType'.
//     // We use std::declval to also 'recreate' an object of type 'Param'.
//     // We can use both of these recreated objects to test the validity!
//     template <typename Param> constexpr auto testValidity(int /* unused */)
//     -> decltype(std::declval<UnnamedType>()(std::declval<Param>()), std::true_type())
//     {
//         // If substitution didn't fail, we can return a true_type.
//         return std::true_type();
//     }

//     template <typename Param> constexpr std::false_type testValidity(...)
//     {
//         // Our sink-hole returns a false_type.
//         return std::false_type();
//     }
// };

// template <typename UnnamedType> constexpr auto is_valid(const UnnamedType& t) 
// {
//     // We used auto for the return type: it will be deduced here.
//     return Container<UnnamedType>();
// }

} // namespace mach

#endif 
