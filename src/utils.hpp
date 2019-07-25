#ifndef MACH_UTILS
#define MACH_UTILS

#include <exception>
#include <iostream>

namespace mach
{
/* This function perform the quadratic interpolation between (x0, y0)
   and (x1, y1) with y0' provided. */
double quadInterp(double x0, double y0, double dydx0, 
                     double x1, double y1)
{
   /// Assume the fuction has the form y(x) = c0 + c1 * x + c2 * x^2;
   double c0, c1, c2;
   c0 = (dydx0*x0*x0*x1 + y1*x0*x0 - dydx0*x0*x1*x1 
         - 2*y0*x0*x1 + y0*x1*x1)/(x0*x0 - 2*x1*x0 + x1*x1);
   c1 = (2*x0*y0 - 2*x0*y1 - x0*x0*dydx0 + x1*x1*dydx0)
         /(x0*x0 - 2*x1*x0 + x1*x1);
   c2 = -(y0 - y1 - x0*dydx0 + x1*dydx0)/(x0*x0 - 2*x1*x0 + x1*x1);
   std::cout << c2 <<  "  "<< c1 << "  " << c0 <<std::endl;
   return -c1/(2*c2);
}

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

} // namespace mach

#endif 