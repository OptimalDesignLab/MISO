#ifndef MACH_UTILS
#define MACH_UTILS

#include <functional>
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

/// Find root of `func` using bisection
/// \param[in] func - function to find root of 
/// \param[in] xl - left bracket of root
/// \param[in] xr - right bracket of root
/// \param[in] ftol - absolute tolerance for root function
/// \param[in] xtol - absolute tolerance for root value
/// \param[in] maxiter - maximum number of iterations
double bisection(std::function<double(double)> func, double xl, double xr,
                 double ftol, double xtol, int maxiter);

/// build the reconstruction matrix that interpolate the GD dofs to quadrature points
/// \param[in] degree - order of reconstructio operator
/// \param[in] x_cent - coordinates of barycenters
/// \param[in] x_quad - coordinates of quadrature points
/// \param[out] interp - interpolation operator
#ifdef MFEM_USE_LAPACK
void buildInterpolation(int dim, int degree, const mfem::DenseMatrix &x_center,
    const mfem::DenseMatrix &x_quad, mfem::DenseMatrix &interp);
#endif

} // namespace mach

#endif 
