#ifndef MACH_UTILS
#define MACH_UTILS

#include <functional>
#include <exception>
#include <iostream>

#include "mfem.hpp"

#include "mach_types.hpp"

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

std::ostream *getOutStream(int rank, bool silent = false);

/// The following are adapted from MFEM's pfem_extras.xpp to use mach types
/// and support serial usage.
/// Serial solves use MFEM's PCG with a GSSmoother preconditioner 
/// Parallel solves use HyprePCG with an AMS preconditioner
class DiscreteInterpolationOperator : public DiscLinOperatorType
{
public:
   DiscreteInterpolationOperator(SpaceType *dfes,
                                 SpaceType *rfes)
      : DiscLinOperatorType(dfes, rfes) {}
   virtual ~DiscreteInterpolationOperator();
};

class DiscreteGradOperator : public DiscreteInterpolationOperator
{
public:
   DiscreteGradOperator(SpaceType *dfes,
                        SpaceType *rfes);
};

class DiscreteCurlOperator : public DiscreteInterpolationOperator
{
public:
   DiscreteCurlOperator(SpaceType *dfes,
                        SpaceType *rfes);
};

class DiscreteDivOperator : public DiscreteInterpolationOperator
{
public:
   DiscreteDivOperator(SpaceType *dfes,
                       SpaceType *rfes);
};

class IrrotationalProjector : public mfem::Operator
{
public:
   IrrotationalProjector(SpaceType &H1FESpace,
                         SpaceType &HCurlFESpace,
                         const int &irOrder,
                         BilinearFormType *s0 = NULL,
                         MixedBilinearFormType *weakDiv = NULL,
                         DiscreteGradOperator *grad = NULL);
   virtual ~IrrotationalProjector();

   // Given a GridFunction 'x' of Nedelec DoFs for an arbitrary vector field,
   // compute the Nedelec DoFs of the irrotational portion, 'y', of
   // this vector field.  The resulting GridFunction will satisfy Curl y = 0
   // to machine precision.
   virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const;

   void Update();

private:
   void InitSolver() const;

   SpaceType *H1FESpace_;
   SpaceType *HCurlFESpace_;

   BilinearFormType *s0_;
   MixedBilinearFormType *weakDiv_;
   DiscreteGradOperator *grad_;

   GridFunType * psi_;
   GridFunType * xDiv_;

   MatrixType *S0_;
   mutable mfem::Vector Psi_;
   mutable mfem::Vector RHS_;

   mutable EMPrecType2 *amg_;
#ifdef MFEM_USE_MPI
   mutable mfem::HyprePCG *pcg_;
#else
   mutable CGType *pcg_;
#endif

   mfem::Array<int> ess_bdr_, ess_bdr_tdofs_;

   bool ownsS0_;
   bool ownsWeakDiv_;
   bool ownsGrad_;
};

/// This class computes the divergence free portion of a vector field.
/// This vector field must be discretized using Nedelec basis
/// functions.
class DivergenceFreeProjector : public IrrotationalProjector
{
public:
   DivergenceFreeProjector(SpaceType &H1FESpace,
                           SpaceType &HCurlFESpace,
                           const int &irOrder,
                           BilinearFormType *s0 = NULL,
                           MixedBilinearFormType *weakDiv = NULL,
                           DiscreteGradOperator *grad = NULL);

   virtual ~DivergenceFreeProjector() {}

   // Given a vector 'x' of Nedelec DoFs for an arbitrary vector field,
   // compute the Nedelec DoFs of the divergence free portion, 'y', of
   // this vector field.  The resulting vector will satisfy Div y = 0
   // in a weak sense.
   virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const;

   void Update();
};
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
/// \note This uses minimum norm reconstruction
#ifdef MFEM_USE_LAPACK
void buildInterpolation(int dim, int degree, const mfem::DenseMatrix &x_center,
    const mfem::DenseMatrix &x_quad, mfem::DenseMatrix &interp);
#endif

/// build the reconstruction matrix that interpolate the GD dofs to quadrature points
/// \param[in] degree - order of reconstructio operator
/// \param[in] x_cent - coordinates of barycenters
/// \param[in] x_quad - coordinates of quadrature points
/// \param[out] interp - interpolation operator
/// \note This uses a least-squares reconstruction
#ifdef MFEM_USE_LAPACK
void buildLSInterpolation(int dim, int degree,
                          const mfem::DenseMatrix &x_center,
                          const mfem::DenseMatrix &x_quad,
                          mfem::DenseMatrix &interp);
#endif

/// \brief transfer GridFunction from one mesh to another
/// \note requires MFEM to be built with GSLIB
void transferSolution(MeshType &old_mesh, MeshType &new_mesh,
                      const GridFunType &in, GridFunType &out);

} // namespace mach

#endif 
