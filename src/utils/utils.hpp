#ifndef MACH_UTILS
#define MACH_UTILS

#include <functional>
#include <exception>
#include <iostream>
#include <utility>
#include <variant>

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
class MachException : public std::exception
{
public:
   /// Class constructor.
   /// \param[in] err_msg - the error message to be printed
   MachException(std::string err_msg) : error_msg(std::move(std::move(err_msg)))
   { }

   /// Overwrites inherieted member that returns a c-string.
   const char *what() const noexcept override { return error_msg.c_str(); }

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
void multiplyElementwise(const mfem::Vector &v1,
                         const mfem::Vector &v2,
                         mfem::Vector &v);

/// performs the Hadamard (elementwise) product: `a(i) *= b(i)`
void multiplyElementwise(const mfem::Vector &b, mfem::Vector &a);

/// performs an elementwise division: `v(i) = v1(i)/v2(i)`
void divideElementwise(const mfem::Vector &v1,
                       const mfem::Vector &v2,
                       mfem::Vector &v);

/// performs elementwise inversion: `y(i) = 1/x(i)`
void invertElementwise(const mfem::Vector &x, mfem::Vector &y);

/// for performing loop unrolling of dot-products using meta-programming
/// \tparam xdouble - `double` or `adept::adouble`
/// \tparam dim - number of dimensions for array
/// This was adapted from
/// http://www.informit.com/articles/article.aspx?p=30667&seqNum=7
template <typename xdouble, int dim>
class DotProduct
{
public:
   static xdouble result(const xdouble *a, const xdouble *b)
   {
      return *a * *b + DotProduct<xdouble, dim - 1>::result(a + 1, b + 1);
   }
};

// partial specialization as end criteria
template <typename xdouble>
class DotProduct<xdouble, 1>
{
public:
   static xdouble result(const xdouble *a, const xdouble *b) { return *a * *b; }
};

/// dot product of two arrays that uses an unrolled loop
/// \param[in] a - first vector involved in product
/// \param[in] b - second vector involved in product
/// \tparam xdouble - typically `double` or `adept::adouble`
/// \tparam dim - number of array dimensions
template <typename xdouble, int dim>
inline xdouble dot(const xdouble *a, const xdouble *b)
{
   return DotProduct<xdouble, dim>::result(a, b);
}

std::ostream *getOutStream(int rank, bool silent = false);

/// Construct a HypreParVector on the a given FES using external data
/// \param[in] buffer - external data for HypreParVector
/// \param[in] fes - finite element space to construct vector on
mfem::HypreParVector bufferToHypreParVector(
    double *buffer,
    const mfem::ParFiniteElementSpace &fes);

/// \brief A helper type for uniform semantics over owning/non-owning pointers
template <typename T>
using MaybeOwningPointer = std::variant<T *, std::unique_ptr<T>>;

/// \brief Retrieves a reference to the underlying object in a
/// MaybeOwningPointer \param[in] obj The object to dereference
template <typename T>
static T &retrieve(MaybeOwningPointer<T> &obj)
{
   return std::visit([](auto &&ptr) -> T & { return *ptr; }, obj);
}
/// \overload
template <typename T>
static const T &retrieve(const MaybeOwningPointer<T> &obj)
{
   return std::visit([](auto &&ptr) -> const T & { return *ptr; }, obj);
}

// /// The following are adapted from MFEM's pfem_extras.xpp
// class DiscreteGradOperator : public mfem::ParDiscreteLinearOperator
// {
// public:
//    DiscreteGradOperator(mfem::ParFiniteElementSpace *dfes,
//                         mfem::ParFiniteElementSpace *rfes);
// };

// class DiscreteCurlOperator : public mfem::ParDiscreteLinearOperator
// {
// public:
//    DiscreteCurlOperator(mfem::ParFiniteElementSpace *dfes,
//                         mfem::ParFiniteElementSpace *rfes);
// };

// class DiscreteDivOperator : public mfem::ParDiscreteLinearOperator
// {
// public:
//    DiscreteDivOperator(mfem::ParFiniteElementSpace *dfes,
//                        mfem::ParFiniteElementSpace *rfes);
// };

// /// This class computes the irrotational portion of a vector field.
// /// This vector field must be discretized using Nedelec basis
// /// functions.
// class IrrotationalProjector : public mfem::Operator
// {
// public:
//    // Given a GridFunction 'x' of Nedelec DoFs for an arbitrary vector field,
//    // compute the Nedelec DoFs of the irrotational portion, 'y', of
//    // this vector field.  The resulting GridFunction will satisfy Curl y = 0
//    // to machine precision.
//    void Mult(const mfem::Vector &x, mfem::Vector &y) const override;

//    /// \brief Reverse-mode differentiation of IrrotationalProjector::Mult
//    /// \param[in] proj_bar - derivative of some output w.r.t. the projection
//    /// \param[in] wrt - string indicating what to take the derivative w.r.t.
//    /// \param[inout] wrt_bar - accumulated sensitivity of output w.r.t. @a
//    wrt void vectorJacobianProduct(const mfem::ParGridFunction &proj_bar,
//                               std::string wrt,
//                               mfem::ParGridFunction &wrt_bar);

//    IrrotationalProjector(mfem::ParFiniteElementSpace &h1_fes,
//                          mfem::ParFiniteElementSpace &nd_fes,
//                          const int &ir_order);

//    void Update();

// private:
//    void InitSolver() const;

//    mfem::ParFiniteElementSpace &h1_fes;
//    mfem::ParFiniteElementSpace &nd_fes;

//    mutable mfem::ParBilinearForm s0;
//    mfem::ParMixedBilinearForm weakDiv;
//    DiscreteGradOperator grad;

//    mutable mfem::ParGridFunction psi;
//    mutable mfem::ParGridFunction xDiv;

//    mutable mfem::HypreParMatrix S0;
//    mutable mfem::Vector Psi;
//    mutable mfem::Vector RHS;

//    mutable mfem::HypreBoomerAMG amg;
//    mutable mfem::HyprePCG pcg;

//    mfem::Array<int> ess_bdr, ess_bdr_tdofs;
// };

// /// This class computes the divergence free portion of a vector field.
// /// This vector field must be discretized using Nedelec basis
// /// functions.
// class DivergenceFreeProjector : public IrrotationalProjector
// {
// public:
//    // Given a GridFunction 'x' of Nedelec DoFs for an arbitrary vector field,
//    // compute the Nedelec DoFs of the divergence free portion, 'y', of
//    // this vector field.  The resulting GridFunction will satisfy Div y = 0
//    // in a weak sense.
//    virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const;

//    /// \brief Reverse-mode differentiation of DivergenceFreeProjector::Mult
//    /// \param[in] proj_bar - derivative of some output w.r.t. the projection
//    /// \param[in] wrt - string indicating what to take the derivative w.r.t.
//    /// \param[inout] wrt_bar - accumulated sensitivity of output w.r.t. @a
//    wrt void vectorJacobianProduct(const mfem::ParGridFunction &proj_bar,
//                               std::string wrt,
//                               mfem::ParGridFunction &wrt_bar);

//    DivergenceFreeProjector(mfem::ParFiniteElementSpace &h1_fes,
//                            mfem::ParFiniteElementSpace &nd_fes,
//                            const int &ir_order);

//    void Update();
// };

/// Find root of `func` using bisection
/// \param[in] func - function to find root of
/// \param[in] xl - left bracket of root
/// \param[in] xr - right bracket of root
/// \param[in] ftol - absolute tolerance for root function
/// \param[in] xtol - absolute tolerance for root value
/// \param[in] maxiter - maximum number of iterations
double bisection(const std::function<double(double)> &func,
                 double xl,
                 double xr,
                 double ftol,
                 double xtol,
                 int maxiter);

/// Returns the root of `func(x) = 0` using the secant method.
/// \param[in] func - function to find root of
/// \param[in] x1 - first approximation of the root
/// \param[in] x2 - second approximation fo the root (x2 != x1)
/// \param[in] ftol - absolute tolerance for root function
/// \param[in] xtol - absolute tolerance for root value
/// \param[in] maxiter - maximum number of iterations
/// \note Considered converged when either `abs(func(x)) < ftol` or
/// `abs(dx) < dxtol`, where `dx` is the increment to the variable.
double secant(const std::function<double(double)> &func,
              double x1,
              double x2,
              double ftol,
              double xtol,
              int maxiter);

/// build the reconstruction matrix that interpolate the GD dofs to quadrature
/// points \param[in] degree - order of reconstructio operator \param[in] x_cent
/// - coordinates of barycenters \param[in] x_quad - coordinates of quadrature
/// points \param[out] interp - interpolation operator \note This uses minimum
/// norm reconstruction
#ifdef MFEM_USE_LAPACK
void buildInterpolation(int dim,
                        int degree,
                        const mfem::DenseMatrix &x_center,
                        const mfem::DenseMatrix &x_quad,
                        mfem::DenseMatrix &interp);
#endif

/// build the reconstruction matrix that interpolate the GD dofs to quadrature
/// points \param[in] degree - order of reconstructio operator \param[in] x_cent
/// - coordinates of barycenters \param[in] x_quad - coordinates of quadrature
/// points \param[out] interp - interpolation operator \note This uses a
/// least-squares reconstruction
#ifdef MFEM_USE_LAPACK
void buildLSInterpolation(int dim,
                          int degree,
                          const mfem::DenseMatrix &x_center,
                          const mfem::DenseMatrix &x_quad,
                          mfem::DenseMatrix &interp);
#endif

/// \brief transfer GridFunction from one mesh to another
/// \note requires MFEM to be built with GSLIB
void transferSolution(MeshType &old_mesh,
                      MeshType &new_mesh,
                      const GridFunType &in,
                      GridFunType &out);

}  // namespace mach

#endif
