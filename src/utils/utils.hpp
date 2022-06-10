#ifndef MACH_UTILS
#define MACH_UTILS

#include <any>
#include <functional>
#include <exception>
#include <iostream>
#include <utility>
#include <variant>
#include <memory>

#include "mfem.hpp"
#include "nlohmann/json.hpp"

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

/// Evaluate the squared exponential exp(- ||x - xc||^2 /len^2 )
/// \param[in] len - a length scale for the squared exponential
/// \param[in] xc - the center for this radial basis function
/// \param[in] x - the point at which to evaluate the function
double squaredExponential(double len,
                          const mfem::Vector &xc,
                          const mfem::Vector &x);

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

/// This group of code enables the conversion from lambdas, member functions,
/// and function pointers to std::functions of the appropriate signature,
/// deduced at compile time.
/// These `function` structs deduce the return and argument types based on the
/// input, and define their `type` member to be a std::function with the
/// matching return and argument types. This allows `make_function` to
/// deduce the appropriate `function` from the callable input, and using that
/// `function`'s type, construct a std::function from the input
namespace detail
{
template <typename T, typename... Ts>
struct first
{
   using type = T;
};

template <typename... T>
using first_t = typename first<T...>::type;

/// If a type cannot be converted to a function, this definition will be used
/// which does not contain a `type` definition, so code that tries to access
/// the function's type will not compile with a template substituion failure
template <typename /*unused*/, typename = void>
struct function
{ };

/// For generic types that are functors, delegate to its 'operator()'
/// (if defined)
template <typename T>
struct function<T, std::void_t<decltype(&T::operator())>>
 : public function<decltype(&T::operator())>
{ };

/// For const pointers to a member function
template <typename ClassType, typename ReturnType, typename... Args>
struct function<ReturnType (ClassType::*)(Args...) const>
{
   using type = std::function<ReturnType(Args...)>;
   using return_t = ReturnType;
   using arg_t = first_t<Args...>;
};

/// For pointers to a member function
template <typename ClassType, typename ReturnType, typename... Args>
struct function<ReturnType (ClassType::*)(Args...)>
{
   using type = std::function<ReturnType(Args...)>;
   using return_t = ReturnType;
   using arg_t = first_t<Args...>;
};

/// For function pointers
template <typename ReturnType, typename... Args>
struct function<ReturnType (*)(Args...)>
{
   using type = std::function<ReturnType(Args...)>;
   using return_t = ReturnType;
   using arg_t = first_t<Args...>;
};

}  // namespace detail

/// \brief converts a callable object @a fun into a std::function with the same
/// signature needed to call @a fun
/// \param[in] fun - callable object to be converted to std::function
/// \tparam T - generic callable type
/// \return std::function wrapping @a fun with appropriately deduced signature
template <typename T>
typename detail::function<T>::type make_function(T fun)
{
   return (typename detail::function<T>::type)(fun);
}

/// Compile time check if T is callable
/// \tparam T - generic (maybe callable) type
/// \note A type being callable means it defines a call operator or can be
/// converted to a function
template <typename /*unused*/, typename = void>
struct is_callable : std::false_type
{ };

/// Compile time check if T is callable
/// \tparam T - generic (maybe callable) type
/// For types that can be converted to a function
/// \note A type being callable means it defines a call operator or can be
/// converted to a function
template <typename T>
struct is_callable<T, std::void_t<typename detail::function<T>::type>>
 : std::true_type
{ };

/// Shorthand for `is_callable` that accesses the underlying value
/// \tparam T - generic (maybe callable) type
template <typename T>
inline constexpr bool is_callable_v = is_callable<T>::value;

/// \brief Convenience function that handles casting std::any to a usable type
/// and then applies a user supplied function on the casted any
/// \param[in] any - std::any that holds some concrete type
/// \param[in] lambda - user supplied function to be called with the result of
/// the any cast as its argument
/// \param[in] rest... - variadic argument that can accept any number of
/// aditional lambdas to try if the previous ones have not been callable with
/// the result of the any cast
/// \tparam T - generic callable type
/// \tparam Ts... - variadic template types for additional callable types
/// \return whatever the user supplied lambdas return, type is deduced from the
/// return type of the lambdas
/// \note In the case where the user supplied lambdas return a double, this
/// function will return NAN if all the any casts field
template <typename T, typename... Ts>
auto useAny(std::any &any, T lambda, Ts... rest) ->
    typename detail::function<T>::return_t
{
   /// For the first lambda given, we deduce the type of its argument,
   /// removing const and reference qualifiers
   using arg_t = std::remove_reference_t<
       std::remove_const_t<typename detail::function<T>::arg_t>>;

   /// Then we try to cast the any to that argument type
   auto *concrete = std::any_cast<arg_t>(&any);

   /// If the cast succeeds, we call the lambda with that argument
   if (concrete != nullptr)
   {
      return lambda(*concrete);
   }

   /// If the cast failed, we recursively call this function with the next user
   /// supplied lambda until we've exhausted all lambdas
   if constexpr (sizeof...(rest) > 0)
   {
      return useAny(any, rest...);
   }

   /// If all any_casts failed, and the return type is a double
   /// we return a NAN to indicate this function couldn't use the any
   using return_t = decltype(useAny(any, lambda));
   if constexpr (std::is_same_v<return_t, double>)
   {
      return NAN;
   }
}

/// \brief helper function to populate the @a ess_bdr array based on options
/// \param[in] options - options dictionary containing "ess-bdr" key
/// \param[out] ess_bdr - binary array that marks essential boundaries
void getEssentialBoundaries(const nlohmann::json &options,
                            mfem::Array<int> &ess_bdr);

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

/// Generate quarter annulus mesh, \f$r \in [1,3], \theta \in [0,\pi/2]\f$.
/// \param[in] degree - polynomial degree of the mapping
/// \param[in] num_rad - number of nodes in the radial direction
/// \param[in] num_ang - number of nodes in the angular direction
/// \param[in] pert - perturbs interior vertices by at most `pert` fraction of h
/// \returns unique pointer to a serial `Mesh` object.
std::unique_ptr<mfem::Mesh> buildQuarterAnnulusMesh(int degree,
                                                    int num_rad,
                                                    int num_ang,
                                                    double pert = 0.0);

}  // namespace mach

#endif
