#ifndef MACH_THERMAL
#define MACH_THERMAL

#include "mfem.hpp"
#include "adept.h"

#include "solver.hpp"
#include "coefficient.hpp"
#include "thermal_integ.hpp"

namespace mach
{

class ThermalSolver : public AbstractSolver
{
public:
	/// Class constructor.
   /// \param[in] opt_file_name - file where options are stored
   /// \param[in] smesh - if provided, defines the mesh for the problem
   /// \param[in] dim - number of dimensions
   ThermalSolver(const std::string &opt_file_name,
                       std::unique_ptr<mfem::Mesh> smesh = nullptr,
							  int dim = 3);


private:
   // /// `bndry_marker[i]` lists the boundaries associated with a particular BC
   // std::vector<mfem::Array<int>> bndry_marker;
   // /// the mass matrix bilinear form
   // std::unique_ptr<BilinearFormType> mass;
   // /// mass matrix (move to AbstractSolver?)
   // std::unique_ptr<MatrixType> mass_matrix;

   /// H(grad) finite element collection
   std::unique_ptr<mfem::FiniteElementCollection> h_grad_coll;
   /// H(grad) finite element space
   std::unique_ptr<SpaceType> h_grad_space;

   /// Temperature T grid function and derivative
   std::unique_ptr<GridFunType> T;
   std::unique_ptr<GridFunType> dTdt;
   std::unique_ptr<GridFunType> rhs;

   OperatorPtr A;
   Vector X, B;


   /// mesh dependent mass density coefficient
   std::unique_ptr<MeshDependentCoefficient> rho_cv;

   /// mesh dependent thermal conductivity tensor
   std::unique_ptr<MeshDependentCoefficient> kappa;

   /// mesh dependent electrical conductivity tensor
   std::unique_ptr<MeshDependentCoefficient> sigma;


   /// essential boundary condition marker array (not using )
   mfem::Array<int> ess_bdr = 0;
   std::unique_ptr<mfem::Coefficient> bc_coef;

   /// the bilinear forms, mass m, stiffness k
   std::unique_ptr<BilinearFormType> m;
   std::unique_ptr<BilinearFormType> k;
   /// the linear form
   std::unique_ptr<LinearFormType> b;
   /// linear system solver
   std::unique_ptr<mfem::HypreGMRES> solver;
//    /// linear system preconditioner used in Newton's method
//    std::unique_ptr<EMPrecType> prec;

   /// time marching method
   std::unique_ptr<mfem::ODESolver> ode_solver;

   /// time dependent operator
   std::unique_ptr<mfem::TimeDependentOperator> evolver;

   /// boundary condition marker array
   std::vector<mfem::Array<int>> bndry_marker;

   /// construct mesh dependent coefficient for density and specific heat
   void constructMassCoeff();

   /// construct vector mesh dependent coefficient for conductivity
   void constructConductivity();
     
   /// construct vector mesh dependent coefficient for electrical conductivity
   void constructElecConductivity();

   /// set up solver for every time step
   void setupSolver(const int idt, const double dt) const;

   /// implementation of implicitsolve
   virtual void ImplicitSolve(const double dt, const Vector &x, Vector &k);

   /// TODO: Source Terms, Flux Boundary Conditions
};

} // namespace mach

#endif