#ifndef MACH_THERMAL
#define MACH_THERMAL

#include "mfem.hpp"
#include "adept.h"

#include "solver.hpp"
#include "coefficient.hpp"
#include "therm_integ.hpp"

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

   /// Temperature phi grid function and derivative
   std::unique_ptr<GridFunType> phi;
   std::unique_ptr<GridFunType> dTdt;
   std::unique_ptr<GridFunType> rhs;

   mfem::HypreParMatrix M;
   mfem::HypreParMatrix K;
   mfem::HypreParMatrix *T;
   mfem::Vector B;


   /// mesh dependent density coefficient
   std::unique_ptr<MeshDependentCoefficient> rho;

   /// mesh dependent specific heat coefficient
   std::unique_ptr<MeshDependentCoefficient> cv;

   /// mesh dependent mass*specificheat coefficient
   std::unique_ptr<MeshDependentCoefficient> rho_cv;

   /// mesh dependent thermal conductivity tensor
   std::unique_ptr<MeshDependentCoefficient> kappa;

   /// mesh dependent i^2(1/sigma) term (purely scalar)
   std::unique_ptr<MeshDependentCoefficient> i2sigmainv;

   /// mesh dependent 
   std::unique_ptr<MeshDependentCoefficient> sigmainv;

   /// natural bc coefficient
   std::unique_ptr<mfem::VectorCoefficient> fluxcoeff;

   /// essential boundary condition marker array (not using )
   mfem::Array<int> ess_bdr;
   std::unique_ptr<mfem::Coefficient> bc_coef;

   /// the bilinear forms, mass m, stiffness k
   std::unique_ptr<BilinearFormType> m;
   std::unique_ptr<BilinearFormType> k;
   /// the linear form
   std::unique_ptr<mfem::LinearForm> b;
   
   /// linear system solvers and preconditioners
   std::unique_ptr<mfem::CGSolver> M_solver;
   std::unique_ptr<mfem::HypreSmoother> M_prec;
   std::unique_ptr<mfem::CGSolver> T_solver;
   std::unique_ptr<mfem::HypreSmoother> T_prec;

   /// time marching method
   std::unique_ptr<mfem::ODESolver> ode_solver;

   /// time dependent operator
   std::unique_ptr<mfem::TimeDependentOperator> evolver;

   /// boundary condition marker array
   std::vector<mfem::Array<int>> bndry_marker;

   /// material Library
   nlohmann::json materials;

   /// construct mesh dependent coefficient for density
   void constructDensityCoeff();

   /// construct mesh dependent coefficient for specific heat
   void constructHeatCoeff();

   /// construct mesh dependent coefficient for density and specific heat
   void constructMassCoeff();

   /// construct vector mesh dependent coefficient for conductivity
   void constructConductivity();
     
   /// construct vector mesh dependent coefficient for joule heating
   void constructJoule();

   /// compute outward flux at boundary, for 
   void FluxFunc(const mfem::Vector &x, mfem::Vector &y );

   /// set up solver for every time step
   void setupSolver(const int idt, const double dt) const;

   /// for calls of mult
   void Mult(const mfem::Vector &X, mfem::Vector &dXdt);

   /// implementation of ImplicitSolve
   virtual void ImplicitSolve(const double dt, const mfem::Vector &x, mfem::Vector &k);

   /// implementation of solveUnsteady
   virtual void solveUnsteady();

   /// work vector
   mutable mfem::Vector z;
};

} // namespace mach

#endif