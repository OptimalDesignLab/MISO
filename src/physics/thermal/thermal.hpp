#ifndef MACH_THERMAL
#define MACH_THERMAL

#include "mfem.hpp"
#include "adept.h"

#include "solver.hpp"
#include "evolver.hpp"
#include "coefficient.hpp"
#include "therm_integ.hpp"
#include "temp_integ.hpp"

namespace mach
{

class ThermalSolver : public AbstractSolver
{
public:
	/// Class constructor.
   /// \param[in] opt_file_name - file where options are stored
   /// \param[in] smesh - if provided, defines the mesh for the problem
   /// \param[in] dim - number of dimensions
   /// \param[in] B - pointer to magnetic field grid function from EM solver
   ThermalSolver(const std::string &opt_file_name,
                       std::unique_ptr<mfem::Mesh> smesh = nullptr,
							  int dim = 3,
                       GridFunType *B = nullptr);
   
   /// Returns the L2 error between the state `u` and given exact solution.
   /// Overload for scalar quantities
   /// \param[in] u_exact - function that defines the exact solution
   /// \param[in] entry - if >= 0, the L2 error of state `entry` is returned
   /// \returns L2 error
   double calcL2Error(double (*u_exact)(const mfem::Vector &),
                      int entry = -1);

private:
   // std::ofstream sol_ofs;

   /// Magnetic flux density B grid function;
   GridFunType *mag_field;

   /// Use for exact solution
   std::unique_ptr<GridFunType> th_exact;

   /// TODO: Need these?
   // mfem::HypreParMatrix M;
   // mfem::HypreParMatrix K;
   // mfem::Vector B;

   /// TODO: don't think this should be a unique ptr, nonlinear form will delete
   /// aggregation functional
   std::unique_ptr<AggregateIntegrator> func;

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
   /// mesh dependent core losses term
   std::unique_ptr<MeshDependentCoefficient> coreloss;
   /// mesh dependent 
   std::unique_ptr<MeshDependentCoefficient> sigmainv;
   /// natural bc coefficient
   std::unique_ptr<mfem::VectorCoefficient> flux_coeff;


   /// essential boundary condition marker array (not using )
   std::unique_ptr<mfem::Coefficient> bc_coef;

   /// the bilinear forms, mass m, stiffness k
   // std::unique_ptr<BilinearFormType> m;
   // std::unique_ptr<BilinearFormType> k;
   /// the source term linear form
   // std::unique_ptr<mfem::LinearForm> bs;

   /// time marching method
   // std::unique_ptr<mfem::ODESolver> ode_solver;

   /// TODO: use from abstract class?
   /// time dependent operator
   // std::unique_ptr<ImplicitLinearEvolver> evolver;

   /// static variables for use in static member functions
   // static double temp_0;

   /// maximum magnetic flux density aka "amplitude"
   // double Bmax;

   /// check if initial conditions are set
   bool setInit;

   // /// set static variables
   // void setStaticMembers();

   /// construct mesh dependent coefficient for density
   void constructDensityCoeff();
   /// construct mesh dependent coefficient for specific heat
   void constructHeatCoeff();
   /// construct mesh dependent coefficient for density and specific heat
   void constructMassCoeff();
   /// construct mesh dependent coefficient for conductivity
   void constructConductivity();
   /// construct mesh dependent coefficient for joule heating
   void constructJoule();
   /// construct mesh dependent coefficient for core loss heating
   void constructCore();

   /// Construct all coefficients for thermal solver
   void constructCoefficients() override;

   void addMassVolumeIntegrators() override;
   void addStiffVolumeIntegrators(double alpha) override;
   void addLoadVolumeIntegrators(double alpha) override;
   void addLoadBoundaryIntegrators(double alpha) override;
   void constructEvolver() override;

   // /// for calls of mult
   // void Mult(const mfem::Vector &X, mfem::Vector &dXdt);

   /// compute outward flux at boundary
   // static void FluxFunc(const mfem::Vector &x, mfem::Vector &y );

   /// initial temperature
   // static double initialTemperature(const mfem::Vector &x);

   /// implementation of solveUnsteady
   void solveUnsteady() override;

   /// Return the number of state variables
   int getNumState() override { return 1; }

   /// aggregation parameter
   double rhoa;

   /// vector of maximum temperature constraints
   mfem::Vector max;

   /// work vector
   mutable mfem::Vector z;
};

class ThermalEvolver : public ImplicitLinearEvolver
{
public:
   ThermalEvolver(BilinearFormType *mass, BilinearFormType *stiff,
                  mfem::Vector *load, std::ostream &outstream,
                  double start_time, mfem::VectorCoefficient *flux_coeff);

   void Mult(const mfem::Vector &x, mfem::Vector &y) const override;

   void ImplicitSolve(const double dt, const mfem::Vector &x,
                      mfem::Vector &k) override;

   /// set updated parameters at time step, specifically boundary conditions
   // void updateParameters();

   /// set static member values
   // void setStaticMembers();

private:   
   /// static variables for use in static member functions
   static double outflux;
   mfem::VectorCoefficient *flux_coeff;
   
};

} // namespace mach

#endif