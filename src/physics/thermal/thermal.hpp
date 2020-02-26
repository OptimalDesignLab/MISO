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
   ThermalSolver(const std::string &opt_file_name,
                       std::unique_ptr<mfem::Mesh> smesh = nullptr,
							  int dim = 3);

   /// Set initial temperature
   /// \param[in] f - static user function that defines the initial condition
   void setInitialTemperature(double (*f)(const mfem::Vector &));

   /// Returns the L2 error between the state `u` and given exact solution.
   /// Overload for scalar quantities
   /// \param[in] u_exact - function that defines the exact solution
   /// \param[in] entry - if >= 0, the L2 error of state `entry` is returned
   /// \returns L2 error
   double calcL2Error(double (*u_exact)(const mfem::Vector &),
                      int entry = -1);

private:
   // /// `bndry_marker[i]` lists the boundaries associated with a particular BC
   std::ofstream sol_ofs;

   /// H(grad) finite element collection
   std::unique_ptr<mfem::FiniteElementCollection> h_grad_coll;
   /// H(grad) finite element space
   std::unique_ptr<SpaceType> h_grad_space;

   /// Temperature theta grid function
   std::unique_ptr<GridFunType> theta;

   /// Magnetic flux density B grid function;
   std::unique_ptr<GridFunType> Bfield; //needed?

   /// Use for exact solution
   std::unique_ptr<GridFunType> th_exact;

   mfem::HypreParMatrix M;
   mfem::HypreParMatrix K;
   mfem::Vector B;

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

   /// essential boundary condition marker array (not using )
   std::unique_ptr<mfem::Coefficient> bc_coef;

   /// the bilinear forms, mass m, stiffness k
   std::unique_ptr<BilinearFormType> m;
   std::unique_ptr<BilinearFormType> k;
   /// the source term linear form
   std::unique_ptr<mfem::LinearForm> bs;

   /// time marching method
   std::unique_ptr<mfem::ODESolver> ode_solver;

   /// time dependent operator
   std::unique_ptr<ImplicitLinearEvolver> evolver;

   /// material Library
   nlohmann::json materials;

   /// static variables for use in static member functions
   static double temp_0;

   /// maximum magnetic flux density aka "amplitude"
   double Bmax;

   /// check if initial conditions are set
   bool setInit;

   /// set static variables
   void setStaticMembers();

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

   /// for calls of mult
   void Mult(const mfem::Vector &X, mfem::Vector &dXdt);

   /// compute outward flux at boundary
   static void FluxFunc(const mfem::Vector &x, mfem::Vector &y );

   /// initial temperature
   static double InitialTemperature(const mfem::Vector &x);

   /// implementation of solveUnsteady
   virtual void solveUnsteady();

   /// Return the number of state variables
   virtual int getNumState() {return 1; }

   /// aggregation parameter
   double rhoa;

   /// work vector
   mutable mfem::Vector z;
};

class ConductionEvolver : public ImplicitLinearEvolver
{
public:
   ConductionEvolver(const std::string &opt_file_name, 
                     MatrixType &m, MatrixType &k, 
                     std::unique_ptr<mfem::LinearForm> b, 
                     std::ostream &outstream);

   /// set updated parameters at time step, specifically boundary conditions
   void updateParameters();

   /// set static member values
   void setStaticMembers();

private:
   /// compute outward flux at boundary
   static void fluxFunc(const mfem::Vector &x, double time, mfem::Vector &y);
   
   /// static variables for use in static member functions
   static double outflux;

   /// the natural boundary linear form
   std::unique_ptr<mfem::LinearForm> bb;

   /// natural bc coefficient
   std::unique_ptr<mfem::VectorCoefficient> fluxcoeff;
   
   /// boundary condition marker array
   std::vector<mfem::Array<int>> bndry_marker;

   mfem::Vector zero;
};

} // namespace mach

#endif