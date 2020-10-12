#ifndef MACH_THERMAL
#define MACH_THERMAL

#include "mfem.hpp"
#include "adept.h"

#include "solver.hpp"
#include "evolver.hpp"
#include "coefficient.hpp"
#include "therm_integ.hpp"
#include "temp_integ.hpp"
#include "res_integ.hpp"
#include "mesh_movement.hpp"

#include <limits>
#include <random>

namespace mach
{

class ThermalSolver : public AbstractSolver
{
public:
   /// Class constructor.
   /// \param[in] options - pre-loaded JSON options object
   /// \param[in] smesh - if provided, defines the mesh for the problem
   /// \param[in] B - pointer to magnetic field grid function from EM solver
   ThermalSolver(const nlohmann::json &options,
                 std::unique_ptr<mfem::Mesh> smesh,
                 MPI_Comm comm);
   
   void initDerived() override;

   /// Set the Magnetic Vector Potential
   void setAField(GridFunType *_a_field) {a_field = _a_field;}

   // /// Returns the L2 error between the state `u` and given exact solution.
   // /// Overload for scalar quantities
   // /// \param[in] u_exact - function that defines the exact solution
   // /// \param[in] entry - if >= 0, the L2 error of state `entry` is returned
   // /// \returns L2 error
   // double calcL2Error(double (*u_exact)(const mfem::Vector &),
   //                    int entry = -1);

   /// \brief Returns a vector of pointers to grid functions that define fields
   /// returns {theta, B}
   std::vector<GridFunType*> getFields() override;

   /// Compute the sensitivity of the aggregate temperature output to the mesh 
   /// nodes, using appropriate mesh sensitivity integrators. Need to compute 
   /// the adjoint first.
   mfem::Vector* getMeshSensitivities() override;

   mfem::Vector* getSurfaceMeshSensitivities();

   void getASensitivity(mfem::Vector &psiTdRtdA) {};

   double getOutput();

   /// perturb the whole mesh and finite difference
   void verifyMeshSensitivities();

   /// perturb the surface mesh, deform interior points, and finite difference
   void verifySurfaceMeshSensitivities();

private:
   /// Magnetic vector potential A grid function (not owned)
   GridFunType *a_field;

   /// Use for exact solution
   std::unique_ptr<GridFunType> th_exact;

   /// aggregation functional (aggregation or temp)
   std::unique_ptr<AggregateIntegrator> funca;
   std::unique_ptr<TempIntegrator> funct;

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

   /// static variables for use in static member functions
   // static double temp_0;

   double dt_real_;

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

   void constructForms() override;

   void addMassIntegrators(double alpha) override;
   void addStiffVolumeIntegrators(double alpha) override;
   void addLoadVolumeIntegrators(double alpha) override;
   void addLoadBoundaryIntegrators(double alpha) override;
   void constructEvolver() override;

   /// compute outward flux at boundary
   // static void FluxFunc(const mfem::Vector &x, mfem::Vector &y );
   static void testFluxFunc(const mfem::Vector &x, double time, mfem::Vector &y);

   /// initial temperature
   // static double initialTemperature(const mfem::Vector &x);

   /// implementation of solveUnsteady
   void solveUnsteady(mfem::ParGridFunction &state) override;

   /// Return the number of state variables
   int getNumState() override { return 1; }

   /// implementation of solveUnsteadyAdjoint, call only after solveForState
   virtual void solveUnsteadyAdjoint(const std::string &fun) override;

   /// implementation of addOutputs
   virtual void addOutputs() override;

   /// aggregation parameter
   double rhoa;

   /// vector of maximum temperature constraints
   mfem::Vector max;

   /// work vector
   mutable mfem::Vector z;

   friend SolverPtr createSolver<ThermalSolver>(
       const nlohmann::json &opt_file_name,
       std::unique_ptr<mfem::Mesh> smesh,
       MPI_Comm comm);
   /// functions for random perturbation
   static void randState(const mfem::Vector &x, mfem::Vector &u);
};

class ThermalEvolver : public MachEvolver
{
public:
   ThermalEvolver(mfem::Array<int> ess_bdr, BilinearFormType *mass, BilinearFormType *stiff,
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

   mfem::Array<int> ess_tdof_list;

   mutable mfem::Vector work; // auxiliary vector

   MatrixType mMat;
   MatrixType kMat;

   
};

} // namespace mach

#endif
