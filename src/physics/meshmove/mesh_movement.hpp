#ifndef MISO_MESH_MOVEMENT
#define MISO_MESH_MOVEMENT

#include "mfem.hpp"

#include "solver.hpp"

namespace mfem
{
class Coefficient;
}  // namespace mfem

namespace miso
{
class MeshMovementSolver : public AbstractSolver
{
protected:
   /// Base Class Constructor
   MeshMovementSolver(const nlohmann::json &options,
                      std::unique_ptr<mfem::Mesh> smesh,
                      MPI_Comm comm)
    : AbstractSolver(options, move(smesh), comm)
   { }
};

// #ifdef MFEM_USE_PUMI
// #ifdef MFEM_USE_EGADS
class LEAnalogySolver : public MeshMovementSolver
{
public:
   /// Class Constructor
   LEAnalogySolver(const nlohmann::json &options,
                   std::unique_ptr<mfem::Mesh> smesh,
                   MPI_Comm comm);

   LEAnalogySolver(const std::string &opt_file_name,
                   mfem::GridFunction *u_bound,
                   std::unique_ptr<mfem::Mesh> smesh = nullptr,
                   int dim = 3);

   void setInitialCondition(
       mfem::ParGridFunction &state,
       const std::function<void(const mfem::Vector &, mfem::Vector &)> &u_init)
       override;

   double calcStepSize(int iter,
                       double t,
                       double t_final,
                       double dt_old,
                       const mfem::ParGridFunction &state) const override;

private:
   int dim;

   // /// `bndry_marker[i]` lists the boundaries associated with a particular BC
   std::ofstream sol_ofs;

   /// Stiffness coefficients
   std::unique_ptr<mfem::Coefficient> lambda_c;
   std::unique_ptr<mfem::Coefficient> mu_c;

   void constructCoefficients() override;

   void constructForms() override;

   void addMassIntegrators(double alpha) override;

   void addResVolumeIntegrators(double alpha) override;

   void setEssentialBoundaries() override;

   int getNumState() override { return dim; }

   double res_norm0 = -1.0;
   void initialHook(const mfem::ParGridFunction &state) override;

   // void iterationHook(int iter, double t, double dt,
   //                    const mfem::ParGridFunction &state) override;

   bool iterationExit(int iter,
                      double t,
                      double t_final,
                      double dt,
                      const mfem::ParGridFunction &state) const override;

   // void terminalHook(int iter, double t_final,
   //                   const mfem::ParGridFunction &state) override;

   // // Lambda element wise function
   // static double LambdaFunc(const mfem::Vector &x, int ie);

   // // Mu element wise function
   // static double MuFunc(const mfem::Vector &x, int ie);
};

}  // namespace miso

#endif