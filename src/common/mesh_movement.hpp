#ifndef MACH_MESH_MOVEMENT
#define MACH_MESH_MOVEMENT

#include "mfem.hpp"

#include "solver.hpp"

namespace mfem
{
class Coefficient;
} // namespace mfem

namespace mach
{

class MeshMovementSolver : public AbstractSolver
{
protected:
   /// Base Class Constructor
   MeshMovementSolver(const nlohmann::json &options,
                      std::unique_ptr<mfem::Mesh> smesh,
                      MPI_Comm comm)
   : AbstractSolver(options, move(smesh), comm)
   {
      /// Fill as would be useful
      // fes = NULL; 
   }
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

   // void addStiffVolumeIntegrators(double alpha) override;
   void addResVolumeIntegrators(double alpha) override;

   void setEssentialBoundaries() override;

   int getNumState() override { return dim; }

   // // Lambda element wise function
   // static double LambdaFunc(const mfem::Vector &x, int ie);

   // // Mu element wise function
   // static double MuFunc(const mfem::Vector &x, int ie);
};

} //namespace mach

#endif