#ifndef MACH_MESH_MOVEMENT
#define MACH_MESH_MOVEMENT

#include "mfem.hpp"
#include "adept.h"
#include "egads.h"

#ifdef MACH_USE_EGADS
#include "mach_egads.h"
#endif

#include "solver.hpp"
#include "coefficient.hpp"

namespace mach
{

class MeshMovementSolver : public AbstractSolver
{
protected:
    /// Base Class Constructor
    MeshMovementSolver(const std::string &opt_file_name,
                       std::unique_ptr<mfem::Mesh> smesh = nullptr,
					    int dim = 3) 
                              : AbstractSolver(opt_file_name, move(smesh))
    {
        /// Fill as would be useful
        fes = NULL; 
    }
};

#ifdef MACH_USE_EGADS
class LEAnalogySolver : public MeshMovementSolver
{
public:
    /// Class Constructor
    LEAnalogySolver(const std::string &opt_file_name,
                       std::unique_ptr<mfem::Mesh> smesh = nullptr,
					    
                        int dim = 3);

private:
   // /// `bndry_marker[i]` lists the boundaries associated with a particular BC
   std::ofstream sol_ofs;

   /// H(grad) finite element collection
   std::unique_ptr<mfem::FiniteElementCollection> h_grad_coll;
   /// H(grad) finite element space
   std::unique_ptr<SpaceType> h_grad_space;

   /// displacement u grid function
   std::unique_ptr<GridFunType> u;

   mfem::HypreParMatrix K;
   mfem::Vector B;

   ///

   /// mesh dependent density coefficient
   /// TODO: Do we assign an attribute to every element and use this?
   /// Or maybe just see how uniform kappa works
   std::unique_ptr<MeshDependentCoefficient> kappa;

   /// essential boundary condition marker array 
   // (whole boundary, plus interior faces)
   std::unique_ptr<mfem::Coefficient> bc_coef;

   /// the bilinear form stiffness k
   std::unique_ptr<BilinearFormType> k;
   
   /// the source term linear form (0)
   std::unique_ptr<mfem::LinearForm> bs;

   /// linear solver
   std::unique_ptr<mfem::Solver> solver;
   
   /// linear system preconditioner
   std::unique_ptr<mfem::Solver> prec;

   /// static variables for use in static member functions
   static double something;

   /// set static variables
   void setStaticMembers() { }

   /// construct element-level coefficient for stiffness
   void constructStiffnessCoeff();

   /// implementation of solveUnsteady
   virtual void solveSteady();

   /// work vector
   mutable mfem::Vector z;
    
};

#endif

} //namespace mach

#endif