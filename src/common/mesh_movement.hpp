#ifndef MACH_MESH_MOVEMENT
#define MACH_MESH_MOVEMENT

#include "mfem.hpp"
#include "adept.h"
#include "../../build/_config.hpp"

#ifdef MFEM_USE_EGADS
#include "egads.h"
#include "mach_egads.hpp"
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

public:
    virtual void initDerived();

    void setMesh(MeshType* mmesh)
    {
        mesh.reset(new MeshType(*mmesh));
    }

    MeshType* getMesh()
    {
        return mesh.get();
    }

    /// must call before computing adjoint
    void setSens(mfem::GridFunction* sens)
    {
        dLdX.reset(new GridFunType(fes.get(), *sens));
    }

    /// retrieve adjoint vector
    GridFunType* getAdjoint()
    {
        return adj.get();
    }
};

#ifdef MFEM_USE_PUMI
#ifdef MFEM_USE_EGADS
class LEAnalogySolver : public MeshMovementSolver
{
public:
    /// Class Constructor
    LEAnalogySolver(const std::string &opt_file_name,
                       std::unique_ptr<mfem::Mesh> smesh = nullptr,
                        int dim = 3);

    LEAnalogySolver(const std::string &opt_file_name,
                        mfem::GridFunction *u_bound,
                       std::unique_ptr<mfem::Mesh> smesh = nullptr,
                        int dim = 3);

    /// Implement InitDerived
    virtual void initDerived();

protected:
    /// passed in boundary displacement array
    mfem::GridFunction *u_bnd;

private:
    /// Moved mesh
    apf::Mesh2 *moved_mesh;

    // static copy of the original mesh
    static mfem::Mesh *mesh_copy;

   // /// `bndry_marker[i]` lists the boundaries associated with a particular BC
   std::ofstream sol_ofs;

   /// H(grad) finite element collection
   std::unique_ptr<mfem::FiniteElementCollection> h_grad_coll;
   /// H(grad) finite element space
   std::unique_ptr<SpaceType> h_grad_space;

   /// displacement u grid function
   std::unique_ptr<GridFunType> u;

   MatrixType K;
   mfem::Vector B, U;

   /// list of boundary displacements
   mfem::Array<mfem::Vector> disp_list;

   /// mesh dependent density coefficient
   /// TODO: Do we assign an attribute to every element and use this?
   /// Or maybe just see how uniform kappa works
   std::unique_ptr<MeshDependentCoefficient> kappa;

   /// essential boundary condition marker array 
   // (whole boundary, plus interior faces)
   std::unique_ptr<mfem::VectorCoefficient> bc_coef;

   /// the bilinear form stiffness k
   std::unique_ptr<BilinearFormType> k;
   
   /// the source term linear form (0)
   std::unique_ptr<mfem::LinearForm> bs;

   /// linear solver
   std::unique_ptr<mfem::CGSolver> solver;
   
   /// linear system preconditioner
   std::unique_ptr<SmootherType> prec;

   /// Stiffness coefficients
   std::unique_ptr<mfem::Coefficient> lambda_c;
   std::unique_ptr<mfem::Coefficient> mu_c;

   /// static variables for use in static member functions
   static double something;

   /// set static variables
   void setStaticMembers() { }

   virtual int getNumState() {return 1; }

   /// construct element-level coefficient for stiffness
   void constructStiffnessCoeff();

   /// implementation of solveUnsteady
   virtual void solveSteady();

   /// implementation of solveUnsteadyAdjoint
   virtual void solveSteadyAdjoint(const std::string &fun);

   /// work vector
   mutable mfem::Vector z;
    
    // Lambda element wise function
    static double LambdaFunc(const mfem::Vector &x, int ie);

    // Mu element wise function
    static double MuFunc(const mfem::Vector &x, int ie);
};

#endif
#endif

} //namespace mach

#endif