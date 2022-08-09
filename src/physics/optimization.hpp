#ifndef MACH_OPTIMIZATION
#define MACH_OPTIMIZATION


#include <fstream>
#include <iostream>
#include <iomanip>

#include "adept.h"
#include "mfem.hpp"
#include "galer_diff.hpp"
#include "centgridfunc.hpp"

#include "utils.hpp"
#include "json.hpp"
#include "mach_types.hpp"
namespace mach
{

class DGDOptimizer : public mfem::Operator
{
public:
   /// class constructor
   /// \param[in] opt_file_name - option file
   /// \param[in] initail - initial condition
   /// \param[in] smesh - mesh file
   DGDOptimizer(mfem::Vector init,
                const std::string &opt_file_name =
                  std::string("mach_optionš.json"),
                std::unique_ptr<mfem::Mesh> smesh = nullptr);
   
   void InitializeSolver();
   void SetInitialCondition(void (*u_init)(const mfem::Vector &,
                                           mfem::Vector &));
   
   void SetInitialCondition(const mfem::Vector uic);
   
   virtual double GetEnergy(const mfem::Vector &x) const;

   /// compute the jacobian of the functional w.r.t the design variable
   virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const;

   void addVolumeIntegrators(double alpha);
   void addBoundaryIntegrators(double alpha);
   void addInterfaceIntegrators(double alpha);
   void addOutputs();

   void updateStencil(const mfem::Vector &basisCenter) const;
   void reSolve() const;
   double computeObj() const;

   void checkJacobian(mfem::Vector &x) const;
   void getFreeStreamState(mfem::Vector &q_ref);

   double calcFunctional() const;
   double calcFullSpaceL2Error(int entry) const;


   void printSolution(const mfem::Vector &c, const std::string &file_name);
   /// class destructor
   ~DGDOptimizer();
   
protected:
   nlohmann::json options;
   static adept::Stack diff_stack;
   int dim;
   bool entvar;
   int num_state;
   int ROMSize;
   int numBasis;
   int FullSize;
   int numDesignVar;
   mfem::Vector designVar;

   void (*u_exact)(const mfem::Vector &, mfem::Vector &);

   // airfoil variables
   double mach_fs;
   double aoa_fs;
   int iroll;
   int ipitch;

   // aux variables
   std::map<std::string, NonlinearFormType> output;
   std::vector<mfem::Array<int>> bndry_marker;
   std::vector<mfem::Array<int>> output_bndry_marker;


   std::unique_ptr<mfem::Mesh> mesh;
   std::unique_ptr<mfem::FiniteElementCollection> fec;
   std::unique_ptr<mfem::DGDSpace> fes_dgd;
   std::unique_ptr<mfem::FiniteElementSpace> fes_full;

	// the constraints
   std::unique_ptr<NonlinearFormType> res_dgd;
	std::unique_ptr<NonlinearFormType> res_full;

   // some working variables
   std::unique_ptr<mfem::CentGridFunction> u_dgd;
   std::unique_ptr<mfem::GridFunction> u_full;

   // nonlinear and linear solver
   std::unique_ptr<mfem::NewtonSolver> newton_solver;
   std::unique_ptr<mfem::UMFPackSolver> solver;
};

} // end of namesapce
#endif