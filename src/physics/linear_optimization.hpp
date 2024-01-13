#ifndef MACH_LINEAR_OPTIMIZATION
#define MACH_LINEAR_OPTIMIZATION


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

class LinearOptimizer : public mfem::Operator
{
public:
   /// class constructor
   /// \param[in] opt_file_name - option file
   /// \param[in] initail - initial condition
   /// \param[in] smesh - mesh file
   LinearOptimizer(mfem::Vector init,
                const std::string &opt_file_name =
                  std::string("mach_optionš.json"),
                std::unique_ptr<mfem::Mesh> smesh = nullptr);
   
   void InitializeSolver(mfem::VectorFunctionCoefficient& velocity, mfem::FunctionCoefficient& inflow);

   void SetInitialCondition(void (*u_init)(const mfem::Vector &,
                                           mfem::Vector &));
   
   void SetInitialCondition(const mfem::Vector uic);

   double GetEnergy(const mfem::Vector &x);

   /// compute the jacobian of the functional w.r.t the design variable
   void Mult(const mfem::Vector &x, mfem::Vector &y) const override;

   void printSolution(const mfem::Vector &c, const std::string &file_name);
   /// class destructor
   ~LinearOptimizer()
   {}
   
protected:
   nlohmann::json options;
   int dim;
   int num_state;
   int ROMSize;
   int numBasis;
   int FullSize;
   int numDesignVar;
   mfem::Vector designVar;

   // aux variables
   mfem::Array<int> outflux_bdr;
   mfem::Array<int> influx_bdr;



   std::unique_ptr<mfem::Mesh> mesh = nullptr;
   std::unique_ptr<mfem::FiniteElementCollection> fec = nullptr;
   std::unique_ptr<mfem::DGDSpace> fes_dgd = nullptr;
   std::unique_ptr<mfem::FiniteElementSpace> fes_full = nullptr;

	// the constraints
   std::unique_ptr<mfem::BilinearForm> res_dgd = nullptr;
	std::unique_ptr<mfem::BilinearForm> res_full = nullptr;

   // rhs operator
   mfem::Vector b_dgd;
   std::unique_ptr<mfem::LinearForm> b_full = nullptr;

   // some working variables
   std::unique_ptr<mfem::CentGridFunction> u_dgd = nullptr;
   std::unique_ptr<mfem::GridFunction> u_full = nullptr;

   // nonlinear and linear solver
   std::unique_ptr<mfem::NewtonSolver> newton_solver = nullptr;
   std::unique_ptr<mfem::UMFPackSolver> solver = nullptr;

   // save operators for convenient
   mfem::SparseMatrix* k_full = nullptr;
   mfem::SparseMatrix* k_dgd = nullptr;
};

} // end of namesapce
#endif