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
   
   virtual double GetEnergy(const mfem::Vector &x) const;

   /// compute the jacobian of the functional w.r.t the design variable
   virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const;

   void addVolumeIntegrators(double alpha);
   void addBoundaryIntegrators(double alpha);

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

   // aux variables
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
};

} // end of namesapce
#endif