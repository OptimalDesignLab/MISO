#ifndef MACH_GDSOLVER
#define MACH_GDSOLVER

#include "mfem.hpp"
#include "euler.hpp"
#include "galer_diff.hpp"

namespace mach
{

template <int dim>
class GDSolver : public EulerSolver<dim>
{
public:
   /// Constructor for the DG solver
   /// reconstruct the gd finite element space
   GDSolver(const std::string &opt_file_name,
            std::unique_ptr<mfem::Mesh> smesh = nullptr);
};

} // end of namesapce mach

#endif