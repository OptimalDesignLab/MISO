#include "gdsolver.hpp"

#ifdef MFEM_USE_PUMI
#ifdef MFEM_USE_MPI

using namespace std;
using namespace mfem;

namespace mach
{

template <int dim>
GDSolver<dim>::GDSolver(const string &opt_file_name,
                        unique_ptr<Mesh> smesh)
   : EulerSolver<dim>(opt_file_name, move(smesh))
{
   int gd_degree = this->options["GD"]["degree"].template get<int>();
   // this->fes.reset(new GalerkinDifference(mesh.get(), fec.get(), num_state,
   //                                  Ordering::byVDIM, gd_degree, pumi_mesh));
}

//template class GDSolver<2>;
} // end of namespace mach

#endif //MFEM_USE_MPI
#endif //MFEM_USE_PUMI