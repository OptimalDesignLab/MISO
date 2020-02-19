#include "gdsolver.hpp"

using namespace std;
using namespace mfem;

namespace mach
{

template <int dim>
GDSolver<dim>::GDSolver(const string &opt_file_name,
                        unique_ptr<Mesh> smesh)
   : EulerSolver<dim>(opt_file_name, move(smesh))
{
   int degree = this->options["GD"]["degree"].template get<int>();
   this->fes.reset(new GalerkinDifference(degree, this->pumi_mesh, dim+2, Ordering::byVDIM));
   this->fes->BuildDGProlongation();
}

//template class GDSolver<2>;
} // end of namespace mach