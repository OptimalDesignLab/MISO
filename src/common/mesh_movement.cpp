#include "mesh_movement.hpp"

#include <fstream>

using namespace std;
using namespace mfem;
#if 0
namespace mach
{

LEAnalogySolver::LEAnalogySolver(
	 const std::string &opt_file_name,
    std::unique_ptr<mfem::Mesh> smesh,
	 int dim)
	: MeshMovementSolver(opt_file_name, move(smesh))
{
    int fe_order = options["space-dis"]["degree"].get<int>();

	/// Create the H(Grad) finite element collection
    h_grad_coll.reset(new H1_FECollection(fe_order, dim));

	/// Create the H(Grad) finite element space
	h_grad_space.reset(new SpaceType(mesh.get(), h_grad_coll.get()));

    /// Create temperature grid function
	u.reset(new GridFunType(h_grad_space.get()));

    /// Set static variables
	setStaticMembers();

    //determine the list of essential boundary dofs, in this case, all of them
    // TODO: Can we make bdr_attributes contain geometric interfaces as well?
    Array<int> ess_tdof_list, ess_bdr(mesh->bdr_attributes.Max());
    ess_bdr = 1;
    h_grad_space->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

    /// zero right hand side
    bs.reset(new LinearForm(h_grad_space.get()));
    bs->Set(0, *bs);
    
    /// assign stiffness
    if(options["uniform-stiff"]["on"].template get<bool>())
    {
        lambda = options["uniform-stiff"]["lambda"].template get<double>();
        mu = options["uniform-stiff"]["mu"].template get<double>();
        lambda_c.reset(new ConstantCoefficient(lambda));
        mu_c.reset(new ConstantCoefficient(mu));
    }
    else
    {

    }

    /// assemble stiffness matrix
    k.reset(new BilinearForm(h_grad_space.get());
    k->AddDomainIntegrator(ElasticityIntegrator(lambda_c.get(),mu_c.get()));
    k->Assemble();

    /// set surface node boundary conditions

    /// set solver
}

void LEAnalogySolver::solveSteady()
{

}

} //namespace mach

#endif