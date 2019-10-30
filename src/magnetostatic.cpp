#include "magnetostatic.hpp"
#include "coefficient.hpp"
#include "electromag_integ.hpp"

using namespace std;
using namespace mfem;

namespace mach
{

MagnetostaticSolver::MagnetostaticSolver(
	 const std::string &opt_file_name,
    std::unique_ptr<mfem::Mesh> smesh,
	 int dim)
	: AbstractSolver(opt_file_name, move(smesh))
{
	int fe_order = options["space-dis"]["degree"].get<int>();

	/// Create the H(Curl) finite element collection
   h_curl_coll.reset(new ND_FECollection(fe_order, dim));
	/// Create the H(Div) finite element collection
   h_div_coll.reset(new RT_FECollection(fe_order, dim));

	/// Create the H(Curl) finite element space
	h_curl_space.reset(new SpaceType(mesh.get(), h_curl_coll.get()));
	/// Create the H(Div) finite element space
	h_div_space.reset(new SpaceType(mesh.get(), h_div_coll.get()));

	/// Create MVP grid function
	A.reset(new GridFunType(h_curl_space.get()));
	/// Create magnetic flux grid function
	B.reset(new GridFunType(h_div_space.get()));

#ifdef MFEM_USE_MPI
   cout << "Number of finite element unknowns: "
        << h_curl_space->GlobalTrueVSize() << endl;
#else
   cout << "Number of finite element unknowns: "
        << h_curl_space->GetTrueVSize() << endl;
#endif

	/// set up the spatial semi-linear form
   double alpha = 1.0;
   res.reset(new NonlinearFormType(h_curl_space.get()));

	/// Construct reluctivity coefficient
	constructReluctivity(alpha);

	/// add curl curl integrator to residual
	res->AddDomainIntegrator(new CurlCurlNLFIntegrator(nu.get()));

	/// Construct current source coefficient on lhs
	constructCurrent(-1.0);

	/// Assemble current source vector
	assembleCurrentSource();

}

void MagnetostaticSolver::assembleCurrentSource()
{
	int fe_order = options["space-dis"]["degree"].get<int>();

	/// get space dim, theres probably a better way
	int dim = h_curl_space->GetElementTransformation(0)->GetSpaceDim();

	/// Create the H1 finite element collection and space, only used by the
	/// divergence free projectors so we define them here and then throw them
	/// away
   auto h1_coll = H1_FECollection(fe_order, dim);
	auto h1_space = SpaceType(mesh.get(), &h1_coll);

	/// get int rule (approach followed my MFEM Tesla Miniapp)
	int irOrder = h_curl_space->GetElementTransformation(0)->OrderW()
                 + 2 * fe_order;
   int geom = h_curl_space->GetFE(0)->GetGeomType();
   const IntegrationRule *ir = &IntRules.Get(geom, irOrder);

	/// compute the divergence free current source
	auto *grad = new DiscreteGradOperator(&h1_space, h_curl_space.get());

	// assemble gradient form
	grad->Assemble();
   grad->Finalize();

	auto div_free_proj = DivergenceFreeProjector(h1_space, *h_curl_space,
                                              irOrder, NULL, NULL, grad);

	GridFunType j = GridFunType(h_curl_space.get());
	j.ProjectCoefficient(*current_coeff);

	GridFunType j_div_free = GridFunType(h_curl_space.get());
	// Compute the discretely divergence-free portion of j
	div_free_proj.Mult(j, j_div_free);

	/// create current linear form vector by multiplying mass matrix by
	/// divergene free current source grid function
	BilinearFormIntegrator h_curl_mass_integ = VectorFEMassIntegrator();
	h_curl_mass_integ.SetIntRule(ir);
	BilinearFormType *h_curl_mass = new BilinearFormType(h_curl_space.get());
	h_curl_mass->AddDomainIntegrator(&h_curl_mass_integ);

	// assemble mass matrix
	h_curl_mass->Assemble();
   h_curl_mass->Finalize();

	// Compute the dual of j
	h_curl_mass->AddMult(j_div_free, *current_vec);

	// I had strange errors when not using pointer versions of these
	delete h_curl_mass;
	delete grad;
}

} // namespace mach