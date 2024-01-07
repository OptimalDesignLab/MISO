#include "linear_optimization.hpp"
#include "default_options.hpp"

#include <ctime>
#include <chrono>
using namespace std;
using namespace mfem;
using namespace mach;
using namespace std::chrono;

namespace mach
{

LinearOptimizer::LinearOptimizer(Vector init,
								 const string &opt_file_name,
								 unique_ptr<mfem::Mesh> smesh)
	: Operator(0), designVar(init)
{
	// get the option fileT
	options = default_options;
	nlohmann::json file_options;
	ifstream options_file(opt_file_name);
	options_file >> file_options;
	options.merge_patch(file_options);
	cout << setw(3) << options << endl;

	// construct mesh
	mesh = std::move(smesh);
	dim = mesh->Dimension();
	num_state = 1;
	cout << "Number of elements: " << mesh->GetNE() << '\n';

	// construct fespaces
	int dgd_degree = options["space-dis"]["DGD-degree"].get<int>();
	int extra = options["space-dis"]["extra-basis"].get<int>();
	fec.reset(new DG_FECollection(options["space-dis"]["degree"].get<int>(),dim, BasisType::GaussLobatto));
	fes_dgd.reset(new DGDSpace(mesh.get(),fec.get(),designVar,dgd_degree,extra,
							num_state,Ordering::byVDIM));
	fes_full.reset(new FiniteElementSpace(mesh.get(),fec.get(),num_state,
							 Ordering::byVDIM));

	// construct the gridfunction
	u_dgd.reset(new CentGridFunction(fes_dgd.get()));
	u_full.reset(new GridFunction(fes_full.get()));

	// variable size
	ROMSize = u_dgd->Size();
	FullSize = u_full->Size();
	numDesignVar = designVar.Size();
	numBasis = numDesignVar/dim;

	// construct the residual forms
	res_dgd.reset(new BilinearForm(fes_dgd.get()));
	res_full.reset(new BilinearForm(fes_full.get()));
	b_dgd.reset(new LinearForm(fes_dgd.get()));
	b_full.reset(new LinearForm(fes_full.get()));

	// check some intermediate info
  cout << "Num of state variables: " << num_state << '\n';
  cout << "dgd_degree is: " << dgd_degree << '\n';
  cout << "u_dgd size is " << u_dgd->Size() << '\n';
  cout << "u_full size is " << u_full->Size() << '\n';
  cout << "Full size model is: "<< fes_full->GetTrueVSize() << '\n';
  cout << "DGD model size is (should be number of basis): " << num_state * dynamic_cast<DGDSpace *>(fes_dgd.get())->GetNDofs() << '\n';
  cout << "res_full size is " << res_full->Height() << " x " << res_full->Width() << '\n';
	cout << "res_dgd size is " << res_dgd->Height() << " x " << res_dgd->Width() << '\n';
}

void LinearOptimizer::InitializeSolver(VectorFunctionCoefficient& velocity, FunctionCoefficient& inflow)
{
	// get options and boundary markers
	double alpha = options["space-dis"]["alpha"].get<double>();
	auto &bcs = options["bcs"];
	vector<int> tmp = bcs["influx"].get<vector<int>>();
  influx_bdr.SetSize(tmp.size(), 0);
  influx_bdr.Assign(tmp.data());
	tmp = bcs["outflux"].get<vector<int>>();
  outflux_bdr.SetSize(tmp.size(), 0);
  outflux_bdr.Assign(tmp.data());

	// set integrators for DGD Operators
  res_dgd->AddDomainIntegrator(new ConservativeConvectionIntegrator(velocity, alpha));
  res_dgd->AddInteriorFaceIntegrator(new DGTraceIntegrator(velocity, alpha));
  res_dgd->AddBdrFaceIntegrator(new DGTraceIntegrator(velocity, alpha), outflux_bdr);


	// set integrators for full problem operator
	res_full->AddDomainIntegrator(new ConservativeConvectionIntegrator(velocity, alpha));
	res_full->AddInteriorFaceIntegrator(new DGTraceIntegrator(velocity, alpha));
  res_full->AddBdrFaceIntegrator(new DGTraceIntegrator(velocity, alpha), outflux_bdr);

	// add rhs integrator
  b_dgd->AddBdrFaceIntegrator(new BoundaryFlowIntegrator(inflow, velocity, alpha), influx_bdr);
	b_full->AddBdrFaceIntegrator(new BoundaryFlowIntegrator(inflow, velocity, alpha), influx_bdr);

	// assemble operators
	int skip_zero = 0;
	res_full->Assemble(skip_zero);
	res_full->Finalize(skip_zero);
	res_dgd->Assemble(skip_zero);
	res_dgd->Finalize(skip_zero);

	
  //  Get operators in handy
	k_full = &res_full->SpMat();
	k_dgd = &res_dgd->SpMat();
}

} // namespace mach