#include "magnetostatic.hpp"

#include <fstream>

using namespace std;
using namespace mfem;
//just copying magnetostatic.cpp for now
namespace mach
{

ThermalSolver::ThermalSolver(
	 const std::string &opt_file_name,
    std::unique_ptr<mfem::Mesh> smesh,
	 int dim)
	: AbstractSolver(opt_file_name, move(smesh))
{
	mesh->ReorientTetMesh();
	int fe_order = options["space-dis"]["degree"].get<int>();

	/// Create the H(Grad) finite element collection
    h_grad_coll.reset(new H1_FECollection(fe_order, dim));

	/// Create the H(Grad) finite element space
	h_grad_space.reset(new SpaceType(mesh.get(), h_grad_coll.get()));

	/// Create MVP grid function
	T.reset(new GridFunType(h_grad_space.get()));

	current_vec.reset(new GridFunType(h_curl_space.get()));

#ifdef MFEM_USE_MPI
   cout << "Number of finite element unknowns: "
        << h_grad_space->GlobalTrueVSize() << endl;
#else
   cout << "Number of finite element unknowns: "
        << h_grad_space->GetTrueVSize() << endl;
#endif

    ifstream material_file(options["material-lib-path"].get<string>());
	/// TODO: replace with mach exception
	if (!material_file)
		std::cerr << "Could not open materials library file!" << std::endl;
	material_file >> materials;

	///TODO: Define these based on materials library
	constructMassCoeff();

    constructConductivity();
     
    constructElecConductivity();

	/// set up the bilinear form
    a.reset(new BilinearFormType(h_curl_space.get()));

	/// add mass integrator to bilinear form
	a->AddDomainIntegrator(new MassIntegrator(rho_cv.get()));

	/// add diffusion integrator to bilinear form
	a->AddDomainIntegrator(new DiffusionIntegrator(kappa.get()));

	/// add boundary integrator to bilinear form for flux BC, elsewhere is natural
	///TODO: Define Boundary Conditions, Make More General, Selectively Apply To Certain Faces
	a->AddBdrFaceIntegrator(new OutwardHeatFluxBC());

	/// set up the linear form (volumetric fluxes)
	a.reset(new LinearFormType(h_curl_space.get()));

	/// add joule heating term
	a->AddDomainIntegrator(new JouleIntegrator(sigma.get()));

	/// add iron loss heating terms
	a->AddDomainIntegrator(new IronLossIntegrator(rho_cv.get()));


	/// Construct linear system solver
	///TODO: Think about preconditioner
#ifdef MFEM_USE_MPI
   // prec.reset(new HypreBoomerAMG());
   prec.reset(new HypreAMS(h_curl_space.get()));
   prec->SetPrintLevel(0); // Don't want preconditioner to print anything
	prec->SetSingularProblem();

   // solver.reset(new HyprePCG(h_curl_space->GetComm()));
	solver.reset(new HypreGMRES(h_curl_space->GetComm()));
	std::cout << "set tol\n";
   solver->SetTol(options["lin-solver"]["tol"].get<double>());
	std::cout << "set tol\n";
	std::cout << "set iter\n";
   solver->SetMaxIter(options["lin-solver"]["max-iter"].get<int>());
	std::cout << "set iter\n";
	std::cout << "set print\n";
   solver->SetPrintLevel(options["lin-solver"]["print-lvl"].get<int>());
	std::cout << "set print\n";
   solver->SetPreconditioner(*prec);
#else
	#ifdef MFEM_USE_SUITESPARSE
	prec = NULL;
	solver.reset(new UMFPackSolver);
	#else
	prec.reset(new GSSmoother);

	solver.reset(new CGSolver());
    solver->SetPrintLevel(options["lin-solver"]["print-lvl"].get<int>());
    solver->SetMaxIter(options["lin-solver"]["max-iter"].get<int>());
    solver->SetRelTol(options["lin-solver"]["rel-tol"].get<double>());
    solver->SetAbsTol(options["lin-solver"]["abs-tol"].get<double>());
    solver->SetPreconditioner(*prec);
	#endif
#endif
}

void MagnetostaticSolver::solveSteady()
{
	newton_solver.Mult(*current_vec, *A);
	MFEM_VERIFY(newton_solver.GetConverged(), "Newton solver did not converge.");

	/// TODO: Move this to a new function `ComputeSecondaryQuantities()`?
	std::cout << "before curl constructed\n";
	DiscreteCurlOperator curl(h_curl_space.get(), h_div_space.get());
	std::cout << "curl constructed\n";
	curl.Assemble();
   curl.Finalize();
	curl.Mult(*A, *B);
	std::cout << "curl taken\n";

	// TODO: Print mesh out in another function?
   ofstream sol_ofs("motor_mesh_fix2.vtk");
   sol_ofs.precision(14);
   mesh->PrintVTK(sol_ofs, 1);
   A->SaveVTK(sol_ofs, "A_Field", 1);
	B->SaveVTK(sol_ofs, "B_Field", 1);
   sol_ofs.close();
	std::cout << "finish steady solve\n";
}

void MagnetostaticSolver::constructReluctivity()
{
	/// set up default reluctivity to be that of free space
	double mu_0 = 4e-7*M_PI;
   std::unique_ptr<Coefficient> nu_free_space(
      new ConstantCoefficient(1.0/mu_0));
	nu.reset(new MeshDependentCoefficient(move(nu_free_space)));

	for (auto& material : options["materials"])
	{
		std::unique_ptr<mfem::Coefficient> temp_coeff;
		std::cout << material << '\n';
		if (!material["linear"].get<bool>())
		{
			auto b = materials["steel"]["B"].get<std::vector<double>>();
			auto h = materials["steel"]["H"].get<std::vector<double>>();
			temp_coeff.reset(new ReluctivityCoefficient(b, h));
		}
		else
		{
			auto mu_r = material["mu_r"].get<double>();
			// std::unique_ptr<mfem::Coefficient> temp_coeff(
			temp_coeff.reset(new ConstantCoefficient(1.0/(mu_r*mu_0)));
		}
		nu->addCoefficient(material["attr"].get<int>(), move(temp_coeff));
	}
		
	

	/// uncomment eventually, for now we use constant linear model
	// std::unique_ptr<mfem::Coefficient> stator_coeff(
	// 	new ReluctivityCoefficient(reluctivity_model));

	/// create constant coefficient for stator body with relative permeability
	/// 3000
	std::unique_ptr<mfem::Coefficient> stator_coeff(
		// new ConstantCoefficient(1.0/(10)));
		new ConstantCoefficient(1.0/(5000*4e-7*M_PI)));
	
	/// create constant coefficient for rotor body with relative permeability
	/// 3000
	std::unique_ptr<mfem::Coefficient> rotor_coeff(
		// new ConstantCoefficient(1.0/(10)));
		new ConstantCoefficient(1.0/(5000*4e-7*M_PI)));

	// std::unique_ptr<mfem::Coefficient> magnet_coeff(
	// 	// new ConstantCoefficient(1.0/(10)));
	// 	new ConstantCoefficient(1.0/(4e-7*M_PI)));

	/// TODO - use options to select material attribute for stator body
	/// picked 2 arbitrarily for now
	nu->addCoefficient(10, move(stator_coeff));
	// nu->addCoefficient(1, move(stator_coeff));
	/// TODO - use options to select material attribute for stator body
	/// picked 2 arbitrarily for now
	nu->addCoefficient(11, move(rotor_coeff));
	// nu->addCoefficient(2, move(rotor_coeff));

	// nu->addCoefficient(5, move(magnet_coeff));
}

void MagnetostaticSolver::constructMagnetization()
{
	mag_coeff.reset(new VectorMeshDependentCoefficient(num_dim));

	std::unique_ptr<mfem::VectorCoefficient> magnet_coeff(
		new VectorFunctionCoefficient(num_dim, magnetization_source));

	/// TODO - use options to select material attribute for magnets
	/// picked 4 arbitrarily for now
	mag_coeff->addCoefficient(5, move(magnet_coeff));
}

void MagnetostaticSolver::constructCurrent()
{
	current_coeff.reset(new VectorMeshDependentCoefficient());

	std::unique_ptr<mfem::VectorCoefficient> winding_coeff(
		new VectorFunctionCoefficient(num_dim, winding_current_source));


	std::unique_ptr<mfem::VectorCoefficient> winding_coeff2(
		new VectorFunctionCoefficient(num_dim, winding_current_source, neg_one.get()));

	/// TODO - use options to select material attribute for windings
	/// picked 1 arbitrarily for now
	current_coeff->addCoefficient(7, move(winding_coeff));
	current_coeff->addCoefficient(8, move(winding_coeff2));
	// current_coeff->addCoefficient(9, move(winding_coeff)); // zero current
	// current_coeff->addCoefficient(1, move(winding_coeff));
	// current_coeff->addCoefficient(2, move(winding_coeff2));
}

} // namespace mach