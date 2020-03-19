#include "magnetostatic.hpp"

#include <fstream>

#include "material_library.hpp"

using namespace std;
using namespace mfem;

namespace mach
{

MagnetostaticSolver::MagnetostaticSolver(
	 const std::string &opt_file_name,
    std::unique_ptr<mfem::Mesh> smesh)
	: AbstractSolver(opt_file_name, move(smesh))
{
	dim = mesh->SpaceDimension();
	num_state = dim;
	mesh->ReorientTetMesh();
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

	current_vec.reset(new GridFunType(h_curl_space.get()));

#ifdef MFEM_USE_MPI
   cout << "Number of finite element unknowns: "
        << h_curl_space->GlobalTrueVSize() << endl;
#else
   cout << "Number of finite element unknowns: "
        << h_curl_space->GetTrueVSize() << endl;
#endif

   // ifstream material_file(options["material-lib-path"].get<string>());
	// if (!material_file)
	// 	throw MachException("Could not open materials library file!");
	// material_file >> materials;
	/// using hpp file instead of json
	materials = material_library;

	/// read options file to set the proper values of static member variables
	// setStaticMembers();

	constructReluctivity();


	neg_one.reset(new ConstantCoefficient(-1.0));

	/// Construct current source coefficient
	constructCurrent();

	/// Assemble current source vector
	assembleCurrentSource();

	/// set up the spatial semi-linear form
   // double alpha = 1.0;
   res.reset(new NonlinearFormType(h_curl_space.get()));

	/// Construct reluctivity coefficient
	// constructReluctivity();

	/// TODO: Add a check in `CurlCurlNLFIntegrator` to check if |B| is close to
	///       zero, and if so set the second term of the Jacobian to be zero.
	/// add curl curl integrator to residual
	res->AddDomainIntegrator(new CurlCurlNLFIntegrator(nu.get()));

	/// TODO: magnetization lines are commented out because they created NaNs
	///       when getting gradient when B was zero because we divide by |B|.
	///       Can probably get away with adding a check if |B| is close to zero
	///       and setting magnetization contribution to the Jacobian to be zero,
	///       but need to verify that that is mathematically corrent based on
	///       the limit of the Jacobian as B goes to zero.
	/// Construct magnetization coefficient
	// constructMagnetization();

	/// add magnetization integrator to residual
	// res->AddDomainIntegrator(new MagnetizationIntegrator(nu.get(), mag_coeff.get(), -1.0));

	/// apply zero tangential boundary condition everywhere
	ess_bdr.SetSize(mesh->bdr_attributes.Max());
	ess_bdr = 1;

	Array<int> ess_tdof_list;
	h_curl_space->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
	res->SetEssentialTrueDofs(ess_tdof_list);

	Vector Zero(3);
   Zero = 0.0;
   // bc_coef.reset(new VectorConstantCoefficient(Zero)); // for motor 
	bc_coef.reset(new VectorFunctionCoefficient(3, a_exact)); // for box problem

	/// I think any of these should work
	// A->ProjectBdrCoefficientTangent(*bc_coef, ess_bdr);
   // A->ProjectBdrCoefficient(*bc_coef, ess_bdr);
   // A->ProjectCoefficient(*bc_coef);

	*A = 0.0;
	A->ProjectBdrCoefficientTangent(*bc_coef, ess_bdr);

	/// set essential boundary conditions in nonlinear form and rhs current vec
	// res->SetEssentialBC(ess_bdr, current_vec.get());
	// res->SetEssentialTrueDofs()

	/// alternative method to set current vector's ess_tdofs to zero
	current_vec->SetSubVector(ess_tdof_list, 0.0);

	/// Costruct linear system solver
#ifdef MFEM_USE_MPI
   prec.reset(new HypreAMS(h_curl_space.get()));
   prec->SetPrintLevel(0); // Don't want preconditioner to print anything
	prec->SetSingularProblem();

	solver.reset(new HypreGMRES(h_curl_space->GetComm()));
	std::cout << "set tol\n";
   solver->SetTol(options["lin-solver"]["tol"].get<double>());
	std::cout << "set tol\n";
	std::cout << "set iter\n";
   solver->SetMaxIter(options["lin-solver"]["maxiter"].get<int>());
	std::cout << "set iter\n";
	std::cout << "set print\n";
   solver->SetPrintLevel(options["lin-solver"]["printlevel"].get<int>());
	std::cout << "set print\n";
   solver->SetPreconditioner(*prec);
#else
	#ifdef MFEM_USE_SUITESPARSE
	prec = NULL;
	solver.reset(new UMFPackSolver);
	#else
	prec.reset(new GSSmoother);

	solver.reset(new CGSolver());
   solver->SetPrintLevel(options["lin-solver"]["printlevel"].get<int>());
   solver->SetMaxIter(options["lin-solver"]["maxiter"].get<int>());
   solver->SetRelTol(options["lin-solver"]["reltol"].get<double>());
   solver->SetAbsTol(options["lin-solver"]["abstol"].get<double>());
   solver->SetPreconditioner(*prec);
	#endif
#endif
	/// Set up Newton solver
	newton_solver.iterative_mode = true;
   newton_solver.SetSolver(*solver);
   newton_solver.SetOperator(*res);
   newton_solver.SetPrintLevel(options["newton"]["printlevel"].get<int>());
   newton_solver.SetRelTol(options["newton"]["reltol"].get<double>());
   newton_solver.SetAbsTol(options["newton"]["abstol"].get<double>());
   newton_solver.SetMaxIter(options["newton"]["maxiter"].get<int>());
	std::cout << "set newton solver\n";
}

void MagnetostaticSolver::solveSteady()
{
	newton_solver.Mult(*current_vec, *A);
	MFEM_VERIFY(newton_solver.GetConverged(), "Newton solver did not converge.");

	computeSecondaryFields();

	// // TODO: Print mesh out in another function?
   // ofstream sol_ofs("motor_mesh_fix2.vtk");
   // sol_ofs.precision(14);
   // mesh->PrintVTK(sol_ofs, 1);
   // A->SaveVTK(sol_ofs, "A_Field", 1);
	// B->SaveVTK(sol_ofs, "B_Field", 1);
   // sol_ofs.close();
	// std::cout << "finish steady solve\n";

	VectorFunctionCoefficient A_exact(3, a_exact);
	VectorFunctionCoefficient B_exact(3, b_exact);

	GridFunType A_ex(h_curl_space.get());
	A_ex.ProjectCoefficient(A_exact);
	
	GridFunType B_ex(h_div_space.get());
	B_ex.ProjectCoefficient(B_exact);

	GridFunType J(h_div_space.get());
	J.ProjectCoefficient(*current_coeff);

	std::cout << "A error: " << calcL2Error(A.get(), a_exact);
	std::cout << " B error: " << calcL2Error(B.get(), b_exact) << "\n";

	auto out_file = options["mesh"]["out-file"].get<std::string>();

	/// TODO: this function seems super slow
	printFields(out_file,
					{A.get(), B.get(), &A_ex, &B_ex, &J},
	            {"A_Field", "B_Field", "A_Exact", "B_exact", "current"});
}

void MagnetostaticSolver::setStaticMembers()
{
	auto material = options["components"]["magnets"]
							["material"].get<std::string>();
	remnant_flux = materials[material]["B_r"].get<double>();
	mag_mu_r = materials[material]["mu_r"].get<double>();
	fill_factor = options["components"]["windings"]["fill_factor"].get<double>();
	current_density = options["components"]["windings"]["current_density"].get<double>();
}

void MagnetostaticSolver::constructReluctivity()
{
	/// set up default reluctivity to be that of free space
	const double mu_0 = 4e-7*M_PI;
   std::unique_ptr<Coefficient> nu_free_space(
      new ConstantCoefficient(1.0/mu_0));
	nu.reset(new MeshDependentCoefficient(move(nu_free_space)));

	/// loop over all components, construct either a linear or nonlinear
	///    reluctivity coefficient for each
	for (auto& component : options["components"])
	{
		std::unique_ptr<mfem::Coefficient> temp_coeff;
		std::string material = component["material"].get<std::string>();
		if (!component["linear"].get<bool>())
		{
			auto b = materials[material]["B"].get<std::vector<double>>();
			auto h = materials[material]["H"].get<std::vector<double>>();
			temp_coeff.reset(new ReluctivityCoefficient(b, h));
		}
		else
		{
			auto mu_r = materials[material]["mu_r"].get<double>();
			temp_coeff.reset(new ConstantCoefficient(1.0/(mu_r*mu_0)));
			std::cout << "new coeff with mu_r: " << mu_r << "\n";
		}
		int attr = component.value("attr", -1);
		if (-1 != attr)
		{
			nu->addCoefficient(attr, move(temp_coeff));
		}
		else
		{
			auto attrs = component["attrs"].get<std::vector<int>>();
			for (auto& attribute : attrs)
			{
				nu->addCoefficient(attribute, move(temp_coeff));
			}
		}
	}
}

/// TODO - this approach cannot support general magnet topologies where the
///        magnetization cannot be described by a single vector function 
void MagnetostaticSolver::constructMagnetization()
{
	mag_coeff.reset(new VectorMeshDependentCoefficient(dim));

	std::unique_ptr<mfem::VectorCoefficient> magnet_coeff_north(
		new VectorFunctionCoefficient(dim, magnetization_source_north));

	std::unique_ptr<mfem::VectorCoefficient> magnet_coeff_south(
		new VectorFunctionCoefficient(dim, magnetization_source_south));

	// /// TODO: error check to make sure this worked
	// int mag_attr = options["components"]["magnets"]
	// 						["attr"].get<int>();

	auto north_attr = options["magnets"]["north"].get<std::vector<int>>();
	auto south_attr = options["magnets"]["south"].get<std::vector<int>>();
	for (auto& attr : north_attr)
	{
		mag_coeff->addCoefficient(attr, move(magnet_coeff_north));
	}
	for (auto& attr : south_attr)
	{
		mag_coeff->addCoefficient(attr, move(magnet_coeff_south));
	}
}

/// TODO - use options to select which winding belongs to which phase, need to
///        finalize mesh reading before I finish this, as the mesh will decide
///        how to do this (each winding getting its own attribute or not)
void MagnetostaticSolver::constructCurrent()
{
	current_coeff.reset(new VectorMeshDependentCoefficient());

	std::unique_ptr<mfem::VectorCoefficient> phase_a_coeff(
		new VectorFunctionCoefficient(dim, phase_a_source));

	std::unique_ptr<mfem::VectorCoefficient> phase_b_coeff(
		new VectorFunctionCoefficient(dim, phase_b_source));

	std::unique_ptr<mfem::VectorCoefficient> phase_c_coeff(
		new VectorFunctionCoefficient(dim, phase_c_source));

	auto phase_a_attr = options["phases"]["A"].get<std::vector<int>>();
	auto phase_b_attr = options["phases"]["B"].get<std::vector<int>>();
	auto phase_c_attr = options["phases"]["C"].get<std::vector<int>>();
	for (auto& attr : phase_a_attr)
	{
		current_coeff->addCoefficient(attr, move(phase_a_coeff));
	}
	for (auto& attr : phase_b_attr)
	{
		current_coeff->addCoefficient(attr, move(phase_b_coeff));
	}
	for (auto& attr : phase_c_attr)
	{
		current_coeff->addCoefficient(attr, move(phase_c_coeff));
	}
}

void MagnetostaticSolver::assembleCurrentSource()
{
	int fe_order = options["space-dis"]["degree"].get<int>();

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
	ConstantCoefficient one(1.0);
	BilinearFormIntegrator *h_curl_mass_integ = new VectorFEMassIntegrator(one);
	h_curl_mass_integ->SetIntRule(ir);
	BilinearFormType *h_curl_mass = new BilinearFormType(h_curl_space.get());
	h_curl_mass->AddDomainIntegrator(h_curl_mass_integ);

	// assemble mass matrix
	h_curl_mass->Assemble();
   h_curl_mass->Finalize();

	h_curl_mass->AddMult(j_div_free, *current_vec);
	std::cout << "below h_curl add mult\n";
	// I had strange errors when not using pointer versions of these
	delete h_curl_mass;
	delete grad;
}

void MagnetostaticSolver::computeSecondaryFields()
{
	std::cout << "before curl constructed\n";
	DiscreteCurlOperator curl(h_curl_space.get(), h_div_space.get());
	std::cout << "curl constructed\n";
	curl.Assemble();
   curl.Finalize();
	curl.Mult(*A, *B);
	std::cout << "secondary quantities computed\n";
}

/// TODO: Find a better way to handle solving the simple box problem
void MagnetostaticSolver::phase_a_source(const Vector &x,
                                         Vector &J)
{
	// example of needed geometric parameters, this should be all you need
	int n_s = 20; //number of slots
	double zb = .25; //bottom of stator
	double zt = .75; //top of stator


	// compute r and theta from x and y
	// double r = sqrt(x(0)*x(0) + x(1)*x(1)); (r not needed)
	double tha = atan2(x(1), x(0));
	double th;

	double thw = 2*M_PI/n_s; //total angle of slot
	int w; //current slot
	J = 0.0;

	// check which winding we're in
	th = remquo(tha, thw, &w);

	// check if we're in the stator body
	if(x(2) >= zb && x(2) <= zt)
	{
		// check if we're in left or right half
		if(th > 0)
		{
			J(2) = -1; // set to 1 for now, and direction depends on current direction
		}
		if(th < 0)
		{
			J(2) = 1;	
		}
	}
	else  // outside of the stator body, check if above or below
	{
		// 'subtract' z position to 0 depending on if above or below
		mfem::Vector rx(x);
		if(x(2) > zt) 
		{
			rx(2) -= zt; 
		}
		if(x(2) < zb) 
		{
			rx(2) -= zb; 
		}

		// draw top rotation axis
		mfem::Vector ax(3);
		mfem::Vector Jr(3);
		ax = 0.0;
		ax(0) = cos(w*thw);
		ax(1) = sin(w*thw);

		// take x cross ax, normalize
		Jr(0) = rx(1)*ax(2) - rx(2)*ax(1);
		Jr(1) = rx(2)*ax(0) - rx(0)*ax(2);
		Jr(2) = rx(0)*ax(1) - rx(1)*ax(0);
		Jr /= Jr.Norml2();
		J = Jr;
	}
	J *= current_density * fill_factor;
}

void MagnetostaticSolver::phase_b_source(const Vector &x,
                                         Vector &J)
{
	// example of needed geometric parameters, this should be all you need
	int n_s = 12; //number of slots
	double zb = .25; //bottom of stator
	double zt = .75; //top of stator


	// compute r and theta from x and y
	// double r = sqrt(x(0)*x(0) + x(1)*x(1)); (r not needed)
	double tha = atan2(x(1), x(0));
	double th;

	double thw = 2*M_PI/n_s; //total angle of slot
	int w; //current slot
	J = 0.0;

	// check which winding we're in
	th = remquo(tha, thw, &w);

	// check if we're in the stator body
	if(x(2) >= zb && x(2) <= zt)
	{
		// check if we're in left or right half
		if(th > 0)
		{
			J(2) = -1; // set to 1 for now, and direction depends on current direction
		}
		if(th < 0)
		{
			J(2) = 1;	
		}
	}
	else  // outside of the stator body, check if above or below
	{
		// 'subtract' z position to 0 depending on if above or below
		mfem::Vector rx(x);
		if(x(2) > zt) 
		{
			rx(2) -= zt; 
		}
		if(x(2) < zb) 
		{
			rx(2) -= zb; 
		}

		// draw top rotation axis
		mfem::Vector ax(3);
		mfem::Vector Jr(3);
		ax = 0.0;
		ax(0) = cos(w*thw);
		ax(1) = sin(w*thw);

		// take x cross ax, normalize
		Jr(0) = rx(1)*ax(2) - rx(2)*ax(1);
		Jr(1) = rx(2)*ax(0) - rx(0)*ax(2);
		Jr(2) = rx(0)*ax(1) - rx(1)*ax(0);
		Jr /= Jr.Norml2();
		J = Jr;
	}
	J *= -current_density * fill_factor;
}

void MagnetostaticSolver::phase_c_source(const Vector &x,
                                         Vector &J)
{
	J = 0.0;
}

/// TODO: Find a better way to handle solving the simple box problem
/// TODO: implement other kinds of sources
void MagnetostaticSolver::magnetization_source_north(const Vector &x,
                          		 					        Vector &M)
{
	Vector plane_vec = x;
	plane_vec(2) = 0;
	M = plane_vec;
	M /= M.Norml2();
	M *= remnant_flux;
}

void MagnetostaticSolver::magnetization_source_south(const Vector &x,
                          		 					        Vector &M)
{
	Vector plane_vec = x;
	plane_vec(2) = 0;
	M = plane_vec;
	M /= M.Norml2();
	M *= -remnant_flux;
}

/// TODO: Find a better way to handle solving the simple box problem
void MagnetostaticSolver::a_exact(const Vector &x, Vector &A)
{
   A.SetSize(3);
   A = 0.0;
   double y = x(1) - .5;
   if ( x(1) <= .5)
   {
   	A(2) = y*y*y; 
   }
   else 
   {
      A(2) = -y*y*y;
   }
}

void MagnetostaticSolver::b_exact(const Vector &x, Vector &B)
{
	B.SetSize(3);
   B = 0.0;
   double y = x(1) - .5;
   if ( x(1) <= .5)
   {
   	B(0) = 3*y*y; 
   }
   else 
   {
      B(0) = -3*y*y;
   }	
}

double MagnetostaticSolver::remnant_flux = 0.0;
double MagnetostaticSolver::mag_mu_r = 0.0;
double MagnetostaticSolver::fill_factor = 0.0;
double MagnetostaticSolver::current_density = 0.0;

} // namespace mach
