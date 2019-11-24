#include "magnetostatic.hpp"

#include <fstream>

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

	neg_one.reset(new ConstantCoefficient(-1.0));

	/// Construct current source coefficient
	constructCurrent();

	/// Assemble current source vector
	assembleCurrentSource();

	/// set up the spatial semi-linear form
   // double alpha = 1.0;
   res.reset(new NonlinearFormType(h_curl_space.get()));

	/// Construct reluctivity coefficient
	constructReluctivity();

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
	constructMagnetization();

	/// add magnetization integrator to residual
	res->AddDomainIntegrator(new MagnetizationIntegrator(nu.get(), mag_coeff.get(), -1.0));

	/// apply zero tangential boundary condition everywhere
	ess_bdr.SetSize(mesh->bdr_attributes.Max());
	ess_bdr = 1;
	Array<int> ess_tdof_list;
	h_curl_space->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
	Vector Zero(3);
   Zero = 0.0;
   bc_coef.reset(new VectorConstantCoefficient(Zero));
	// bc_coef.reset(new VectorFunctionCoefficient(3, a_bc_uniform));
   A->ProjectBdrCoefficientTangent(*bc_coef, ess_bdr);
   // A->ProjectCoefficient(*bc_coef);

	/// set essential boundary conditions in nonlinear form and rhs current vec
	res->SetEssentialBC(ess_bdr, current_vec.get());

	/// Costruct linear system solver
#ifdef MFEM_USE_MPI
   // prec.reset(new HypreBoomerAMG());
   prec.reset(new HypreAMS(h_curl_space.get()));
   prec->SetPrintLevel(0); // Don't want preconditioner to print anything
	prec->SetSingularProblem();

   solver.reset(new HyprePCG(h_curl_space->GetComm()));
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
	/// Set up Newton solver
	newton_solver.iterative_mode = false;
   newton_solver.SetSolver(*solver);
   newton_solver.SetOperator(*res);
   newton_solver.SetPrintLevel(options["newton"]["print-lvl"].get<int>());
   newton_solver.SetRelTol(options["newton"]["rel-tol"].get<double>());
   newton_solver.SetAbsTol(options["newton"]["abs-tol"].get<double>());
   newton_solver.SetMaxIter(options["newton"]["max-iter"].get<int>());
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
   std::unique_ptr<Coefficient> nu_free_space(
      new ConstantCoefficient(1.0/(4e-7*M_PI)));
   
	nu.reset(new MeshDependentCoefficient(move(nu_free_space)));

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

	std::unique_ptr<mfem::Coefficient> magnet_coeff(
		// new ConstantCoefficient(1.0/(10)));
		new ConstantCoefficient(1.0/(4e-7*M_PI*1.05)));

	/// TODO - use options to select material attribute for stator body
	/// picked 2 arbitrarily for now
	nu->addCoefficient(10, move(stator_coeff));
	// nu->addCoefficient(1, move(stator_coeff));
	/// TODO - use options to select material attribute for stator body
	/// picked 2 arbitrarily for now
	nu->addCoefficient(11, move(rotor_coeff));
	// nu->addCoefficient(2, move(rotor_coeff));
	nu->addCoefficient(5, move(magnet_coeff));
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

void MagnetostaticSolver::assembleCurrentSource()
{
	int fe_order = options["space-dis"]["degree"].get<int>();

	/// Create the H1 finite element collection and space, only used by the
	/// divergence free projectors so we define them here and then throw them
	/// away
   auto h1_coll = H1_FECollection(fe_order, num_dim);
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

/// TODO: Find a better way to handle solving the simple box problem
void MagnetostaticSolver::winding_current_source(const mfem::Vector &x,
                                                 mfem::Vector &J)
{
	// /*
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
	J *= 100000.0;
	// */

   // J.SetSize(3);
   // J = 0.0;
   // if ( x(1) <= .5)
   // {
   //    J(2) = -(1/(10.0))*2;
   // }
   // if ( x(1) > .5)
   // {
   //    J(2) = (1/(10.0))*2;
   // }
}

/// TODO: Find a better way to handle solving the simple box problem
/// TODO: implement other kinds of sources
void MagnetostaticSolver::magnetization_source(const mfem::Vector &x,
                          		 					  mfem::Vector &B_r)
{
	int n_p = 12; // number of poles

	// compute theta from x and y
	double tha = atan2(x(1), x(0));

	Vector plane_vec = x;
	plane_vec(2) = 0;
	
	B_r = 0.0;

	B_r(0) = plane_vec(0)/plane_vec.Norml2();
	B_r(1) = plane_vec(1)/plane_vec.Norml2();
	B_r(2) = 0.0;

	if ((int)round(tha) % 2 == 0)
	{
		B_r *= -1.0;
	}
}

/// TODO: Find a better way to handle solving the simple box problem
void MagnetostaticSolver::a_bc_uniform(const Vector &x, Vector &a)
{
   a.SetSize(3);
   a = 0.0;
   double y = x(1) - .5;
   if ( x(1) <= .5)
   {
      a(2) = y*y; 
   }
   else 
   {
      a(2) = -y*y;
   }
   //a(2) = b_uniform_(0) * x(1);
}

} // namespace mach