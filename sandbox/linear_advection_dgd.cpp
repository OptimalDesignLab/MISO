#include "mfem.hpp"
#include "galer_diff.hpp"
#include "centgridfunc.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mach;


void velocity_function(const Vector &x, Vector &v)
{
	v(0) = 3./ sqrt(10.0);
	v(1) = 1./ sqrt(10.0);
}

double inflow1_function(const Vector &x)
{
	// 1. at y = -1
	if (fabs(x(1)+1.0) < 1e-14)
	{
			return 1.0;
	}

	// 2. at x = -1, -1 <= y <= -0.6667
	if (fabs(x(0)+1.0) < 1e-14)
	{
		if (-1.0 <= x(1) && x(1) <= -0.33333333333)	
		{
			return 1.0;
		}	
	}
	return 0.0;
}

double inflow2_function(const Vector &x)
{
	return 0.0;
}

void GetBasisCenters(mfem::Mesh& mesh, mfem::Vector& vec);

int main(int argc, char* argv[])
{

	// default options
	constexpr double alpha = 1.0;
	const char *options_file = "linear_advection.json";
	const char *mesh_file = "square_triangle.mesh";
	int order = 1;
	int ref_levels = 1;
	int extra_basis = 1;
  // Parse command-line options
  OptionsParser args(argc, argv);

  args.AddOption(&options_file, "-o", "--options",
                "Options file to use.");
	args.AddOption(&ref_levels, "-r", "--refine",
                "mesh refinement level.");
	args.AddOption(&order, "-d", "--degree",
                "mesh refinement level.");
	args.AddOption(&extra_basis, "-e", "--extra",
                "extra basis to use");

  args.Parse();
  if (!args.Good())
  {
    args.PrintUsage(cout);
    return 1;
  }

	// 1. read mesh
	// 1.a mesh
  Mesh mesh(mesh_file, 1, 1);
  int dim = mesh.Dimension();
  for (int lev = 0; lev < ref_levels; lev++)
  {
		mesh.UniformRefinement();
  }
	// 1.b element centers
	int numElement = mesh.GetNE();
	mfem::Vector center(dim * numElement);
	GetBasisCenters(mesh, center);

	// 2. Finite element space and DGD space
	DG_FECollection fec(order, dim, BasisType::GaussLobatto);
	DGDSpace dgd_fes(&mesh, &fec, center, order, extra_basis, 1, Ordering::byVDIM);
	FiniteElementSpace fes(&mesh, &fec);
	auto* p = dgd_fes.GetCP();
	cout << "Num of state variables: " << 1 << '\n';
	cout << "Finite element collection degree = " << order << '\n';
	cout << "Number of unknowns: " << fes.GetVSize() << '\n';


  // 3. problem coefficient
	VectorFunctionCoefficient velocity(dim, velocity_function);
	FunctionCoefficient inflow1(inflow1_function);
	FunctionCoefficient inflow2(inflow2_function);

	Array<int> influx1_bdr(mesh.bdr_attributes.Max());
	Array<int> influx2_bdr(mesh.bdr_attributes.Max());
	Array<int> outflux_bdr(mesh.bdr_attributes.Max());
	cout << "Boundary attribute size is " << mesh.bdr_attributes.Max() << '\n';
	influx1_bdr = 0; influx1_bdr[1] = 1;
	influx2_bdr = 0; influx2_bdr[3] = 1;
	outflux_bdr = 0; outflux_bdr[2] = 1;
	

	// 3. domain operators
	BilinearForm k(&dgd_fes);
	//k.SetAssemblyLevel(AssemblyLevel::FULL);
  k.AddDomainIntegrator(new ConservativeConvectionIntegrator(velocity, alpha));
  k.AddInteriorFaceIntegrator(
    new DGTraceIntegrator(velocity, alpha));
	k.AddBdrFaceIntegrator(
		new DGTraceIntegrator(velocity, alpha), outflux_bdr);

  // 4. rhs
	// question: alpha or -alpha
	LinearForm b(&dgd_fes);
  b.AddBdrFaceIntegrator(
    new BoundaryFlowIntegrator(inflow1, velocity, alpha), influx1_bdr);
	// actually zero boudary influx can be omitted
	// b.AddBdrFaceIntegrator(
  //   new BoundaryFlowIntegrator(inflow2, velocity, alpha), influx2_bdr);


	// 5. Assemble the original operators
	int skip_zero = 0;
	k.Assemble(skip_zero);
	k.Finalize(skip_zero);
	b.Assemble();
	cout << "Origin System assembled.\n";

	// 6. Construct the DGD space operators
	SparseMatrix& kref = k.SpMat();
	SparseMatrix* kdgd = RAP(*p, kref, *p);
	Vector bdgd(dgd_fes.GetTrueVSize());
	p->MultTranspose(b, bdgd);
	// 6. a write operators to check
	ofstream kcout("k_dgd.txt");
	kdgd->PrintMatlab(kcout);
	kcout.close();
	ofstream bcout("b_dgd.txt");
	bdgd.Print(bcout,4);
	bcout.close();

	
	// 7. solution vec
	GridFunction u(&fes);
	mfem::CentGridFunction uc(&dgd_fes);
	uc = 0.0;
	u = 0.0;


	// 7. form the linear system
	cout << "\n\nStiffness matrix info:";
	kdgd->PrintInfo(cout);

  // 8. solve the system
  UMFPackSolver umf_solver;
  umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
  umf_solver.SetOperator(*kdgd);
  umf_solver.Mult(bdgd, uc);
  // or use GMERS
	// GSSmoother M((SparseMatrix&)(*K));
	// GMRES(*K, M, B, U, 1, 500, 10, 1e-24, 0.0);
	cout << "Solved.\n";



	// 9. recover and output
	ofstream pout("prolong.txt");
	dgd_fes.GetProlongationMatrix()->PrintMatlab(pout);
	pout.close();
	dgd_fes.GetProlongationMatrix()->Mult(uc, u);
	ofstream sol_ofs("linear_advection_dgd.vtk");
  sol_ofs.precision(14);
  mesh.PrintVTK(sol_ofs,0);
  u.SaveVTK(sol_ofs,"phi",0);
	delete kdgd;
  return 0;
}


void GetBasisCenters(mfem::Mesh& mesh, mfem::Vector& vec)
{
	int ne = mesh.GetNE();
	int dim = mesh.Dimension();
	Vector loc(dim);
	for (int i = 0; i < ne; ++i)
	{
		mesh.GetElementCenter(i, loc);
		vec(dim * i) = loc(0);
		vec(dim* i+1) = loc(1);
	}
}