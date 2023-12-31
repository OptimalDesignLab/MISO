#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

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

int main(int argc, char* argv[])
{

	// default options
	constexpr double alpha = 1.0;
	const char *options_file = "linear_advection.json";
	const char *mesh_file = "square_triangle.mesh";
	int order = 1;
	int ref_levels = 1;

  // Parse command-line options
  OptionsParser args(argc, argv);

  args.AddOption(&options_file, "-o", "--options",
                "Options file to use.");
	args.AddOption(&ref_levels, "-r", "--refine",
                "mesh refinement level.");
	args.AddOption(&order, "-d", "--degree",
                "mesh refinement level.");
  args.Parse();
  if (!args.Good())
  {
    args.PrintUsage(cout);
    return 1;
  }

	// 1. read mesh
  Mesh mesh(mesh_file, 1, 1);
  int dim = mesh.Dimension();
  for (int lev = 0; lev < ref_levels; lev++)
  {
		mesh.UniformRefinement();
  }

	// 2. DGD space
	DG_FECollection fec(order, dim, BasisType::GaussLobatto);
	FiniteElementSpace fes(&mesh, &fec);
	cout << "Number of unknowns: " << fes.GetVSize() << endl;


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
	BilinearForm k(&fes);
	//k.SetAssemblyLevel(AssemblyLevel::FULL);
  k.AddDomainIntegrator(new ConservativeConvectionIntegrator(velocity, alpha));
  k.AddInteriorFaceIntegrator(
    new DGTraceIntegrator(velocity, alpha));
	k.AddBdrFaceIntegrator(
		new DGTraceIntegrator(velocity, alpha), outflux_bdr);

  // 4. rhs
	// question: alpha or -alpha
	LinearForm b(&fes);
  b.AddBdrFaceIntegrator(
    new BoundaryFlowIntegrator(inflow1, velocity, alpha), influx1_bdr);
	// actually zero boudary influx can be omitted
	// b.AddBdrFaceIntegrator(
  //   new BoundaryFlowIntegrator(inflow2, velocity, alpha), influx2_bdr);


	// 5. Assemble operators
	int skip_zero = 0;
	k.Assemble(skip_zero);
	k.Finalize(skip_zero);
	b.Assemble();
	cout << "System assembled.\n";
	
	// 6. solution vec
	GridFunction u(&fes);
	u = 0.0;


	// 7. form the linear system
	OperatorPtr K;
  Vector B, U;
	Array<int> ess_tdof_list;
  k.FormLinearSystem(ess_tdof_list, u, b, K, U, B);
	cout << "Linear system formed.\n";
	ofstream kout("k.txt");
	SparseMatrix* k_mat = dynamic_cast<SparseMatrix*>(&*K);
	k_mat->PrintMatlab(kout);
	kout.close();
	cout << "Stiffness matrix info:";
	k_mat->PrintInfo(cout);

	ofstream bout("b.txt");
	B.Print(bout, 1);
	bout.close();

  // 8. solve the system
  UMFPackSolver umf_solver;
  umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
  umf_solver.SetOperator(*K);
  umf_solver.Mult(B, U);
  // or use GMERS
	// GSSmoother M((SparseMatrix&)(*K));
	// GMRES(*K, M, B, U, 1, 500, 10, 1e-24, 0.0);


	cout << "Solved.\n";

	// 9. recover and output
	k.RecoverFEMSolution(U, b, u);
	ofstream sol_ofs("linear_advection.vtk");
  sol_ofs.precision(14);
  mesh.PrintVTK(sol_ofs,0);
  u.SaveVTK(sol_ofs,"phi",0);
  return 0;
}