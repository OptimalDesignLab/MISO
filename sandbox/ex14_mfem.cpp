#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "gd_def.hpp"
#include "gd.hpp"
#include <chrono>
using namespace std::chrono;
using namespace std;
using namespace mfem;
double u_exact(const Vector &);
double f_exact(const Vector &);
void exact_function(const Vector &x, Vector &v);
int main(int argc, char *argv[])
{
    //  Initialize MPI.
    int num_procs, myid;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    //  Parse command-line options.
    int ref_levels = -1;
    int order = 1;
    int N = 5;
    double sigma = -1.0;
    double kappa = 50.0;
    bool visualization = 1;

    OptionsParser args(argc, argv);
    // args.AddOption(&mesh_file, "-m", "--mesh",
    //                "Mesh file to use.");
    args.AddOption(&N, "-n", "--#elements",
                   "number of mesh elements.");
    args.AddOption(&ref_levels, "-r", "--refine",
                   "Number of times to refine the mesh uniformly, -1 for auto.");
    args.AddOption(&order, "-o", "--order",
                   "Finite element order (polynomial degree) >= 0.");
    args.AddOption(&sigma, "-s", "--sigma",
                   "One of the two DG penalty parameters, typically +1/-1."
                   " See the documentation of class DGDiffusionIntegrator.");
    args.AddOption(&kappa, "-k", "--kappa",
                   "One of the two DG penalty parameters, should be positive."
                   " Negative values are replaced with (order+1)^2.");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");
    args.Parse();
    if (!args.Good())
    {
        if (myid == 0)
        {
            args.PrintUsage(cout);
        }
        MPI_Finalize();
        return 1;
    }
    if (kappa < 0)
    {
        kappa = (order + 1) * (order + 1);
    }
    if (myid == 0)
    {
        args.PrintOptions(cout);
    }

    ///  Construct mesh
    // Mesh *mesh = new Mesh(N, N, Element::TRIANGLE, true,
    //                       1, 1, true);
    Mesh *mesh = new Mesh(N, N, Element::QUADRILATERAL, true,
                          1, 1, true);
    int dim = mesh->Dimension();
    cout << "number of elements " << mesh->GetNE() << endl;
    ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
    delete mesh;

    //  Define a finite element space on the mesh. Here we use discontinuous
    //    finite elements of the specified order >= 0.
    FiniteElementCollection *fec = new DG_FECollection(order, dim);
    ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
    HYPRE_Int size = fespace->GlobalTrueVSize();
    if (myid == 0)
    {
        cout << "Number of unknowns: " << size << endl;
    }
    int num_state = 1;

    ///  GD finite element space
    ParFiniteElementSpace *fes_GD = new GalerkinDifference(pmesh, fec, num_state, Ordering::byVDIM, order);
    cout << "Number of GD unknowns: " << fes_GD->GetTrueVSize() << endl;
    cout << "#dofs " << fes_GD->GetNDofs() << endl;

    /// define different coefficients
    ConstantCoefficient one(1.0);
    ConstantCoefficient zero(0.0);
    FunctionCoefficient f(f_exact);
    FunctionCoefficient u(u_exact);
    VectorFunctionCoefficient exact(1, exact_function);

    /// GD grid function
    ParGridFunction x(fespace);
    ParCentGridFunction y(fes_GD);
    y.ProjectCoefficient(exact);

    /// Create parallel linear form 
    ParLinearForm *b = new ParLinearForm(fespace);
    b->AddDomainIntegrator(new DomainLFIntegrator(f));
    b->AddBdrFaceIntegrator(
        new DGDirichletLFIntegrator(u, one, sigma, kappa));
    b->Assemble();

    /// Create parallel Bilinear form a(.,.)
    ParBilinearForm *a = new ParBilinearForm(fespace);
    a->AddDomainIntegrator(new DiffusionIntegrator(one));
    a->AddInteriorFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));
    a->AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));
    a->Assemble();
    a->Finalize();

    HYPRE_Int glob_size = fes_GD->GetTrueVSize();
    cout << "fes_GD->GlobalVSize() " << fes_GD->GetTrueVSize()<< endl;

    /// Apply Prolongation matrix 
    // P`A P 
    SparseMatrix &Aold = a->SpMat();
    SparseMatrix *cp = dynamic_cast<GalerkinDifference *>(fes_GD)->GetCP();
    SparseMatrix *p = RAP(*cp, Aold, *cp);
    SparseMatrix &Ap = *p;
    cout << "Ap.Size() " << Ap.Size() << endl;
    y.Print();
    // P`b 
    Vector bnew(Ap.Width());
    fespace->GetProlongationMatrix()->MultTranspose(*b, bnew);

    /// Define hypre matrix and vectors
    HypreParVector *Y = y.ParallelProject();
    const int mat_size = Ap.Height();
    HYPRE_Int mat_row_idx[2] = {0, Ap.Height()};
    HypreParMatrix A(fes_GD->GetComm(), mat_size,
                               mat_row_idx, &Ap);
    HYPRE_Int glob_vsize = bnew.Size();
    int b_row_idx[2] = {0, bnew.Size()};
    HypreParVector B(fes_GD->GetComm(), glob_vsize, bnew.GetData(),
                     b_row_idx);
    
    /// Solve the system
    HypreBoomerAMG amg(A);
    cout << "problem here " << endl;
    HyprePCG pcg(A); 
    pcg.SetTol(1e-12);
    pcg.SetMaxIter(200);
    pcg.SetPrintLevel(2);
    pcg.SetPreconditioner(amg);
    pcg.Mult(B, *Y);

    delete fespace;
    delete fec;
    MPI_Finalize();
    return 0;
}
double u_exact(const Vector &x)
{
    return 2.0;
    //return exp(x(0));
    //return exp(x(0)+x(1));
    //return (x(0) * x(0) ) + (x(1) * x(1));
    //return x(0)*x(0);
    //return sin(M_PI * x(0)) * sin(M_PI * x(1));
    //return (2*x(0)) - (2*x(1));
}
double f_exact(const Vector &x)
{
    //return 0.0;
    //return -2*exp(x(0)+x(1));
    //return -exp(x(0));
    //return -4.0;
    return 2 * M_PI * M_PI * sin(M_PI * x(0)) * sin(M_PI * x(1));
}

void exact_function(const Vector &x, Vector &v)
{
    v(0) = 2.0;
    // v(0) = sin(M_PI* x(0));
    //v(0) = exp(x(0)+x(1));
    //v(0) = x(0)+ x(1);
    //v(0) = sin(M_PI * x(0)) * sin(M_PI * x(1));
}
