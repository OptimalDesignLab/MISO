/// Solve the steady isentropic vortex problem on a quarter annulus

// set this const expression to true in order to use entropy variables for state
constexpr bool entvar = false;

#include <fstream>
#include <iostream>
#include <random>
#include "dgdsolver.hpp"

using namespace std;
using namespace mfem;
using namespace mach;

std::default_random_engine gen(std::random_device{}());
std::uniform_real_distribution<double> normal_rand(-1.0, 1.0);
static std::uniform_real_distribution<double> uniform_rand(0.0, 1.0);
const double rho = 0.9856566615165173;
const double rhoe = 2.061597236955558;
const double rhou[3] = {0.09595562550099601, -0.030658751626551423, -0.13471469906596886};

/// \brief Defines the random function for the jabocian check
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - conservative variables stored as a 4-vector
void pert(const Vector &x, Vector &p);

/// \brief Returns the value of the integrated math entropy over the domain
double calcEntropyTotalExact();

/// \brief Defines the exact solution for the steady isentropic vortex
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - state variables stored as a 4-vector
void uexact(const Vector &x, Vector &u);

/// Generate quarter annulus mesh
/// \param[in] degree - polynomial degree of the mapping
/// \param[in] num_rad - number of nodes in the radial direction
/// \param[in] num_ang - number of nodes in the angular direction
std::unique_ptr<Mesh> buildQuarterAnnulusMesh(int degree, int num_rad,
                                              int num_ang);

// main
int main(int argc, char *argv[])
{
    const char *options_file = "vortex_GD_options.json";
    // 1. Initialize MPI.
    int num_procs, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ostream *out = getOutStream(rank);
    // Parse command-line options
    OptionsParser args(argc, argv);
    int degree = 1;
    int nx = 5;
    int ny = 5;
    int order = 1;
    int ref_levels = -1;
    int nc_ref = -1;
    args.AddOption(&degree, "-d", "--degree", "poly. degree of mesh mapping");
    args.AddOption(&order, "-o", "--order",
                   "Finite element order (polynomial degree) >= 0.");
    args.AddOption(&nx, "-nr", "--num-rad", "number of radial segments");
    args.AddOption(&ny, "-nt", "--num-theta", "number of angular segments");
    args.AddOption(&ref_levels, "-ref", "--refine",
                   "refine levels");

    args.Parse();
    if (!args.Good())
    {
        if (rank == 0)
        {
            args.PrintUsage(cout);
        }
        MPI_Finalize();
        return 1;
    }
    if (rank == 0)
    {
        args.PrintOptions(cout);
    }

    /// degree = p+1
    degree = order + 1;
    /// number of state variables
    int num_state = 4;
    try
    {
        // construct the solver, set the initial condition, and solve
        string opt_file_name(options_file);
        /// construct the mesh
        unique_ptr<Mesh> mesh = buildQuarterAnnulusMesh(degree, nx, ny);
        cout << "Number of elements " << mesh->GetNE() << '\n';

        /// dimension
        const int dim = mesh->Dimension();

        for (int l = 0; l < ref_levels; l++)
        {
            mesh->UniformRefinement();
        }

        cout << "Number of elements after refinement " << mesh->GetNE() << '\n';
        // out->precision(15);
        // construct the solver and set initial conditions
        auto solver = createSolver<DGDSolver<2, entvar>>(opt_file_name,
                                                         move(mesh));
        solver->setGDInitialCondition(uexact);

        /// get the initial density error
        double l2_error = (static_cast<DGDSolver<2, entvar> &>(*solver)
                               .calcConservativeVarsL2Error(uexact, 0));
        double res_error = solver->calcResidualNorm();
        *out << "\n|| rho_h - rho ||_{L^2} = " << l2_error << endl;
        *out << "\ninitial residual norm = " << res_error << endl;
        solver->checkJacobian(pert);
        solver->solveForState();
        res_error = solver->calcResidualNorm();
        double drag = solver->calcOutput("drag");
        double drag_err = abs(drag - (-1 / mach::euler::gamma));
        l2_error = (static_cast<DGDSolver<2, entvar> &>(*solver)
                        .calcConservativeVarsL2Error(uexact, 0));
        cout << "======================================================= " << endl;
        *out << "|| rho_h - rho ||_{L^2} = " << l2_error << endl;
        *out << "Drag is = " << drag << endl;
        *out << "Drag error = " << drag_err << endl;
        cout << "======================================================= " << endl;
    }
    catch (MachException &exception)
    {
        exception.print_message();
    }
    catch (std::exception &exception)
    {
        cerr << exception.what() << endl;
    }
    MPI_Finalize();
    return 0;
}

// perturbation function used to check the jacobian in each iteration
void pert(const Vector &x, Vector &p)
{
    p.SetSize(4);
    for (int i = 0; i < 4; i++)
    {
        p(i) = normal_rand(gen);
    }
}

// Returns the exact total entropy value over the quarter annulus
// Note: the number 8.74655... that appears below is the integral of r*rho over the radii
// from 1 to 3.  It was approixmated using a degree 51 Gaussian quadrature.
double calcEntropyTotalExact()
{
    double rhoi = 2.0;
    double prsi = 1.0 / euler::gamma;
    double si = log(prsi / pow(rhoi, euler::gamma));
    return -si * 8.746553803443305 * M_PI * 0.5 / 0.4;
}

// Exact solution; note that I reversed the flow direction to be clockwise, so
// the problem and mesh are consistent with the LPS paper (that is, because the
// triangles are subdivided from the quads using the opposite diagonal)
void uexact(const Vector &x, Vector &q)
{
    q.SetSize(4);
    Vector u(4);
    double ri = 1.0;
    double Mai = 0.5; //0.95
    double rhoi = 2.0;
    double prsi = 1.0 / euler::gamma;
    double rinv = ri / sqrt(x(0) * x(0) + x(1) * x(1));
    double rho = rhoi * pow(1.0 + 0.5 * euler::gami * Mai * Mai * (1.0 - rinv * rinv),
                            1.0 / euler::gami);
    double Ma = sqrt((2.0 / euler::gami) * ((pow(rhoi / rho, euler::gami)) *
                                                (1.0 + 0.5 * euler::gami * Mai * Mai) -
                                            1.0));
    double theta;
    if (x(0) > 1e-15)
    {
        theta = atan(x(1) / x(0));
    }
    else
    {
        theta = M_PI / 2.0;
    }
    double press = prsi * pow((1.0 + 0.5 * euler::gami * Mai * Mai) /
                                  (1.0 + 0.5 * euler::gami * Ma * Ma),
                              euler::gamma / euler::gami);
    double a = sqrt(euler::gamma * press / rho);

    u(0) = rho;
    u(1) = -rho * a * Ma * sin(theta);
    u(2) = rho * a * Ma * cos(theta);
    u(3) = press / euler::gami + 0.5 * rho * a * a * Ma * Ma;

    q = u;
    // double mach_fs = 0.3;
    // q(0) = 1.0;
    // q(1) = q(0) * mach_fs; // ignore angle of attack
    // q(2) = 0.0;
    // q(3) = 1 / (euler::gamma * euler::gami) + 0.5 * mach_fs * mach_fs;
}

unique_ptr<Mesh> buildQuarterAnnulusMesh(int degree, int num_rad, int num_ang)
{
    auto mesh_ptr = unique_ptr<Mesh>(new Mesh(num_rad, num_ang,
                                              Element::QUADRILATERAL, true /* gen. edges */,
                                              2.0, M_PI * 0.5, true));
    // strategy:
    // 1) generate a fes for Lagrange elements of desired degree
    // 2) create a Grid Function using a VectorFunctionCoefficient
    // 4) use mesh_ptr->NewNodes(nodes, true) to set the mesh nodes

    // Problem: fes does not own fec, which is generated in this function's scope
    // Solution: the grid function can own both the fec and fes
    H1_FECollection *fec = new H1_FECollection(degree, 2 /* = dim */);
    FiniteElementSpace *fes = new FiniteElementSpace(mesh_ptr.get(), fec, 2,
                                                     Ordering::byVDIM);

    // This lambda function transforms from (r,\theta) space to (x,y) space
    auto xy_fun = [](const Vector &rt, Vector &xy) {
        xy(0) = (rt(0) + 1.0) * cos(rt(1)); // need + 1.0 to shift r away from origin
        xy(1) = (rt(0) + 1.0) * sin(rt(1));
    };
    VectorFunctionCoefficient xy_coeff(2, xy_fun);
    GridFunction *xy = new GridFunction(fes);
    xy->MakeOwner(fec);
    xy->ProjectCoefficient(xy_coeff);

    mesh_ptr->NewNodes(*xy, true);
    return mesh_ptr;
}