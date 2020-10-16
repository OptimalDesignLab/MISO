// set this const expression to true in order to use entropy variables for state (doesn't work for rans)
constexpr bool entvar = false;

#include<random>
#include "adept.h"

#include "mfem.hpp"
#include "rans.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mach;

std::default_random_engine gen(std::random_device{}());
std::uniform_real_distribution<double> uniform_rand(0.0,1.0);

static double pert_fs;
static double mu;
static double mach_fs;
static double aoa_fs;
static int iroll;
static int ipitch;
static double chi_fs;

static double m_offset;
static double m_coeff;
static int m_x;
static int m_y;

/// \brief Defines the random function for the jacobian check
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - conservative + SA variables stored as a 5-vector
void pert(const Vector &x, Vector& p);

/// \brief Defines the random function for the jacobian check
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - conservative variables stored as a 4-vector
void pert_ns(const Vector &x, Vector& p);

/// \brief Defines the exact solution for the rans freestream problem
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - conservative + SA variables stored as a 5-vector
void uexact(const Vector &x, Vector& u);

/// \brief Defines a perturbed solution for the rans freestream problem
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - conservative + SA variables stored as a 5-vector
void uinit_pert(const Vector &x, Vector& u);

/// \brief Defines a perturbed solution for the ns freestream problem
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - conservative variables stored as a 4-vector
void uinit_pert_ns(const Vector &x, Vector& u);

/// \brief Defines a pseudo wall-distance function
/// \param[in] x - coordinate of the point at which the distance is needed
double walldist(const Vector &x);

/// Generate a quadrilateral wall mesh, more dense near the wall
/// \param[in] num_x - number of nodes in x
/// \param[in] num_y - number of nodes in y
std::unique_ptr<Mesh> buildWalledMesh(int num_x,
                                              int num_y);

int main(int argc, char *argv[])
{
   const char *options_file = "rans_walltest_options.json";

   // Initialize MPI
   int num_procs, rank;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   ostream *out = getOutStream(rank);
  
   // Parse command-line options
   OptionsParser args(argc, argv);
   int degree = 2.0;
   int nx = 14;
   int ny = 10;
   args.AddOption(&nx, "-nx", "--numx",
                  "Number of elements in x direction");
   args.AddOption(&ny, "-ny", "--numy",
                  "Number of elements in y direction");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(*out);
      return 1;
   }

   try
   {
      // construct the mesh
      string opt_file_name(options_file);
      nlohmann::json file_options;
      std::ifstream opts(opt_file_name);
      opts >> file_options;
      pert_fs = 1.0 + file_options["init-pert"].template get<double>();
      mu = file_options["flow-param"]["mu"].template get<double>();
      mach_fs = file_options["flow-param"]["mach"].template get<double>();
      aoa_fs = file_options["flow-param"]["aoa"].template get<double>()*M_PI/180;
      iroll = file_options["flow-param"]["roll-axis"].template get<int>();
      ipitch = file_options["flow-param"]["pitch-axis"].template get<int>();
      chi_fs = file_options["flow-param"]["chi"].template get<double>();

      m_offset = file_options["mesh"]["offset"].template get<double>(); //thickness of elements on wall
      m_coeff = file_options["mesh"]["coeff"].template get<double>();
      m_x = file_options["mesh"]["num-x"].template get<int>();
      m_y = file_options["mesh"]["num-y"].template get<int>();

      string file = file_options["file-names"].template get<std::string>();
      stringstream fileinit; stringstream filefinal;
      fileinit << file << "_init";
      filefinal << file << "_final";

      std::vector<GridFunType*> u_ns(1);
      std::vector<GridFunType*> u_rans(1);

      // generate the walled mesh, and its distance function
      std::unique_ptr<Mesh> smesh = buildWalledMesh(m_x, m_y);
      
      if (file_options["try-rans"].template get<bool>())
      {
      // construct the solver and set initial conditions
      auto solver = createSolver<RANavierStokesSolver<2, entvar>>(opt_file_name,
                                                         move(smesh));
      solver->setInitialCondition(uinit_pert);
      solver->printSolution(fileinit.str(), 0);

      double res_error = solver->calcResidualNorm();
      *out << "\ninitial rans residual norm = " << res_error << endl;
      //solver->checkJacobian(pert);
      solver->solveForState();
      solver->printSolution(filefinal.str(),0);
      // get the final density error
      res_error = solver->calcResidualNorm();

      *out << "\nfinal rans residual norm = " << res_error;
      u_rans = solver->getFields();
      }

      if (file_options["try-ns"].template get<bool>())
      {
      
      // construct the solver and set initial conditions
      std::unique_ptr<Mesh> smesh2 = buildWalledMesh(m_x, m_y);
      auto solver2 = createSolver<NavierStokesSolver<2, entvar>>(opt_file_name,
                                                         move(smesh2));
      solver2->setInitialCondition(uinit_pert_ns);
      fileinit << "_ns";
      filefinal << "_ns";
      solver2->printSolution(fileinit.str(), 0);

      double res_error2 = solver2->calcResidualNorm();
      *out << "\ninitial ns residual norm = " << res_error2 << endl;
      solver2->checkJacobian(pert_ns);
      solver2->solveForState();
      solver2->printSolution(filefinal.str(),0);
      res_error2 = solver2->calcResidualNorm();

      *out << "\nfinal ns residual norm = " << res_error2;
      u_ns = solver2->getFields();
      }

      //if (file_options["compare"].template get<bool>())
      //{
      
      *out << "\n Before Reording " << endl;
      mfem::Vector u_rans_comp(u_ns[0]->Size());
      for(int i = 0; i < u_rans[0]->Size()/5; i++)
      {
         for(int j = 0; j < 4; j++)
         {
            u_rans_comp(j + i*4) = u_rans[0]->Elem(j + i*5);
         }
      }
      
      // u_rans[0]->ReorderByNodes();
      // u_ns[0]->ReorderByNodes();
      *out << "\n After Reording = " << endl;

      //u_rans[0]->SetSize(u_ns[0]->Size());

      *u_ns[0] -= u_rans_comp;

      *out << "\n ns-rans result norm = " << u_ns[0]->Norml2() << endl;
      //}

   }
   catch (MachException &exception)
   {
      exception.print_message();
   }
   catch (std::exception &exception)
   {
      cerr << exception.what() << endl;
   }

#ifdef MFEM_USE_PETSC
   MFEMFinalizePetsc();
#endif

   MPI_Finalize();
}

// perturbation function used to check the jacobian in each iteration
void pert(const Vector &x, Vector& p)
{
   p.SetSize(5);
   for (int i = 0; i < 5; i++)
   {
      p(i) = 2.0 * uniform_rand(gen) - 1.0;
   }
}

// perturbation function used to check the jacobian in each iteration
void pert_ns(const Vector &x, Vector& p)
{
   p.SetSize(4);
   for (int i = 0; i < 4; i++)
   {
      p(i) = 2.0 * uniform_rand(gen) - 1.0;
   }
}

// Exact solution; same as freestream bc
void uexact(const Vector &x, Vector& q)
{
   // q.SetSize(4);
   // Vector u(4);
   q.SetSize(5);
   Vector u(5);
   
   u = 0.0;
   u(0) = 1.0;
   u(1) = u(0)*mach_fs*cos(aoa_fs);
   u(2) = u(0)*mach_fs*sin(aoa_fs);
   u(3) = 1/(euler::gamma*euler::gami) + 0.5*mach_fs*mach_fs;
   u(4) = u(0)*chi_fs*mu;

   if (entvar == false)
   {
      q = u;
   }
   else
   {
      throw MachException("No entvar for this");
   }
}

// initial guess perturbed from exact
void uinit_pert(const Vector &x, Vector& q)
{
   // q.SetSize(4);
   // Vector u(4);
   q.SetSize(5);
   Vector u(5);
   
   u = 0.0;
   u(0) = pert_fs*1.0;
   u(1) = u(0)*mach_fs*cos(aoa_fs);
   u(2) = u(0)*mach_fs*sin(aoa_fs);
   u(3) = pert_fs*1/(euler::gamma*euler::gami) + 0.5*mach_fs*mach_fs;
   u(4) = pert_fs*chi_fs*abs(mu);

   // if(x(1) == 0.0)
   // {
   //    u(1) = 1e-10;
   // }

   q = u;
}

// initial guess perturbed from exact
void uinit_pert_ns(const Vector &x, Vector& q)
{
   // q.SetSize(4);
   // Vector u(4);
   q.SetSize(4);
   Vector u(4);
   
   u = 0.0;
   u(0) = pert_fs*1.0;
   u(1) = u(0)*mach_fs*cos(aoa_fs);
   u(2) = u(0)*mach_fs*sin(aoa_fs);
   u(3) = pert_fs*1/(euler::gamma*euler::gami) + 0.5*mach_fs*mach_fs;
   // if(x(1) == 0.0)
   // {
   //    u(1) = 1e-10;
   // }

   q = u;
}

std::unique_ptr<Mesh> buildWalledMesh(int num_x, int num_y)
{
   auto mesh_ptr = unique_ptr<Mesh>(new Mesh(num_x, num_y,
                                             Element::TRIANGLE, true /* gen. edges */,
                                             2.33333, 1.0, true));
   // strategy:
   // 1) generate a fes for Lagrange elements of desired degree
   // 2) create a Grid Function using a VectorFunctionCoefficient
   // 4) use mesh_ptr->NewNodes(nodes, true) to set the mesh nodes
   
   // Problem: fes does not own fec, which is generated in this function's scope
   // Solution: the grid function can own both the fec and fes
   H1_FECollection *fec = new H1_FECollection(1, 2 /* = dim */);
   FiniteElementSpace *fes = new FiniteElementSpace(mesh_ptr.get(), fec, 2,
                                                    Ordering::byVDIM);

   // Lambda function increases element density towards wall
   double offset = m_offset;
   double coeff = m_coeff;
   auto xy_fun = [coeff, offset, num_y](const Vector& rt, Vector &xy)
   {
      xy(0) = rt(0) - 0.33333; 
      xy(1) = rt(1);

      double c = 1.0/num_y;
      double b = log(offset)/(c - 1.0);
      double a = 1.0/exp(1.0*b);

      ///Condense mesh near wall
      if(rt(1) > 0.0 && rt(1) < 1.0)
      {
         xy(1) = coeff*a*exp(b*rt(1));
         //std::cout << xy(1) << std::endl;
      }

      ///TODO: condense mesh near wall transition as well
      //double cx1 = 2.0/
      // if(rt(0) > 0.0 && rt(0) < 2.0)
      // {
      //    xy(1) = coeff*a*exp(b*rt(0));
      // }
   };
   VectorFunctionCoefficient xy_coeff(2, xy_fun);
   GridFunction *xy = new GridFunction(fes);
   xy->MakeOwner(fec);
   xy->ProjectCoefficient(xy_coeff);

   mesh_ptr->NewNodes(*xy, true);

   // Assign extra boundary attribute
   for (int i = 0; i < mesh_ptr->GetNBE(); ++i)
   {
      Element *elem = mesh_ptr->GetBdrElement(i);

      Array<int> verts;
      elem->GetVertices(verts);

      bool before = true; //before wall
      for (int j = 0; j < 2; ++j)
      {
         auto vtx = mesh_ptr->GetVertex(verts[j]);
         if (vtx[1] == 0.0 && vtx[0] <= 0.3333334)
         {
            before = before & true;
         }
         else
         {
            before = before & false;
         }
      }
      if (before)
      {
         elem->SetAttribute(5);
      }
   }
   mesh_ptr->bdr_attributes.Append(5);
   // mesh_ptr->SetAttributes();
   // mesh_ptr->Finalize();

   cout << "Number of bdr attr " << mesh_ptr->bdr_attributes.Size() <<'\n';
   cout << "Number of elements " << mesh_ptr->GetNE() <<'\n';

   return mesh_ptr;
}
