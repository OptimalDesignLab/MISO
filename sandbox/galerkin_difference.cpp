/// Solve the steady isentropic vortex problem on a quarter annulus
#include "mfem.hpp"
#include "euler.hpp"
#include "galer_diff.hpp"
#include "centgridfunc.hpp"

#include <fstream>
#include <iostream>
#include <random>
const bool entvar = true;
// #ifdef MFEM_USE_SIMMETRIX
// #include <SimUtil.h>
// #include <gmi_sim.h>
// #endif
// #include <apfMDS.h>
// #include <gmi_null.h>
// #include <PCU.h>
// #include <apfConvert.h>
// #include <gmi_mesh.h>
// #include <crv.h>

using namespace std;
using namespace mfem;
using namespace mach;

/// \brief multiply exact solutions for testing
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - conservative variables stored as a 4-vector
void u_exact(const mfem::Vector &x, mfem::Vector &u);
void conserv_exact(const mfem::Vector &x, mfem::Vector &u);
// void uexact_1d(const mfem::Vector &x, mfem::Vector &u);
// void uexact_single(const mfem::Vector &x, mfem::Vector &u);
// void upoly(const mfem::Vector &x, mfem::Vector &u);

/// Generate quarter annulus mesh
/// \param[in] degree - polynomial degree of the mapping
/// \param[in] num_rad - number of nodes in the radial direction
/// \param[in] num_ang - number of nodes in the angular direction
std::unique_ptr<Mesh> buildQuarterAnnulusMesh(int degree, int num_rad,
                                              int num_ang);

// This function will be used to check the local R and the assembled prolongation matrix
int main(int argc, char *argv[])
{
   //Parse command-line options
   OptionsParser args(argc, argv);
   int p = 1;
   int degree = p+1; // annulus mesh degree, should be p+1
   int nx = 10;
   int ny = 10;
   int rf = 0;
   args.AddOption(&nx, "-nr", "--num-rad", "number of radial segments");
   args.AddOption(&ny, "-nt", "--num-thetat", "number of angular segments");
   args.AddOption(&p, "-p", "--order", "projection operator degree");
   args.AddOption(&rf, "-r", "--refine", "level of refinement");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }

   try
   {
      int dim;
      int num_state = 3;
   //================== Multiply mesh files for testing ====================
      // unique_ptr<Mesh> mesh = buildQuarterAnnulusMesh(degree, nx, ny);
      // unique_ptr<Mesh> mesh(new Mesh(nx));
      // ofstream sol_ofs("gd_test.vtk");
      // sol_ofs.precision(14);
      // smesh->PrintVTK(sol_ofs, 0);
      // sol_ofs.close();
      mfem::Mesh *mesh = new Mesh(nx);
      //mfem::Mesh *mesh = new Mesh("periodic_segment.mesh", 1, 1);
      // for (int l = 0; l < rf; l++)
      // {
      //    mesh->UniformRefinement();
      // }
      std::cout << "Number of elements " << mesh->GetNE() << '\n';
      dim = mesh->Dimension();

   //================== Construct the gd and normal finite element spaces =========
      cout << "Construct the GD fespace.\n";
      DSBPCollection fec(p, dim);
      GalerkinDifference gd(mesh, &fec, num_state, Ordering::byVDIM, p);
      FiniteElementSpace fes(mesh, &fec, num_state, Ordering::byVDIM);

   //================== Construct the gridfunction and apply the exact solution ========
      mfem::CentGridFunction x_cent(&gd);
      mfem::GridFunction x(&fes);
      mfem::GridFunction x_exact(&fes);
      mfem::GridFunction conserv_solution(&fes);

      mfem::VectorFunctionCoefficient u0_fun(num_state, u_exact);
      mfem::VectorFunctionCoefficient conserv_fun(num_state, conserv_exact);
      x_cent.ProjectCoefficient(u0_fun);
      x_exact.ProjectCoefficient(u0_fun);
      conserv_solution.ProjectCoefficient(conserv_fun);

   //============== Prolong the solution to SBP nodes =================================
   //============== and check the simple l2 norm error ================================
      gd.GetProlongationMatrix()->Mult(x_cent, x);
      x -= x_exact;
      // cout << "\nCheck the nodal error:\n";
      // x.Print(cout, num_state);
      cout << "Check the projection l2 error: " << x.Norml2() << '\n';

   //============= check the prolonged entropy error in terms of the conservative variables
      gd.GetProlongationMatrix()->Mult(x_cent, x);
      int num_nodes, offset;
      const mfem::FiniteElement *fe;
      mfem::Vector w(num_state), q(num_state);
      mfem::GridFunction conserv_converted(&fes);
      Array<int> vdofs(num_state);
      ofstream write_exact("exact_solution.txt");
      ofstream write_prolong("prolong_solution.txt");
      for (int i = 0; i < gd.GetNE(); i++)
      {
         fe = gd.GetFE(i);
         num_nodes = fe->GetDof();
         for (int j = 0; j < num_nodes; j++)
         {
            offset = i * num_nodes * num_state + j * num_state;
            for (int k = 0; k < num_state; k++)
            {
               vdofs[k] = offset + k;
            }
            x.GetSubVector(vdofs, w);
            if (entvar)
            {
               calcConservativeVars<double, 1>(w, q);
            }
            else
            {
               q = w;
            }
            conserv_converted.SetSubVector(vdofs, q);
            for (int k = 0; k < num_state; k++)
            {
               write_exact << conserv_solution(vdofs[k]) << ' ';
               write_prolong << conserv_converted(vdofs[k]) << ' ';
            }
            write_exact << endl;
            write_prolong << endl;
         }
      }
      write_exact.close();
      write_prolong.close();
      conserv_converted -= conserv_solution;
      //conserv_converted.Print(cout, num_state);
      cout << "Check the converted state l2 error: " << conserv_converted.Norml2() << '\n';

      //============== a funny test ===========
      mfem::Vector q_1(3), w_1(3), q_2(3);
      q_1(0) = 0.02;
      q_1(1) = 0.1 * q_1(0);
      q_1(2) = 20.0/0.4 + 0.5 * q_1(0) * 0.1 * 0.1;
      calcEntropyVars<double, 1>(q_1, w_1);
      calcConservativeVars<double, 1>(w_1, q_2);
      cout << "original state: "<< q_1(0) << ' ' << q_1(1) << ' ' << q_1(2) << endl;
      cout << "entropy variable: "<< w_1(0) << ' ' << w_1(1) << ' ' << w_1(2) << endl;
      cout << "converted back: "<< q_2(0) << ' ' << q_2(1) << ' ' <<q_2(2) << endl;

      GalerkinDifference gd2(mesh, &fec, 1, Ordering::byVDIM, p);
      
      // std::cout << "Check P transpose: ";
      // mfem::GridFunction y(&gd);
      // mfem::CentGridFunction y_cent(&gd);

      // mfem::VectorFunctionCoefficient u1(1, uexact_1d);
      // y.ProjectCoefficient(u1);
      // y_cent.ProjectCoefficient(u1);

      // mfem::GridFunction y_project(&gd);
      // gd.GetProlongationMatrix()->Mult(y_cent, y_project);
      
      // const FiniteElement *fe;
      // ElementTransformation *T;
      // double h;
      // int num_pt;

      // // sum up the L2 error over all states
      // for (int i = 0; i < gd.GetNE(); i++)
      // {
      //    fe = gd.GetFE(i);
      //    const IntegrationRule *ir = &(fe->GetNodes());
      //    T = gd.GetElementTransformation(i);
      //    num_pt = ir->GetNPoints();
      //    for (int j = 0; j < num_pt; j++)
      //    {
      //       const IntegrationPoint &ip = ir->IntPoint(j);
      //       T->SetIntPoint(&ip);
      //       h = ip.weight * T->Weight();
      //       y_project(i*num_pt+j) = h * y_project(i*num_pt+j);
      //    }
      // }
      // double lhs = x_exact * y_project;

      // mfem::CentGridFunction x_project(&gd);
      // for (int i = 0; i < gd.GetNE(); i++)
      // {
      //    fe = gd.GetFE(i);
      //    const IntegrationRule *ir = &(fe->GetNodes());
      //    T = gd.GetElementTransformation(i);
      //    num_pt = ir->GetNPoints();
      //    for (int j = 0; j < num_pt; j++)
      //    {
      //       const IntegrationPoint &ip = ir->IntPoint(j);
      //       T->SetIntPoint(&ip);
      //       h = ip.weight * T->Weight();
      //       x_exact(i*num_pt+j) = h * x_exact(i*num_pt+j);
      //    }
      // }
      // gd.GetProlongationMatrix()->MultTranspose(x_exact, x_project);
      // double rhs = y_cent * x_project;
      // std::cout << "x^t * P * yc - yc^t * P^t * x^t = " << lhs - rhs << '\n';
      // std::cout << "lhs = " << lhs << ", rhs = " << rhs << '\n';
      // Test the prolongation matrix with gridfunction vdim = 1
      // mfem::GridFunction x(&gd);
      // mfem::GridFunction x_exact(&gd);
      // cout << "Size of x and x_exact is " << x.Size() << '\n';

      // mfem::VectorFunctionCoefficient u0(1, uexact_single);
      // x_exact.ProjectCoefficient(u0);
      // cout << "Check the exact solution:\n";
      // x_exact.Print(cout ,7);

      // mfem::CentGridFunction x_cent(&gd);
      // cout << "Size of x_cent is " << x_cent.Size() << '\n';
      // x_cent.ProjectCoefficient(u0);
      // cout << "\n\n\n\nCheck the the center values:\n";
      // x_cent.Print(cout, 7);

      // gd.GetProlongationMatrix()->Mult(x_cent, x);
      // cout << "\n\n\n\nCheck the results:\n";
      // x.Print(cout,7);
      // x -= x_exact;
      // cout << "Check the error: " << x.Norml2() << '\n';
      delete mesh;
   }

   catch (MachException &exception)
   {
      exception.print_message();
   }
   catch (std::exception &exception)
   {
      cerr << exception.what() << endl;
   }
   return 0;
}

// Exact solution; note that I reversed the flow direction to be clockwise, so
// the problem and mesh are consistent with the LPS paper (that is, because the
// triangles are subdivided from the quads using the opposite diagonal)
void u_exact(const mfem::Vector &x, mfem::Vector &u)
{
   mfem::Vector u0(3);
   u.SetSize(3);

   double rho = 1.0 + 0.98 * sin(2.0 * x(0) * M_PI );
   double v = 0.1;
   double p = 20.0;
   
   u0(0) = rho;
   u0(1) = rho * v;
   u0(2) = p/euler::gami + 0.5 * rho * v * v;

   if (entvar == false)
   {
      u = u0;
   }
   else
   {
      calcEntropyVars<double, 1>(u0.GetData(), u.GetData());
   }

}

void conserv_exact(const mfem::Vector &x, mfem::Vector &u)
{
   u.SetSize(3);
   double rho = 1.0 + 0.98 * sin(2.0 * x(0) * M_PI );
   double v = 0.1;
   double p = 20.0;

   u(0) = rho;
   u(1) = rho * v;
   u(2) = p/euler::gami + 0.5 * rho * v * v;
}

// void upoly(const mfem::Vector &x, mfem::Vector &u)
// {
//    mfem::Vector u0(3);
//    u.SetSize(3);

//    double rho = 1.0 + 0.98 * sin(2.0 * x(0) * M_PI );
//    double v = 0.1;
//    double p = 20.0;
   
//    u0(0) = rho;
//    u0(1) = rho * v;
//    u0(2) = p/euler::gami + 0.5 * rho * v * v;

//    if (entvar == false)
//    {
//       u = u0;
//    }
//    else
//    {
//       calcEntropyVars<double, 1>(u0.GetData(), u.GetData());
//    }
// }



// void uexact_single(const mfem::Vector &x, mfem::Vector &u)
// {
//    u(0) = 20.0;
//    //u(0) = x(0) * x (0) - 7.0 * x(0) + 3.0;
// }

// void uexact_1d(const mfem::Vector &x, mfem::Vector &u)
// {
//    u(0) = 0;
//    for (int i = 2; i >= 0; i--)
//    {
//       u(0) += pow(x(0), i);
//    }
// }


unique_ptr<Mesh> buildQuarterAnnulusMesh(int degree, int num_rad, int num_ang)
{
   auto mesh_ptr = unique_ptr<Mesh>(new Mesh(num_rad, num_ang,
                                             Element::TRIANGLE, true /* gen. edges */,
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