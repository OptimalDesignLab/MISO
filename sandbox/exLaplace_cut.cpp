#include "poisson_dg_cut.hpp"
#include "gd.hpp"
#include "cut_quad.hpp"
#include "pcentgridfunc.hpp"
#include "mach_types.hpp"
#include <chrono>
using namespace std::chrono;
using namespace std;
using namespace mfem;
using namespace mach;
double u_exact(const Vector &);
double f_exact(const Vector &);
void exact_function(const Vector &x, Vector &v);
void u_neumann(const Vector &, Vector &);
/// Generate uniform mesh
/// \param[in] N - number of elements in x-y direction
Mesh buildMesh(int N);
double CutComputeL2Error(ParGridFunction &x,
                         ParFiniteElementSpace *fes_gd,
                         Coefficient &exsol,
                         const std::vector<bool> &embeddedElements,
                         std::map<int, IntegrationRule *> &cutSquareIntRules);

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   // 1. Parse command-line options.
   int order = 1;
   int N = 5;
   bool static_cond = false;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = true;
   double sigma = -1.0;
   double kappa = 100.0;
   double cutsize;
   double radius = 1.0;
   OptionsParser args(argc, argv);
   args.AddOption(&order,
                  "-o",
                  "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&N, "-n", "--#elements", "number of mesh elements.");
   args.AddOption(&radius, "-r", "--radius", "radius of circle.");
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

   unique_ptr<Mesh> smesh(new Mesh(buildMesh(N)));
   std::unique_ptr<MeshType> mesh;
   ofstream sol_ofv("square_mesh.vtk");
   sol_ofv.precision(14);
   smesh->PrintVTK(sol_ofv, 0);
   cout << "#elements before nc refinement " << smesh->GetNE() << endl;
   CutCell<2, 1> cutcell(smesh.get());
   circle<2> phi;
   // phi = cutcell.constructLevelSet();

   /// find the elements to refine
   for (int k = 0; k < -1; ++k)
   {
      mfem::Array<int> marked_elements;
      for (int i = 0; i < smesh->GetNE(); ++i)
      {
         if ((cutcell.cutByGeom(i) == true) &&
             (cutcell.insideBoundary(i) == false))
         {
            marked_elements.Append(i);
         }
      }
      smesh->GeneralRefinement(marked_elements, 1);
   }
   mesh.reset(new MeshType(MPI_COMM_WORLD, *smesh));
   cout << "#elements after refinement " << mesh->GetNE() << endl;

   // write mesh after refinement
   ofstream refine("refined_square_mesh_nc.vtk");
   refine.precision(14);
   mesh->PrintVTK(refine, 0);

   // vector of cut interior faces
   std::vector<int> cutInteriorFaces;
   std::map<int, IntegrationRule *> cutSquareIntRules;
   std::map<int, IntegrationRule *> cutSegmentIntRules;
   std::map<int, IntegrationRule *> cutInteriorFaceIntRules;
   // find the elements cut by boundary
   vector<int> cutelems;
   vector<int> solid_elements;
   vector<int> cutinteriorFaces;
   for (int i = 0; i < mesh->GetNE(); ++i)
   {
      if (cutcell.cutByGeom(i) == true)
      {
         cutelems.push_back(i);
      }
      if (cutcell.insideBoundary(i) == true)
      {
         solid_elements.push_back(i);
      }
   }
   // cout << "elements cut by circle:  " << endl;
   // for (int i = 0; i < cutelems.size(); ++i)
   // {
   //    cout << cutelems.at(i) << endl;
   // }
   // cout << "elements completely inside circle:  " << endl;
   // for (int i = 0; i < solid_elements.size(); ++i)
   // {
   //    cout << solid_elements.at(i) << endl;
   // }
   cout << "#elements completely inside circle:  " << solid_elements.size()
        << endl;
   int dim = mesh->Dimension();
   for (int i = 0; i < mesh->GetNumFaces(); ++i)
   {
      FaceElementTransformations *tr;
      // tr = mesh->GetInteriorFaceTransformations(i);
      tr = mesh->GetFaceElementTransformations(i);
      if (tr->Elem2No >= 0)
      {
         if ((find(cutelems.begin(), cutelems.end(), tr->Elem1No) !=
              cutelems.end()) &&
             (find(cutelems.begin(), cutelems.end(), tr->Elem2No) !=
              cutelems.end()))
         {
            if (!cutcell.findImmersedFace(tr->Face->ElementNo))
            {
               cutInteriorFaces.push_back(tr->Face->ElementNo);
            }
         }
      }
   }
   cout << "dimension is " << dim << endl;
   std::cout << "Number of elements: " << mesh->GetNE() << '\n';
   int deg = order + 2;
   deg = min(deg * deg, 10);
   // define map for integration rule for cut elements
   cutcell.GetCutElementIntRule(cutelems, deg, radius, cutSquareIntRules);
   cutcell.GetCutSegmentIntRule(cutelems,
                                cutInteriorFaces,
                                deg,
                                radius,
                                cutSegmentIntRules,
                                cutInteriorFaceIntRules);
   std::vector<bool> embeddedElements;
   for (int i = 0; i < mesh->GetNE(); ++i)
   {
      if (cutcell.insideBoundary(i) == true)
      {
         embeddedElements.push_back(true);
      }
      else
      {
         embeddedElements.push_back(false);
      }
   }
   std::map<int, bool> immersedFaces;
   for (int i = 0; i < mesh->GetNumFaces(); ++i)
   {
      FaceElementTransformations *tr;
      tr = mesh->GetInteriorFaceTransformations(i);
      if (tr != NULL)
      {
         if ((embeddedElements.at(tr->Elem1No) == true) ||
             (embeddedElements.at(tr->Elem2No)) == true)
         {
            immersedFaces[tr->Face->ElementNo] = true;
         }
         else if (cutcell.findImmersedFace(tr->Face->ElementNo))
         {
            immersedFaces[tr->Face->ElementNo] = true;
         }
         else
         {
            // cout << "immersed Face element is: " << tr->Elem1No << endl;
            immersedFaces[tr->Face->ElementNo] = false;
         }
      }
   }
   // 4. Define a finite element space on the mesh. Here we use discontinuous
   //    finite elements of the specified order >= 0.
   std::unique_ptr<mfem::FiniteElementCollection> fec;
   fec.reset(new DG_FECollection(order, dim));
   std::unique_ptr<SpaceType> fes;
   fes.reset(new ParFiniteElementSpace(mesh.get(), fec.get(), 1));
   using GDSpaceType = mfem::ParGalerkinDifference;
   std::unique_ptr<GDSpaceType> fes_gd;
   fes_gd.reset(new GDSpaceType(mesh.get(),
                                fec.get(),
                                embeddedElements,
                                1,
                                Ordering::byVDIM,
                                order,
                                MPI_COMM_WORLD));
   /// GD finite element space
   cout << "fes_gd created " << endl;
   cout << "Number of unknowns in GD: " << fes_gd->GetTrueVSize() << endl;
   cout << "Number of unknowns: " << fes->GetVSize() << endl;
   cout << "Number of finite element unknowns: " << fes->GetTrueVSize() << endl;
   ParLinearForm *b = new ParLinearForm(fes.get());
   FunctionCoefficient f(f_exact);
   FunctionCoefficient u(u_exact);
   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);
   ConstantCoefficient two(2.0);
   ConstantCoefficient zerop(0.01);
   VectorFunctionCoefficient uN(dim, u_neumann);
   // linear form
   b->AddDomainIntegrator(
       new CutDomainLFIntegrator(f, cutSquareIntRules, embeddedElements));
   // b->AddDomainIntegrator(new CutDGDirichletLFIntegrator(u, one, sigma,
   // kappa,
   //                                                       cutSegmentIntRules));
   b->AddDomainIntegrator(new CutDGNeumannLFIntegrator(uN, phi, cutSegmentIntRules));
   b->AddBdrFaceIntegrator(new DGDirichletLFIntegrator(u, one, sigma, kappa));
   b->Assemble();
   // cout << "RHS: " << endl;
   // b->Print();
   ParGridFunction x(fes.get());
   ParCentGridFunction y(fes_gd.get());
   VectorFunctionCoefficient exact(1, exact_function);
   ParGridFunction xexact(fes.get());
   xexact.ProjectCoefficient(exact);
   // cout << "exact sol created " << endl;
   // xexact.Print();
   cout << "y size " << y.Size() << endl;
   // cout << "prolongated sol " << endl;
   y.ProjectCoefficient(exact);
   fes_gd->GetProlongationMatrix()->Mult(y, x);
   // x.Print();
   // print initial prolonagted solution
   ofstream sol_ofs("dgSolcirclelap_gd_init.vtk");
   sol_ofs.precision(14);
   mesh->PrintVTK(sol_ofs, 1);
   xexact.SaveVTK(sol_ofs, "Solution", 1);
   sol_ofs.close();
   // Vector xdiff(x.Size());
   // x -= xexact;
   // // xdiff = x - xexact;
   // x.Print();
   // cout << "norm " << x.Norml2()<< endl;
   cout << "check prolongation operator " << endl;
   cout << CutComputeL2Error(
               x, fes.get(), u, embeddedElements, cutSquareIntRules)
        << endl;
   // bilinear form
   ParBilinearForm *a = new ParBilinearForm(fes.get());
   a->AddDomainIntegrator(
       new CutDiffusionIntegrator(one, cutSquareIntRules, embeddedElements));
   // a->AddDomainIntegrator(new CutBoundaryFaceIntegrator(one, sigma, kappa,
   // cutSegmentIntRules));
   a->AddInteriorFaceIntegrator(
       new CutDGDiffusionIntegrator(one,
                                    sigma,
                                    kappa,
                                    immersedFaces,
                                    cutinteriorFaces,
                                    cutInteriorFaceIntRules));
   a->AddBdrFaceIntegrator(new DGDiffusionIntegrator(one, sigma, kappa));
   a->Assemble();
   a->Finalize();
   // fes_gd->GetProlongationMatrix()->MultTranspose(*b, bnew);
   // 10. Define the parallel (hypre) matrix and vectors representing a(.,.),
   //     b(.) and the finite element approximation.
   HypreParMatrix *A = a->ParallelAssemble();
   HypreParMatrix *A_gd = RAP(A, fes_gd->Dof_TrueDof_Matrix());
   HYPRE_Int *mat_row_idx;
   mat_row_idx = new HYPRE_Int[2];
   HypreParVector *B = b->ParallelAssemble();
   mat_row_idx[0] = 0;
   mat_row_idx[1] = fes_gd->GlobalTrueVSize();
   cout << "fes_gd->GlobalTrueVSize() " << fes_gd->GlobalTrueVSize() << endl;
   HypreParVector *B_gd = new HypreParVector(
       MPI_COMM_WORLD, (fes_gd->GlobalTrueVSize()), mat_row_idx);
   cout << "B size " << B_gd->GlobalSize() << endl;
   fes_gd->GetProlongationMatrix()->MultTranspose(*B, *B_gd);

   HypreParVector *X = x.ParallelProject();
   HypreParVector *Y = new HypreParVector(
       MPI_COMM_WORLD, (fes_gd->GlobalTrueVSize()), mat_row_idx);
   cout << "Y size " << Y->GlobalSize() << endl;
   fes_gd->GetProlongationMatrix()->MultTranspose(*X, *Y);

   delete a;
   delete b;

   // 11. Depending on the symmetry of A, define and apply a parallel PCG
   // or
   //     GMRES solver for AX=B using the BoomerAMG preconditioner from
   // hypre.
   HypreSolver *amg = new HypreBoomerAMG(*A_gd);
   if (sigma == -1.0)
   {
      HyprePCG pcg(*A_gd);
      pcg.SetTol(1e-12);
      pcg.SetMaxIter(200);
      pcg.SetPrintLevel(2);
      pcg.SetPreconditioner(*amg);
      pcg.Mult(*B_gd, *Y);
   }
   else
   {
      // CustomSolverMonitor monitor(pmesh, &x);
      GMRESSolver gmres(MPI_COMM_WORLD);
      gmres.SetAbsTol(0.0);
      gmres.SetRelTol(1e-12);
      gmres.SetMaxIter(200);
      gmres.SetKDim(10);
      gmres.SetPrintLevel(1);
      gmres.SetOperator(*A_gd);
      gmres.SetPreconditioner(*amg);
      // gmres.SetMonitor(monitor);
      gmres.Mult(*B_gd, *Y);
   }
   delete amg;

   // 12. Extract the parallel grid function corresponding to the
   //    finite element
   //     approximation X. This is the local solution on each
   // processor.
   fes_gd->GetProlongationMatrix()->Mult(*Y, *X);
   x = *X;
   ofstream adj_ofs("poisson_cut_sol_gd.vtk");
   adj_ofs.precision(14);
   mesh->PrintVTK(adj_ofs, 1);
   x.SaveVTK(adj_ofs, "Solution", 1);
   adj_ofs.close();
   double norm =
       CutComputeL2Error(x, fes.get(), u, embeddedElements, cutSquareIntRules);
   cout << "----------------------------- " << endl;
   cout << "mesh size, h = " << 1.0 / N << endl;
   cout << "solution norm: " << norm << endl;
   // x.Print();
   //  11. Free the used memory.
   MPI_Finalize();
   return 0;
}

void exact_function(const Vector &x, Vector &v)
{
   // int dim = x.Size();
   //  v(0) = x(0)*x(0);
   // v(0) = exp(x(0));
   //  v(0) = sin(M_PI*x(0));
   // v(0) = 2.0;
   double s = 20.0;
   v(0) = sin(M_PI * x(0) / s) * sin(M_PI * x(1) / s);
   // v(0) = x(0);
}

double u_exact(const Vector &x)
{
   double s = 20.0;
   return sin(M_PI * x(0) / s) * sin(M_PI * x(1) / s);
   // return 2.0;
   //  return x(0);
   //   return (2*x(0)) - (2*x(1));
}
double f_exact(const Vector &x)
{
   double s = 20.0;
   return 2 * M_PI * M_PI * sin(M_PI * x(0) / s) * sin(M_PI * x(1) / s) * 1.0 /
          (s * s);
   // return 0.0;
}

void u_neumann(const Vector &x, Vector &u)
{
   double s = 20.0;
   u(0) = M_PI * cos(M_PI * x(0) / s) * sin(M_PI * x(1) / s) / s;
   u(1) = M_PI * sin(M_PI * x(0) / s) * cos(M_PI * x(1) / s) / s;
   // u(0) = 0.0;
   // u(1) = 0.0;
}

double CutComputeL2Error(ParGridFunction &x,
                         ParFiniteElementSpace *fes_gd,
                         Coefficient &exsol,
                         const std::vector<bool> &embeddedElements,
                         std::map<int, IntegrationRule *> &cutSquareIntRules)
{
   double error = 0.0;
   const FiniteElement *fe;
   ElementTransformation *T;
   Vector vals;
   int p = 2;
   for (int i = 0; i < fes_gd->GetNE(); i++)
   {
      if (embeddedElements.at(i) == true)
      {
         error += 0.0;
      }
      else
      {
         fe = fes_gd->GetFE(i);
         const IntegrationRule *ir;
         ir = cutSquareIntRules[i];
         if (ir == NULL)
         {
            int intorder = 2 * fe->GetOrder() + 1;  // <----------
            ir = &(IntRules.Get(fe->GetGeomType(), intorder));
         }
         x.GetValues(i, *ir, vals);
         T = fes_gd->GetElementTransformation(i);
         for (int j = 0; j < ir->GetNPoints(); j++)
         {
            const IntegrationPoint &ip = ir->IntPoint(j);
            T->SetIntPoint(&ip);
            double err = fabs(vals(j) - exsol.Eval(*T, ip));
            if (p < infinity())
            {
               err = pow(err, p);
               error += ip.weight * T->Weight() * err;
            }
            else
            {
               error = std::max(error, err);
            }
         }
      }
   }
   if (p < infinity())
   {
      // negative quadrature weights may cause the error to be negative
      if (error < 0.)
      {
         error = -pow(-error, 1. / p);
      }
      else
      {
         error = pow(error, 1. / p);
      }
   }
   return error;
}

Mesh buildMesh(int N)
{
   Mesh mesh = Mesh::MakeCartesian2D(
       N, N, Element::QUADRILATERAL, true, 20.0, 20.0, true);
   return mesh;
}