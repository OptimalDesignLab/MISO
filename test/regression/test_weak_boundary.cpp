#include <fstream>
#include <iostream>

#include "catch.hpp"
#include "nlohmann/json.hpp"
#include "mfem.hpp"

#include "electromag_integ.hpp"
#include "coefficient.hpp"

class LinearCoefficient : public mach::StateCoefficient
{
public:
   LinearCoefficient(double val = 1.0) : value(val) {}

   double Eval(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               const double state) override
   {
      return value;
   }

   double EvalStateDeriv(mfem::ElementTransformation &trans,
                        const mfem::IntegrationPoint &ip,
                        const double state) override
   {
      return 0.0;
   }

private:
   double value;
};

class NonLinearCoefficient : public mach::StateCoefficient
{
public:
   NonLinearCoefficient(double exponent = -0.5) : exponent(exponent) {}

   double Eval(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               const double state) override
   {
      // mfem::Vector state;
      // stateGF->GetVectorValue(trans.ElementNo, ip, state);
      // double state_mag = state.Norml2();
      // return pow(state, 2.0);
      // return state;
      // return 0.5*pow(state+1, -0.5);
      return 0.5*pow(state+1, exponent);
   }

   double EvalStateDeriv(mfem::ElementTransformation &trans,
                         const mfem::IntegrationPoint &ip,
                         const double state) override
   {
      // mfem::Vector state;
      // stateGF->GetVectorValue(trans.ElementNo, ip, state);
      // double state_mag = state.Norml2();
      // return 2.0*pow(state, 1.0);
      // return 1.0;
      // return -0.25*pow(state+1, -1.5);
      return 0.5*(exponent)*pow(state+1, exponent-1);
   }

   double EvalState2ndDeriv(mfem::ElementTransformation &trans,
                            const mfem::IntegrationPoint &ip,
                            const double state) override
   {
      // return 2.0;
      // return 0.0;
      // return 0.375*pow(state+1, -2.5);
      return 0.5*(exponent)*(exponent-1)*pow(state+1, exponent-2);
   }

   void EvalRevDiff(double Q_bar,
                    mfem::ElementTransformation &trans,
                    const mfem::IntegrationPoint &ip,
                    const double state,
                    mfem::DenseMatrix &PointMat_bar) override
   { }

private:
   double exponent;
};

double exact_sol(const mfem::Vector &x)
{
   return sin(2*M_PI*x(0))*sin(2*M_PI*x(1));
}

double load_func(const mfem::Vector &x)
{
   return 8*M_PI*M_PI*sin(2*M_PI*x(0))*sin(2*M_PI*x(1));
}

TEST_CASE("Test nonlinear diffusion weak boundary (precondidtioned)")
{
   // define the target state solution error
   std::vector<std::vector<double>> target_strong_error = {
      //nxy = 1, nxy = 2,   nyx = 4,     nyx = 8,     nxy = 16
      {0.474305, 0.52444,   0.255168,    0.0830811,   0.0223573},   // p = 1
      {0.487637, 0.196975,  0.0336973,   0.0043374,   0.000547961}, // p = 2
      {0.508414, 0.0910162, 0.00546391,  0.000329692, 1.96767e-05}, // p = 3
      {0.230863, 0.0100481, 0.000712626, 2.40575e-05, 7.74174e-07}  // p = 4
   };

   // define the target state solution error
   std::vector<std::vector<double>> target_weak_1en1_error = {
      //nxy = 1, nxy = 2,  nyx = 4,    nyx = 8,    nxy = 16
      {2.79379,  5.03961,  0.369677,   6.91149,  0.0425589},   // p = 1
      {0.657405, 0.208379, 0.0504724,  0.0053212, 0.000614984}, // p = 2
      {0.518383, 0.145859, 0.0224066,  0.00259956,  6.13213e-05}, // p = 3
      {0.338661, 0.011035, 0.00220337, 3.37265e-05, 9.04746e-07}  // p = 4
   };

   // define the target state solution error
   std::vector<std::vector<double>> target_weak_1e0_error = {
      //nxy = 1, nxy = 2,   nyx = 4,    nyx = 8,     nxy = 16
      {3.68214,  0.774931,  1.33248,    0.973699,    0.547534},   // p = 1
      {0.701787, 0.265945,  0.865823,   0.00586359,  0.000651138}, // p = 2
      {0.46665,  0.162539,  0.0416442,  0.000922937, 0.000177956}, // p = 3
      {0.312673, 0.0109343, 0.00617102, 0.000165238, 9.68129e-07}  // p = 4
   };

   // define the target state solution error
   std::vector<std::vector<double>> target_weak_1e1_error = {
      //nxy = 1, nxy = 2,  nyx = 4,    nyx = 8,    nxy = 16
      {2.27642,  0.382413, 0.189875,   0.0760158,  0.0216009},   // p = 1
      {0.703506, 0.124923, 0.0281331,  0.00395686, 0.000522676}, // p = 2
      {1.01364,  0.833657, 0.0237724,  0.0021978,  0.000113873}, // p = 3
      {0.401188, 0.141032, 0.00320038, 0.00014236, 4.04948e-05}  // p = 4
   };

   using namespace mfem;

   // generate a 6 element mesh
   int num_edge = 32;
   auto smesh = Mesh::MakeCartesian2D(num_edge,
                                      num_edge,
                                      Element::TRIANGLE);
   

   ParMesh mesh(MPI_COMM_WORLD, smesh);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   LinearCoefficient nu(1e5);
   // NonLinearCoefficient nu;
   FunctionCoefficient bc_val(exact_sol);
   FunctionCoefficient load_coeff(load_func);

   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;
   // ess_bdr = 0;

   double mu = 1e4;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         H1_FECollection fec(p, dim);
         ParFiniteElementSpace fes(&mesh, &fec);

         Array<int> ess_tdof_list;
         fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

         ParNonlinearForm res(&fes);
         res.AddDomainIntegrator(new mach::NonlinearDiffusionIntegrator(nu));

         res.AddBdrFaceIntegrator(
            new mach::NonlinearDGDiffusionIntegrator(nu, bc_val, mu));

         res.SetEssentialTrueDofs(ess_tdof_list);

         ParLinearForm load(&fes);
         load.AddDomainIntegrator(new DomainLFIntegrator(load_coeff));
         load.Assemble();

         ParGridFunction state(&fes);
         state.ProjectCoefficient(load_coeff);
         state = 0.0;

         ParGridFunction res_vec(&fes);
         res_vec = 0.0;
         res.Mult(state, res_vec);
         res_vec -= load;

         res_vec.SetSubVector(ess_tdof_list, 0.0);

         Operator& jac = res.GetGradient(state);

         ParGridFunction delta_state(&fes);
         delta_state = 0.0;

         // HypreILU prec;
         // prec.SetLevelOfFill(1);

         HypreBoomerAMG prec;
         prec.SetPrintLevel(0);

         GMRESSolver gmres(MPI_COMM_WORLD);
         gmres.SetPreconditioner(prec);
         gmres.SetMaxIter(1000);
         gmres.SetKDim(1000);
         gmres.SetPrintLevel(mfem::IterativeSolver::PrintLevel().Summary());
         // gmres.SetPrintLevel(mfem::IterativeSolver::PrintLevel().All());
         gmres.SetOperator(jac);
         gmres.SetAbsTol(1e-12);
         gmres.SetRelTol(1e-12);
         gmres.Mult(res_vec, delta_state);

         state -= delta_state;

         double error = state.ComputeL2Error(bc_val);

         std::cout << "state l2 error (p = " << p << "): " << error << "\n";

      }
   }
}

TEST_CASE("Test nonlinear diffusion weak boundary")
{
   // define the target state solution error
   std::vector<std::vector<double>> target_strong_error = {
      //nxy = 1, nxy = 2,   nyx = 4,     nyx = 8,     nxy = 16
      {0.474305, 0.52444,   0.255168,    0.0830811,   0.0223573},   // p = 1
      {0.487637, 0.196975,  0.0336973,   0.0043374,   0.000547961}, // p = 2
      {0.508414, 0.0910162, 0.00546391,  0.000329692, 1.96767e-05}, // p = 3
      {0.230863, 0.0100481, 0.000712626, 2.40575e-05, 7.74174e-07}  // p = 4
   };

   // define the target state solution error
   std::vector<std::vector<double>> target_weak_1en1_error = {
      //nxy = 1, nxy = 2,  nyx = 4,    nyx = 8,    nxy = 16
      {2.79379,  5.03961,  0.369677,   6.91149,  0.0425589},   // p = 1
      {0.657405, 0.208379, 0.0504724,  0.0053212, 0.000614984}, // p = 2
      {0.518383, 0.145859, 0.0224066,  0.00259956,  6.13213e-05}, // p = 3
      {0.338661, 0.011035, 0.00220337, 3.37265e-05, 9.04746e-07}  // p = 4
   };

   // define the target state solution error
   std::vector<std::vector<double>> target_weak_1e0_error = {
      //nxy = 1, nxy = 2,   nyx = 4,    nyx = 8,     nxy = 16
      {3.68214,  0.774931,  1.33248,    0.973699,    0.547534},   // p = 1
      {0.701787, 0.265945,  0.865823,   0.00586359,  0.000651138}, // p = 2
      {0.46665,  0.162539,  0.0416442,  0.000922937, 0.000177956}, // p = 3
      {0.312673, 0.0109343, 0.00617102, 0.000165238, 9.68129e-07}  // p = 4
   };

   // define the target state solution error
   std::vector<std::vector<double>> target_weak_1e1_error = {
      //nxy = 1, nxy = 2,  nyx = 4,    nyx = 8,    nxy = 16
      {2.27642,  0.382413, 0.189875,   0.0760158,  0.0216009},   // p = 1
      {0.703506, 0.124923, 0.0281331,  0.00395686, 0.000522676}, // p = 2
      {1.01364,  0.833657, 0.0237724,  0.0021978,  0.000113873}, // p = 3
      {0.401188, 0.141032, 0.00320038, 0.00014236, 4.04948e-05}  // p = 4
   };

   using namespace mfem;

   // generate a 6 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge,
                                     num_edge,
                                     Element::TRIANGLE);
   

   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   // LinearCoefficient nu;
   NonLinearCoefficient nu;
   FunctionCoefficient bc_val(exact_sol);
   FunctionCoefficient load_coeff(load_func);

   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   // ess_bdr = 1;
   ess_bdr = 0;

   double mu = 1e0;

   for (int p = 1; p <= 1; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         Array<int> ess_tdof_list;
         fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

         NonlinearForm res(&fes);
         res.AddDomainIntegrator(new mach::NonlinearDiffusionIntegrator(nu));

         res.AddBdrFaceIntegrator(
            new mach::NonlinearDGDiffusionIntegrator(nu, bc_val, mu));

         res.SetEssentialTrueDofs(ess_tdof_list);

         LinearForm load(&fes);
         load.AddDomainIntegrator(new DomainLFIntegrator(load_coeff));
         load.Assemble();

         GridFunction state(&fes);
         state.ProjectCoefficient(load_coeff);
         state = 0.0;

         GridFunction res_vec(&fes);
         res_vec = 0.0;
         res.Mult(state, res_vec);
         res_vec -= load;

         res_vec.SetSubVector(ess_tdof_list, 0.0);

         Operator& jac = res.GetGradient(state);


         auto &spmat_jac = dynamic_cast<SparseMatrix &>(jac);

         auto is_symmetric = spmat_jac.IsSymmetric();
         std::cout << "jacobian is symmetric: " << is_symmetric << "\n";

         // spmat_jac.PrintCSR(std::cout);

         DenseMatrix dense_jac;
         spmat_jac.ToDenseMatrix(dense_jac);
         dense_jac.Print(mfem::out, 100);

         // GridFunction delta_state(&fes);
         // delta_state = 0.0;


         // GMRESSolver gmres;
         // gmres.SetMaxIter(1000);
         // gmres.SetKDim(1000);
         // gmres.SetPrintLevel(mfem::IterativeSolver::PrintLevel().Summary());
         // gmres.SetOperator(jac);
         // gmres.SetAbsTol(1e-12);
         // gmres.SetRelTol(1e-12);
         // gmres.Mult(res_vec, delta_state);

         // state -= delta_state;

         // double error = state.ComputeL2Error(bc_val);

         // std::cout << "state l2 error (p = " << p << "): " << error << "\n";

      }
   }
}