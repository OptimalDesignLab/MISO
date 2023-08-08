#include <random>

#include "catch.hpp"
#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "current_source_functions.hpp"
#include "mfem_common_integ.hpp"

#include "thermal_residual.hpp"

/// Generate mesh 
/// \param[in] nxy - number of nodes in the x and y directions
mfem::Mesh buildMesh(int nxy = 2, int internal_bdr_attr = 5);

/// Simple nonlinear coefficient
class ThermalConductivityCoefficient : public mach::StateCoefficient
{
public:
   double Eval(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               const double state) override
   {
      return 1.0;
   }

   double EvalStateDeriv(mfem::ElementTransformation &trans,
                         const mfem::IntegrationPoint &ip,
                         const double state) override
   {
      return 0.0;
   }

   double EvalState2ndDeriv(mfem::ElementTransformation &trans,
                            const mfem::IntegrationPoint &ip,
                            const double state) override
   {
      return 0.0;
   }

   void EvalRevDiff(const double Q_bar,
                    mfem::ElementTransformation &trans,
                    const mfem::IntegrationPoint &ip,
                    mfem::DenseMatrix &PointMat_Bar) override
   {}
};

TEMPLATE_TEST_CASE("ThermalResidual sensitivity wrt thermal_load", "", 
                   mfem::H1_FECollection, mfem::L2_FECollection)
{
   std::default_random_engine gen;
   std::uniform_real_distribution<double> uniform_rand(-1.0,1.0);

   using namespace mfem;

   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   auto smesh = buildMesh(num_edge);

   mfem::ParMesh mesh(MPI_COMM_WORLD, smesh);

   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   // ThermalConductivityCoefficient kappa;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         TestType fec(p, dim);
         mfem::ParFiniteElementSpace fes(&mesh, &fec);

         std::map<std::string, mach::FiniteElementState> fields;
         fields.emplace("state", mach::FiniteElementState(mesh, fes, "state"));
         auto &state = fields.at("state");

         fields.emplace("dirichlet_bc", mach::FiniteElementState(mesh, fes, "dirichlet_bc"));

         auto &mesh_gf = *dynamic_cast<mfem::ParGridFunction *>(mesh.GetNodes());
         auto *mesh_fespace = mesh_gf.ParFESpace();
         /// create new state vector copying the mesh's fe space
         fields.emplace("mesh_coords",
                        mach::FiniteElementState(mesh, *mesh_fespace, "mesh_coords"));
         auto &mesh_coords = fields.at("mesh_coords");
         /// set the values of the new GF to those of the mesh's old nodes
         mesh_coords.gridFunc() = mesh_gf;
         /// tell the mesh to use this GF for its Nodes
         /// (and that it doesn't own it)
         mesh.NewNodes(mesh_coords.gridFunc(), false);


         mfem::Vector state_tv(state.space().GetTrueVSize());
         for (int i = 0; i < state_tv.Size(); ++i)
         {
            state_tv(i) = uniform_rand(gen);
         }

         mfem::Vector mesh_coords_tv(mesh_coords.space().GetTrueVSize());
         mesh_coords.setTrueVec(mesh_coords_tv);

         mfem::Vector thermal_load_tv(state.space().GetTrueVSize());
         for (int i = 0; i < thermal_load_tv.Size(); ++i)
         {
            thermal_load_tv(i) = uniform_rand(gen);
         }

         mfem::Vector res_bar(state.space().GetTrueVSize());
         for (int i = 0; i < res_bar.Size(); ++i)
         {
            res_bar(i) = uniform_rand(gen);
         }

         mfem::Vector pert_vec(state.space().GetTrueVSize());
         for (int i = 0; i < pert_vec.Size(); ++i)
         {
            pert_vec(i) = uniform_rand(gen);
         }

         auto options = R"({
            "space-dis": {
               "basis-type": "h1",
               "degree": 1
            },
            "lin-prec": {
               "type": "hypreboomeramg",
               "printlevel": 0
            },
            "components": {
               "box1": {
                  "attrs": [1],
                  "material": {
                     "name": "box1"
                  }
               },
               "box2": {
                  "attrs": [2],
                  "material": {
                     "name": "box2"
                  }
               }
            },
            "bcs": {
               "convection": [1, 2],
               "outflux": [3, 4]
            }
         })"_json;

         options["space-dis"]["degree"] = p;
         if (std::is_same<TestType, mfem::H1_FECollection>::value)
         {
            options["space-dis"]["basis-type"] = "h1";
         }
         else if (std::is_same<TestType, mfem::L2_FECollection>::value)
         {
            options["space-dis"]["basis-type"] = "dg";
         }

         auto materials = R"({
            "box1": {
               "kappa": 1.0
            },
            "box2": {
               "kappa": 1.0
            }
         })"_json;

         mach::ThermalResidual res(fes, fields, options, materials);

         mach::MachInputs inputs{
            {"state", state_tv},
            {"mesh_coords", mesh_coords_tv},
            {"h", 10.0},
            {"fluid_temp", 20.0},
            {"thermal_load", thermal_load_tv}
         };

         setInputs(res, inputs);

         mfem::Vector res_dot(state.space().GetTrueVSize());
         res_dot = 0.0;
         jacobianVectorProduct(res, pert_vec, "thermal_load", res_dot);
         double drdp_fwd = res_bar * res_dot;

         mfem::Vector wrt_bar(state.space().GetTrueVSize());
         wrt_bar = 0.0;
         vectorJacobianProduct(res, res_bar, "thermal_load", wrt_bar);
         double drdp_rev = wrt_bar * pert_vec;

         // now compute the finite-difference approximation...
         thermal_load_tv.Add(delta, pert_vec);
         mfem::Vector drdp_fd_p(state.space().GetTrueVSize());
         drdp_fd_p = 0.0;
         setInputs(res, inputs);
         evaluate(res, inputs, drdp_fd_p);

         thermal_load_tv.Add(-2 * delta, pert_vec);
         mfem::Vector drdp_fd_m(state.space().GetTrueVSize());
         drdp_fd_m = 0.0;
         setInputs(res, inputs);
         evaluate(res, inputs, drdp_fd_m);

         mfem::Vector scratch(state.space().GetTrueVSize());
         scratch = 0.0;
         scratch += drdp_fd_p;
         scratch -= drdp_fd_m;
         scratch /= (2 * delta);

         double drdp_fd = res_bar * scratch;

         std::cout << "drdp_fwd: " << drdp_fwd << " drdp_rev: " << drdp_rev << " drdp_fd: " << drdp_fd << "\n";

         REQUIRE(drdp_fwd == Approx(drdp_fd).margin(1e-8));
         REQUIRE(drdp_rev == Approx(drdp_fd).margin(1e-8));
      }
   }
}

TEMPLATE_TEST_CASE("ThermalResidual sensitivity wrt mesh_coords", "", 
                   mfem::H1_FECollection, mfem::L2_FECollection)
{
   std::default_random_engine gen;
   std::uniform_real_distribution<double> uniform_rand(-1.0,1.0);

   using namespace mfem;

   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   auto smesh = buildMesh(num_edge);

   mfem::ParMesh mesh(MPI_COMM_WORLD, smesh);

   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   // ThermalConductivityCoefficient kappa;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         TestType fec(p, dim);
         mfem::ParFiniteElementSpace fes(&mesh, &fec);

         std::map<std::string, mach::FiniteElementState> fields;
         fields.emplace("state", mach::FiniteElementState(mesh, fes, "state"));
         auto &state = fields.at("state");

         fields.emplace("dirichlet_bc", mach::FiniteElementState(mesh, fes, "dirichlet_bc"));

         auto &mesh_gf = *dynamic_cast<mfem::ParGridFunction *>(mesh.GetNodes());
         auto *mesh_fespace = mesh_gf.ParFESpace();
         /// create new state vector copying the mesh's fe space
         fields.emplace("mesh_coords",
                        mach::FiniteElementState(mesh, *mesh_fespace, "mesh_coords"));
         auto &mesh_coords = fields.at("mesh_coords");
         /// set the values of the new GF to those of the mesh's old nodes
         mesh_coords.gridFunc() = mesh_gf;
         /// tell the mesh to use this GF for its Nodes
         /// (and that it doesn't own it)
         mesh.NewNodes(mesh_coords.gridFunc(), false);


         mfem::Vector state_tv(state.space().GetTrueVSize());
         for (int i = 0; i < state_tv.Size(); ++i)
         {
            state_tv(i) = uniform_rand(gen);
         }

         mfem::Vector mesh_coords_tv(mesh_coords.space().GetTrueVSize());
         mesh_coords.setTrueVec(mesh_coords_tv);

         mfem::Vector thermal_load_tv(state.space().GetTrueVSize());
         for (int i = 0; i < thermal_load_tv.Size(); ++i)
         {
            thermal_load_tv(i) = uniform_rand(gen);
         }

         mfem::Vector res_bar(state.space().GetTrueVSize());
         for (int i = 0; i < res_bar.Size(); ++i)
         {
            res_bar(i) = uniform_rand(gen);
         }

         mfem::Vector pert_vec(mesh_coords.space().GetTrueVSize());
         for (int i = 0; i < pert_vec.Size(); ++i)
         {
            pert_vec(i) = uniform_rand(gen);
         }

         // res_bar = 0.0;
         // res_bar(0) = 1.0;
         // pert_vec = 0.0;
         // pert_vec(0) = 1.0;

         auto options = R"({
            "space-dis": {
               "basis-type": "h1",
               "degree": 1
            },
            "lin-prec": {
               "type": "hypreboomeramg",
               "printlevel": 0
            },
            "components": {
               "box1": {
                  "attrs": [1],
                  "material": {
                     "name": "box1"
                  }
               },
               "box2": {
                  "attrs": [2],
                  "material": {
                     "name": "box2"
                  }
               }
            },
            "bcs": {
               "convection": [1, 2],
               "outflux": [3, 4]
            }
         })"_json;

         options["space-dis"]["degree"] = p;
         if (std::is_same<TestType, mfem::H1_FECollection>::value)
         {
            options["space-dis"]["basis-type"] = "h1";
         }
         else if (std::is_same<TestType, mfem::L2_FECollection>::value)
         {
            options["space-dis"]["basis-type"] = "dg";
         }

         auto materials = R"({
            "box1": {
               "kappa": 1.0
            },
            "box2": {
               "kappa": 1.0
            }
         })"_json;

         mach::ThermalResidual res(fes, fields, options, materials);

         mach::MachInputs inputs{
            {"state", state_tv},
            {"mesh_coords", mesh_coords_tv},
            {"h", 10.0},
            {"fluid_temp", 20.0},
            {"thermal_load", thermal_load_tv}
         };

         setInputs(res, inputs);

         mfem::Vector res_dot(state.space().GetTrueVSize());
         res_dot = 0.0;
         jacobianVectorProduct(res, pert_vec, "mesh_coords", res_dot);
         double drdp_fwd = res_bar * res_dot;

         mfem::Vector wrt_bar(mesh_coords.space().GetTrueVSize());
         wrt_bar = 0.0;
         vectorJacobianProduct(res, res_bar, "mesh_coords", wrt_bar);
         double drdp_rev = wrt_bar * pert_vec;

         // now compute the finite-difference approximation...
         mesh_coords_tv.Add(delta, pert_vec);
         mfem::Vector drdp_fd_p(state.space().GetTrueVSize());
         drdp_fd_p = 0.0;
         setInputs(res, inputs);
         evaluate(res, inputs, drdp_fd_p);

         mesh_coords_tv.Add(-2 * delta, pert_vec);
         mfem::Vector drdp_fd_m(state.space().GetTrueVSize());
         drdp_fd_m = 0.0;
         setInputs(res, inputs);
         evaluate(res, inputs, drdp_fd_m);

         mfem::Vector scratch(state.space().GetTrueVSize());
         scratch = 0.0;
         scratch += drdp_fd_p;
         scratch -= drdp_fd_m;
         scratch /= (2 * delta);

         double drdp_fd = res_bar * scratch;

         std::cout << "drdp_rev: " << drdp_rev << " drdp_fd: " << drdp_fd << "\n";
         // REQUIRE(drdp_fwd == Approx(drdp_fd).margin(1e-8));
         REQUIRE(drdp_rev == Approx(drdp_fd).margin(1e-8));
         mesh_coords_tv.Add(delta, pert_vec);
      }
   }
}

TEMPLATE_TEST_CASE("ThermalResidual sensitivity wrt h", "", 
                   mfem::H1_FECollection, mfem::L2_FECollection)
{
   std::default_random_engine gen;
   std::uniform_real_distribution<double> uniform_rand(-1.0,1.0);

   using namespace mfem;

   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   auto smesh = buildMesh(num_edge);

   mfem::ParMesh mesh(MPI_COMM_WORLD, smesh);

   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   // ThermalConductivityCoefficient kappa;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         TestType fec(p, dim);
         mfem::ParFiniteElementSpace fes(&mesh, &fec);

         std::map<std::string, mach::FiniteElementState> fields;
         fields.emplace("state", mach::FiniteElementState(mesh, fes, "state"));
         auto &state = fields.at("state");

         fields.emplace("dirichlet_bc", mach::FiniteElementState(mesh, fes, "dirichlet_bc"));

         auto &mesh_gf = *dynamic_cast<mfem::ParGridFunction *>(mesh.GetNodes());
         auto *mesh_fespace = mesh_gf.ParFESpace();
         /// create new state vector copying the mesh's fe space
         fields.emplace("mesh_coords",
                        mach::FiniteElementState(mesh, *mesh_fespace, "mesh_coords"));
         auto &mesh_coords = fields.at("mesh_coords");
         /// set the values of the new GF to those of the mesh's old nodes
         mesh_coords.gridFunc() = mesh_gf;
         /// tell the mesh to use this GF for its Nodes
         /// (and that it doesn't own it)
         mesh.NewNodes(mesh_coords.gridFunc(), false);


         mfem::Vector state_tv(state.space().GetTrueVSize());
         for (int i = 0; i < state_tv.Size(); ++i)
         {
            state_tv(i) = uniform_rand(gen);
         }

         mfem::Vector mesh_coords_tv(mesh_coords.space().GetTrueVSize());
         mesh_coords.setTrueVec(mesh_coords_tv);

         mfem::Vector thermal_load_tv(state.space().GetTrueVSize());
         for (int i = 0; i < thermal_load_tv.Size(); ++i)
         {
            thermal_load_tv(i) = uniform_rand(gen);
         }

         mfem::Vector res_bar(state.space().GetTrueVSize());
         for (int i = 0; i < res_bar.Size(); ++i)
         {
            res_bar(i) = uniform_rand(gen);
         }

         mfem::Vector pert_vec(mesh_coords.space().GetTrueVSize());
         for (int i = 0; i < pert_vec.Size(); ++i)
         {
            pert_vec(i) = uniform_rand(gen);
         }

         // res_bar = 0.0;
         // res_bar(0) = 1.0;
         // pert_vec = 0.0;
         // pert_vec(0) = 1.0;

         auto options = R"({
            "space-dis": {
               "basis-type": "h1",
               "degree": 1
            },
            "lin-prec": {
               "type": "hypreboomeramg",
               "printlevel": 0
            },
            "components": {
               "box1": {
                  "attrs": [1],
                  "material": {
                     "name": "box1"
                  }
               },
               "box2": {
                  "attrs": [2],
                  "material": {
                     "name": "box2"
                  }
               }
            },
            "bcs": {
               "convection": [1, 2],
               "outflux": [3, 4]
            }
         })"_json;

         options["space-dis"]["degree"] = p;

         if (std::is_same<TestType, mfem::H1_FECollection>::value)
         {
            options["space-dis"]["basis-type"] = "h1";
         }
         else if (std::is_same<TestType, mfem::L2_FECollection>::value)
         {
            options["space-dis"]["basis-type"] = "dg";
         }

         auto materials = R"({
            "box1": {
               "kappa": 1.0
            },
            "box2": {
               "kappa": 1.0
            }
         })"_json;

         mach::ThermalResidual res(fes, fields, options, materials);

         mach::MachInputs inputs{
            {"state", state_tv},
            {"mesh_coords", mesh_coords_tv},
            {"h", 10.0},
            {"fluid_temp", 20.0},
            {"thermal_load", thermal_load_tv}
         };

         setInputs(res, inputs);

         double pert = uniform_rand(gen);
         // evaluate reverse mode sensitivity
         auto h_bar = vectorJacobianProduct(res, res_bar, "h");
         auto drdp_rev = pert * h_bar;

         // now compute the finite-difference approximation...
         inputs["h"] = 10.0 + delta * pert;
         mfem::Vector drdp_fd_p(state.space().GetTrueVSize());
         drdp_fd_p = 0.0;
         setInputs(res, inputs);
         evaluate(res, inputs, drdp_fd_p);

         inputs["h"] = 10.0 - delta * pert;
         mfem::Vector drdp_fd_m(state.space().GetTrueVSize());
         drdp_fd_m = 0.0;
         setInputs(res, inputs);
         evaluate(res, inputs, drdp_fd_m);

         mfem::Vector scratch(state.space().GetTrueVSize());
         scratch = 0.0;
         scratch += drdp_fd_p;
         scratch -= drdp_fd_m;
         scratch /= (2 * delta);

         double drdp_fd = res_bar * scratch;

         std::cout << "drdp_rev: " << drdp_rev << " drdp_fd: " << drdp_fd << "\n";
         // REQUIRE(drdp_fwd == Approx(drdp_fd).margin(1e-8));
         REQUIRE(drdp_rev == Approx(drdp_fd).margin(1e-8));
         mesh_coords_tv.Add(delta, pert_vec);
      }
   }
}

TEMPLATE_TEST_CASE("ThermalResidual sensitivity wrt fluid_temp", "", 
                   mfem::H1_FECollection, mfem::L2_FECollection)
{
   std::default_random_engine gen;
   std::uniform_real_distribution<double> uniform_rand(-1.0,1.0);

   using namespace mfem;

   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   auto smesh = buildMesh(num_edge);

   mfem::ParMesh mesh(MPI_COMM_WORLD, smesh);

   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   // ThermalConductivityCoefficient kappa;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         TestType fec(p, dim);
         mfem::ParFiniteElementSpace fes(&mesh, &fec);

         std::map<std::string, mach::FiniteElementState> fields;
         fields.emplace("state", mach::FiniteElementState(mesh, fes, "state"));
         auto &state = fields.at("state");

         fields.emplace("dirichlet_bc", mach::FiniteElementState(mesh, fes, "dirichlet_bc"));

         auto &mesh_gf = *dynamic_cast<mfem::ParGridFunction *>(mesh.GetNodes());
         auto *mesh_fespace = mesh_gf.ParFESpace();
         /// create new state vector copying the mesh's fe space
         fields.emplace("mesh_coords",
                        mach::FiniteElementState(mesh, *mesh_fespace, "mesh_coords"));
         auto &mesh_coords = fields.at("mesh_coords");
         /// set the values of the new GF to those of the mesh's old nodes
         mesh_coords.gridFunc() = mesh_gf;
         /// tell the mesh to use this GF for its Nodes
         /// (and that it doesn't own it)
         mesh.NewNodes(mesh_coords.gridFunc(), false);


         mfem::Vector state_tv(state.space().GetTrueVSize());
         for (int i = 0; i < state_tv.Size(); ++i)
         {
            state_tv(i) = uniform_rand(gen);
         }

         mfem::Vector mesh_coords_tv(mesh_coords.space().GetTrueVSize());
         mesh_coords.setTrueVec(mesh_coords_tv);

         mfem::Vector thermal_load_tv(state.space().GetTrueVSize());
         for (int i = 0; i < thermal_load_tv.Size(); ++i)
         {
            thermal_load_tv(i) = uniform_rand(gen);
         }

         mfem::Vector res_bar(state.space().GetTrueVSize());
         for (int i = 0; i < res_bar.Size(); ++i)
         {
            res_bar(i) = uniform_rand(gen);
         }

         mfem::Vector pert_vec(mesh_coords.space().GetTrueVSize());
         for (int i = 0; i < pert_vec.Size(); ++i)
         {
            pert_vec(i) = uniform_rand(gen);
         }

         // res_bar = 0.0;
         // res_bar(0) = 1.0;
         // pert_vec = 0.0;
         // pert_vec(0) = 1.0;

         auto options = R"({
            "space-dis": {
               "basis-type": "h1",
               "degree": 1
            },
            "lin-prec": {
               "type": "hypreboomeramg",
               "printlevel": 0
            },
            "components": {
               "box1": {
                  "attrs": [1],
                  "material": {
                     "name": "box1"
                  }
               },
               "box2": {
                  "attrs": [2],
                  "material": {
                     "name": "box2"
                  }
               }
            },
            "bcs": {
               "convection": [1, 2],
               "outflux": [3, 4]
            }
         })"_json;

         options["space-dis"]["degree"] = p;

         if (std::is_same<TestType, mfem::H1_FECollection>::value)
         {
            options["space-dis"]["basis-type"] = "h1";
         }
         else if (std::is_same<TestType, mfem::L2_FECollection>::value)
         {
            options["space-dis"]["basis-type"] = "dg";
         }

         auto materials = R"({
            "box1": {
               "kappa": 1.0
            },
            "box2": {
               "kappa": 1.0
            }
         })"_json;

         mach::ThermalResidual res(fes, fields, options, materials);

         mach::MachInputs inputs{
            {"state", state_tv},
            {"mesh_coords", mesh_coords_tv},
            {"h", 10.0},
            {"fluid_temp", 20.0},
            {"thermal_load", thermal_load_tv}
         };

         setInputs(res, inputs);

         double pert = uniform_rand(gen);
         // evaluate reverse mode sensitivity
         auto h_bar = vectorJacobianProduct(res, res_bar, "fluid_temp");
         auto drdp_rev = pert * h_bar;

         // now compute the finite-difference approximation...
         inputs["fluid_temp"] = 20.0 + delta * pert;
         mfem::Vector drdp_fd_p(state.space().GetTrueVSize());
         drdp_fd_p = 0.0;
         setInputs(res, inputs);
         evaluate(res, inputs, drdp_fd_p);

         inputs["fluid_temp"] = 20.0 - delta * pert;
         mfem::Vector drdp_fd_m(state.space().GetTrueVSize());
         drdp_fd_m = 0.0;
         setInputs(res, inputs);
         evaluate(res, inputs, drdp_fd_m);

         mfem::Vector scratch(state.space().GetTrueVSize());
         scratch = 0.0;
         scratch += drdp_fd_p;
         scratch -= drdp_fd_m;
         scratch /= (2 * delta);

         double drdp_fd = res_bar * scratch;

         std::cout << "drdp_rev: " << drdp_rev << " drdp_fd: " << drdp_fd << "\n";
         // REQUIRE(drdp_fwd == Approx(drdp_fd).margin(1e-8));
         REQUIRE(drdp_rev == Approx(drdp_fd).margin(1e-8));
         mesh_coords_tv.Add(delta, pert_vec);
      }
   }
}

TEST_CASE("ThermalResidual with interfaces sensitivity wrt mesh_coords")
{
   std::default_random_engine gen;
   std::uniform_real_distribution<double> uniform_rand(-1.0,1.0);

   using namespace mfem;

   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   auto smesh = buildMesh(num_edge);

   mfem::ParMesh mesh(MPI_COMM_WORLD, smesh);

   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         mfem::L2_FECollection fec(p, dim);
         mfem::ParFiniteElementSpace fes(&mesh, &fec);

         std::map<std::string, mach::FiniteElementState> fields;
         fields.emplace("state", mach::FiniteElementState(mesh, fes, "state"));
         auto &state = fields.at("state");

         fields.emplace("dirichlet_bc", mach::FiniteElementState(mesh, fes, "dirichlet_bc"));

         auto &mesh_gf = *dynamic_cast<mfem::ParGridFunction *>(mesh.GetNodes());
         auto *mesh_fespace = mesh_gf.ParFESpace();
         /// create new state vector copying the mesh's fe space
         fields.emplace("mesh_coords",
                        mach::FiniteElementState(mesh, *mesh_fespace, "mesh_coords"));
         auto &mesh_coords = fields.at("mesh_coords");
         /// set the values of the new GF to those of the mesh's old nodes
         mesh_coords.gridFunc() = mesh_gf;
         /// tell the mesh to use this GF for its Nodes
         /// (and that it doesn't own it)
         mesh.NewNodes(mesh_coords.gridFunc(), false);


         mfem::Vector state_tv(state.space().GetTrueVSize());
         for (int i = 0; i < state_tv.Size(); ++i)
         {
            state_tv(i) = uniform_rand(gen);
         }

         mfem::Vector mesh_coords_tv(mesh_coords.space().GetTrueVSize());
         mesh_coords.setTrueVec(mesh_coords_tv);

         mfem::Vector thermal_load_tv(state.space().GetTrueVSize());
         for (int i = 0; i < thermal_load_tv.Size(); ++i)
         {
            thermal_load_tv(i) = uniform_rand(gen);
         }

         mfem::Vector res_bar(state.space().GetTrueVSize());
         for (int i = 0; i < res_bar.Size(); ++i)
         {
            res_bar(i) = uniform_rand(gen);
         }

         mfem::Vector pert_vec(mesh_coords.space().GetTrueVSize());
         for (int i = 0; i < pert_vec.Size(); ++i)
         {
            pert_vec(i) = uniform_rand(gen);
         }

         // res_bar = 0.0;
         // res_bar(0) = 1.0;
         // pert_vec = 0.0;
         // pert_vec(0) = 1.0;

         auto options = R"({
            "space-dis": {
               "basis-type": "dg",
               "degree": 1
            },
            "lin-prec": {
               "type": "hypreboomeramg",
               "printlevel": 0
            },
            "components": {
               "box1": {
                  "attrs": [1],
                  "material": {
                     "name": "box1"
                  }
               },
               "box2": {
                  "attrs": [2],
                  "material": {
                     "name": "box2"
                  }
               }
            },
            "interfaces": {
            },
            "bcs": {
               "convection": [1, 2],
               "outflux": [3, 4]
            }
         })"_json;

         options["space-dis"]["degree"] = p;

         for (int g = 0; g < 2; ++g)
         {
            DYNAMIC_SECTION("...with " << (g == 0 ? "thermal contact" : "internal convection") << " interface")
            {
               if (g == 0)
               {
                  options["interfaces"] = R"({
                     "thermal_contact_resistance": {
                        "interface": {
                           "attrs": [5],
                           "h_c": 10
                        }
                     }
                  })"_json;
               }
               else
               {
                  options["interfaces"] = R"({
                     "convection": {
                        "interface": {
                           "attrs": [5],
                           "h_c": 300,
                           "theta_f": 300
                        }
                     }
                  })"_json;
               }

               auto materials = R"({
                  "box1": {
                     "kappa": 1.0
                  },
                  "box2": {
                     "kappa": 1.0
                  }
               })"_json;

               mach::ThermalResidual res(fes, fields, options, materials);

               mach::MachInputs inputs{
                  {"state", state_tv},
                  {"mesh_coords", mesh_coords_tv},
                  {"h", 10.0},
                  {"fluid_temp", 20.0},
                  {"thermal_load", thermal_load_tv}
               };

               setInputs(res, inputs);

               // mfem::Vector res_dot(state.space().GetTrueVSize());
               // res_dot = 0.0;
               // jacobianVectorProduct(res, pert_vec, "mesh_coords", res_dot);
               // double drdp_fwd = res_bar * res_dot;

               mfem::Vector wrt_bar(mesh_coords.space().GetTrueVSize());
               wrt_bar = 0.0;
               vectorJacobianProduct(res, res_bar, "mesh_coords", wrt_bar);

               double drdp_rev = wrt_bar * pert_vec;

               // now compute the finite-difference approximation...
               mesh_coords_tv.Add(delta, pert_vec);
               mfem::Vector drdp_fd_p(state.space().GetTrueVSize());
               drdp_fd_p = 0.0;
               setInputs(res, inputs);
               evaluate(res, inputs, drdp_fd_p);

               mesh_coords_tv.Add(-2 * delta, pert_vec);
               mfem::Vector drdp_fd_m(state.space().GetTrueVSize());
               drdp_fd_m = 0.0;
               setInputs(res, inputs);
               evaluate(res, inputs, drdp_fd_m);

               mfem::Vector scratch(state.space().GetTrueVSize());
               scratch = 0.0;
               scratch += drdp_fd_p;
               scratch -= drdp_fd_m;
               scratch /= (2 * delta);

               double drdp_fd = res_bar * scratch;

               std::cout << "drdp_rev: " << drdp_rev << " drdp_fd: " << drdp_fd << "\n";
               // REQUIRE(drdp_fwd == Approx(drdp_fd).margin(1e-8));
               REQUIRE(drdp_rev == Approx(drdp_fd).margin(1e-8));
               mesh_coords_tv.Add(delta, pert_vec);
            }
         }
      }
   }
}

TEST_CASE("ThermalResidual with interfaces sensitivity wrt h_c:interface")
{
   std::default_random_engine gen;
   std::uniform_real_distribution<double> uniform_rand(-1.0,1.0);

   using namespace mfem;

   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   auto smesh = buildMesh(num_edge);

   mfem::ParMesh mesh(MPI_COMM_WORLD, smesh);

   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         mfem::L2_FECollection fec(p, dim);
         mfem::ParFiniteElementSpace fes(&mesh, &fec);

         std::map<std::string, mach::FiniteElementState> fields;
         fields.emplace("state", mach::FiniteElementState(mesh, fes, "state"));
         auto &state = fields.at("state");

         fields.emplace("dirichlet_bc", mach::FiniteElementState(mesh, fes, "dirichlet_bc"));

         auto &mesh_gf = *dynamic_cast<mfem::ParGridFunction *>(mesh.GetNodes());
         auto *mesh_fespace = mesh_gf.ParFESpace();
         /// create new state vector copying the mesh's fe space
         fields.emplace("mesh_coords",
                        mach::FiniteElementState(mesh, *mesh_fespace, "mesh_coords"));
         auto &mesh_coords = fields.at("mesh_coords");
         /// set the values of the new GF to those of the mesh's old nodes
         mesh_coords.gridFunc() = mesh_gf;
         /// tell the mesh to use this GF for its Nodes
         /// (and that it doesn't own it)
         mesh.NewNodes(mesh_coords.gridFunc(), false);


         mfem::Vector state_tv(state.space().GetTrueVSize());
         for (int i = 0; i < state_tv.Size(); ++i)
         {
            state_tv(i) = uniform_rand(gen);
         }

         mfem::Vector mesh_coords_tv(mesh_coords.space().GetTrueVSize());
         mesh_coords.setTrueVec(mesh_coords_tv);

         mfem::Vector thermal_load_tv(state.space().GetTrueVSize());
         for (int i = 0; i < thermal_load_tv.Size(); ++i)
         {
            thermal_load_tv(i) = uniform_rand(gen);
         }

         mfem::Vector res_bar(state.space().GetTrueVSize());
         for (int i = 0; i < res_bar.Size(); ++i)
         {
            res_bar(i) = uniform_rand(gen);
         }

         mfem::Vector pert_vec(mesh_coords.space().GetTrueVSize());
         for (int i = 0; i < pert_vec.Size(); ++i)
         {
            pert_vec(i) = uniform_rand(gen);
         }

         // res_bar = 0.0;
         // res_bar(0) = 1.0;
         // pert_vec = 0.0;
         // pert_vec(0) = 1.0;

         auto options = R"({
            "space-dis": {
               "basis-type": "dg",
               "degree": 1
            },
            "lin-prec": {
               "type": "hypreboomeramg",
               "printlevel": 0
            },
            "components": {
               "box1": {
                  "attrs": [1],
                  "material": {
                     "name": "box1"
                  }
               },
               "box2": {
                  "attrs": [2],
                  "material": {
                     "name": "box2"
                  }
               }
            },
            "interfaces": {
            },
            "bcs": {
               "convection": [1, 2],
               "outflux": [3, 4]
            }
         })"_json;

         options["space-dis"]["degree"] = p;

         for (int g = 0; g < 2; ++g)
         {
            DYNAMIC_SECTION("...with " << (g == 0 ? "thermal contact" : "internal convection") << " interface")
            {
               if (g == 0)
               {
                  options["interfaces"] = R"({
                     "thermal_contact_resistance": {
                        "interface": {
                           "attrs": [5],
                           "h_c": 10
                        }
                     }
                  })"_json;
               }
               else
               {
                  options["interfaces"] = R"({
                     "convection": {
                        "interface": {
                           "attrs": [5],
                           "h_c": 300,
                           "theta_f": 300
                        }
                     }
                  })"_json;
               }

               auto materials = R"({
                  "box1": {
                     "kappa": 1.0
                  },
                  "box2": {
                     "kappa": 1.0
                  }
               })"_json;

               mach::ThermalResidual res(fes, fields, options, materials);

               mach::MachInputs inputs{
                  {"state", state_tv},
                  {"mesh_coords", mesh_coords_tv},
                  {"h", 10.0},
                  {"h_c:interface", 100.0},
                  {"fluid_temp", 20.0},
                  {"thermal_load", thermal_load_tv}
               };

               setInputs(res, inputs);

               double pert = uniform_rand(gen);
               // evaluate reverse mode sensitivity
               auto h_bar = vectorJacobianProduct(res, res_bar, "h_c:interface");
               auto drdp_rev = pert * h_bar;

               // now compute the finite-difference approximation...
               inputs["h_c:interface"] = 100.0 + delta * pert;
               mfem::Vector drdp_fd_p(state.space().GetTrueVSize());
               drdp_fd_p = 0.0;
               setInputs(res, inputs);
               evaluate(res, inputs, drdp_fd_p);

               inputs["h_c:interface"] = 100.0 - delta * pert;
               mfem::Vector drdp_fd_m(state.space().GetTrueVSize());
               drdp_fd_m = 0.0;
               setInputs(res, inputs);
               evaluate(res, inputs, drdp_fd_m);

               mfem::Vector scratch(state.space().GetTrueVSize());
               scratch = 0.0;
               scratch += drdp_fd_p;
               scratch -= drdp_fd_m;
               scratch /= (2 * delta);

               double drdp_fd = res_bar * scratch;

               std::cout << "drdp_rev: " << drdp_rev << " drdp_fd: " << drdp_fd << "\n";
               // REQUIRE(drdp_fwd == Approx(drdp_fd).margin(1e-8));
               REQUIRE(drdp_rev == Approx(drdp_fd).margin(1e-8));
            }
         }
      }
   }
}

TEST_CASE("ThermalResidual with interfaces sensitivity wrt fluid_temp:interface")
{
   std::default_random_engine gen;
   std::uniform_real_distribution<double> uniform_rand(-1.0,1.0);

   using namespace mfem;

   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   auto smesh = buildMesh(num_edge);

   mfem::ParMesh mesh(MPI_COMM_WORLD, smesh);

   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         mfem::L2_FECollection fec(p, dim);
         mfem::ParFiniteElementSpace fes(&mesh, &fec);

         std::map<std::string, mach::FiniteElementState> fields;
         fields.emplace("state", mach::FiniteElementState(mesh, fes, "state"));
         auto &state = fields.at("state");

         fields.emplace("dirichlet_bc", mach::FiniteElementState(mesh, fes, "dirichlet_bc"));

         auto &mesh_gf = *dynamic_cast<mfem::ParGridFunction *>(mesh.GetNodes());
         auto *mesh_fespace = mesh_gf.ParFESpace();
         /// create new state vector copying the mesh's fe space
         fields.emplace("mesh_coords",
                        mach::FiniteElementState(mesh, *mesh_fespace, "mesh_coords"));
         auto &mesh_coords = fields.at("mesh_coords");
         /// set the values of the new GF to those of the mesh's old nodes
         mesh_coords.gridFunc() = mesh_gf;
         /// tell the mesh to use this GF for its Nodes
         /// (and that it doesn't own it)
         mesh.NewNodes(mesh_coords.gridFunc(), false);


         mfem::Vector state_tv(state.space().GetTrueVSize());
         for (int i = 0; i < state_tv.Size(); ++i)
         {
            state_tv(i) = uniform_rand(gen);
         }

         mfem::Vector mesh_coords_tv(mesh_coords.space().GetTrueVSize());
         mesh_coords.setTrueVec(mesh_coords_tv);

         mfem::Vector thermal_load_tv(state.space().GetTrueVSize());
         for (int i = 0; i < thermal_load_tv.Size(); ++i)
         {
            thermal_load_tv(i) = uniform_rand(gen);
         }

         mfem::Vector res_bar(state.space().GetTrueVSize());
         for (int i = 0; i < res_bar.Size(); ++i)
         {
            res_bar(i) = uniform_rand(gen);
         }

         mfem::Vector pert_vec(mesh_coords.space().GetTrueVSize());
         for (int i = 0; i < pert_vec.Size(); ++i)
         {
            pert_vec(i) = uniform_rand(gen);
         }

         // res_bar = 0.0;
         // res_bar(0) = 1.0;
         // pert_vec = 0.0;
         // pert_vec(0) = 1.0;

         auto options = R"({
            "space-dis": {
               "basis-type": "dg",
               "degree": 1
            },
            "lin-prec": {
               "type": "hypreboomeramg",
               "printlevel": 0
            },
            "components": {
               "box1": {
                  "attrs": [1],
                  "material": {
                     "name": "box1"
                  }
               },
               "box2": {
                  "attrs": [2],
                  "material": {
                     "name": "box2"
                  }
               }
            },
            "interfaces": {
            },
            "bcs": {
               "convection": [1, 2],
               "outflux": [3, 4]
            }
         })"_json;

         options["space-dis"]["degree"] = p;

         for (int g = 0; g < 2; ++g)
         {
            DYNAMIC_SECTION("...with " << (g == 0 ? "thermal contact" : "internal convection") << " interface")
            {
               if (g == 0)
               {
                  options["interfaces"] = R"({
                     "thermal_contact_resistance": {
                        "interface": {
                           "attrs": [5],
                           "h_c": 10
                        }
                     }
                  })"_json;
               }
               else
               {
                  options["interfaces"] = R"({
                     "convection": {
                        "interface": {
                           "attrs": [5],
                           "h_c": 300,
                           "theta_f": 300
                        }
                     }
                  })"_json;
               }

               auto materials = R"({
                  "box1": {
                     "kappa": 1.0
                  },
                  "box2": {
                     "kappa": 1.0
                  }
               })"_json;

               mach::ThermalResidual res(fes, fields, options, materials);

               mach::MachInputs inputs{
                  {"state", state_tv},
                  {"mesh_coords", mesh_coords_tv},
                  {"h", 10.0},
                  {"fluid_temp", 20.0},
                  {"h_c:interface", 100.0},
                  {"fluid_temp:interface", 150.0},
                  {"thermal_load", thermal_load_tv}
               };

               setInputs(res, inputs);

               double pert = uniform_rand(gen);
               // evaluate reverse mode sensitivity
               auto h_bar = vectorJacobianProduct(res, res_bar, "fluid_temp:interface");
               auto drdp_rev = pert * h_bar;

               // now compute the finite-difference approximation...
               inputs["fluid_temp:interface"] = 150.0 + delta * pert;
               mfem::Vector drdp_fd_p(state.space().GetTrueVSize());
               drdp_fd_p = 0.0;
               setInputs(res, inputs);
               evaluate(res, inputs, drdp_fd_p);

               inputs["fluid_temp:interface"] = 150.0 - delta * pert;
               mfem::Vector drdp_fd_m(state.space().GetTrueVSize());
               drdp_fd_m = 0.0;
               setInputs(res, inputs);
               evaluate(res, inputs, drdp_fd_m);

               mfem::Vector scratch(state.space().GetTrueVSize());
               scratch = 0.0;
               scratch += drdp_fd_p;
               scratch -= drdp_fd_m;
               scratch /= (2 * delta);

               double drdp_fd = res_bar * scratch;

               std::cout << "drdp_rev: " << drdp_rev << " drdp_fd: " << drdp_fd << "\n";
               // REQUIRE(drdp_fwd == Approx(drdp_fd).margin(1e-8));
               REQUIRE(drdp_rev == Approx(drdp_fd).margin(1e-8));
            }
         }
      }
   }
}

mfem::Mesh buildMesh(int nxy, int internal_bdr_attr)
{
   auto mesh = mfem::Mesh::MakeCartesian2D(nxy, nxy, mfem::Element::TRIANGLE, true, 2.0, 1.0);
   // auto mesh = Mesh::MakeCartesian2D(nxy, nxy, Element::QUADRILATERAL, true, 2.0, 1.0);

   // assign element attributes to left and right sides
   for (int i = 0; i < mesh.GetNE(); ++i)
   {
      auto *elem = mesh.GetElement(i);

      mfem::Array<int> verts;
      elem->GetVertices(verts);

      bool left = true;
      for (int j = 0; j < verts.Size(); ++j)
      {
         auto *vtx = mesh.GetVertex(verts[j]);
         if (vtx[0] <= 1.0)
         {
            continue;
         }
         else
         {
            left = false;
         }
      }
      if (left)
      {
         elem->SetAttribute(1);
      }
      else
      {
         elem->SetAttribute(2);
      }
   }

   // assign boundary element attributes to left and right sides
   for (int i = 0; i < mesh.GetNBE(); ++i)
   {
      auto *elem = mesh.GetBdrElement(i);

      mfem::Array<int> verts;
      elem->GetVertices(verts);

      bool left = true;
      bool right = true;
      bool top = true;
      bool bottom = true;
      for (int j = 0; j < verts.Size(); ++j)
      {
         auto *vtx = mesh.GetVertex(verts[j]);
         left = left && abs(vtx[0] - 0.0) < 1e-12;
         right = right && abs(vtx[0] - 2.0) < 1e-12;
         top = top && abs(vtx[1] - 1.0) < 1e-12;
         bottom = bottom && abs(vtx[1] - 0.0) < 1e-12;
      }
      if (left)
      {
         elem->SetAttribute(1);
      }
      else if (right)
      {
         elem->SetAttribute(2);
      }
      else if (top)
      {
         elem->SetAttribute(3);
      }
      else if (bottom)
      {
         elem->SetAttribute(4);
      }
   }

   // add internal boundary elements
   for (int i = 0; i < mesh.GetNumFaces(); ++i)
   {
      int e1, e2;
      mesh.GetFaceElements(i, &e1, &e2);
      if (e1 >= 0 && e2 >= 0 && mesh.GetAttribute(e1) != mesh.GetAttribute(e2))
      {
         // This is the internal face between attributes.
         auto *new_elem = mesh.GetFace(i)->Duplicate(&mesh);
         new_elem->SetAttribute(internal_bdr_attr);
         mesh.AddBdrElement(new_elem);
      }
   }

   mesh.FinalizeTopology(); // Finalize to build relevant tables
   mesh.Finalize();
   mesh.SetAttributes();

   return mesh;
}