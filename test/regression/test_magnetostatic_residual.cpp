#include <random>

#include "catch.hpp"
#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "current_source_functions.hpp"
#include "mfem_common_integ.hpp"

#include "magnetostatic_residual.hpp"

/// Generate mesh 
/// \param[in] nxy - number of nodes in the x and y directions
mfem::Mesh buildMesh(int nxy = 2);

/// Simple nonlinear coefficient
class NonLinearCoefficient : public mach::StateCoefficient
{
public:
   double Eval(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               const double state) override
   {
      return 0.5*pow(state+1, -0.5);
   }

   double EvalStateDeriv(mfem::ElementTransformation &trans,
                         const mfem::IntegrationPoint &ip,
                         const double state) override
   {
      return -0.25*pow(state+1, -1.5);
   }

   double EvalState2ndDeriv(mfem::ElementTransformation &trans,
                            const mfem::IntegrationPoint &ip,
                            const double state) override
   {
      return 0.375*pow(state+1, -2.5);
   }

   void EvalRevDiff(const double Q_bar,
                    mfem::ElementTransformation &trans,
                    const mfem::IntegrationPoint &ip,
                    mfem::DenseMatrix &PointMat_Bar) override
   {}
};

// TEST_CASE("CurrentLoad sensitivity wrt mesh_coords")
// {
//    std::default_random_engine gen;
//    std::uniform_real_distribution<double> uniform_rand(-1.0,1.0);

//    using namespace mfem;

//    double delta = 1e-5;

//    // generate a 6 element mesh
//    int num_edge = 2;
//    auto smesh = buildMesh(num_edge);

//    mfem::ParMesh mesh(MPI_COMM_WORLD, smesh);

//    mesh.EnsureNodes();
//    const auto dim = mesh.SpaceDimension();

//    for (int p = 1; p <= 1; ++p)
//    {
//       DYNAMIC_SECTION( "...for degree p = " << p )
//       {
//          mfem::H1_FECollection fec(p, dim);
//          mfem::ParFiniteElementSpace fes(&mesh, &fec);

//          std::map<std::string, mach::FiniteElementState> fields;
//          fields.emplace("state", mach::FiniteElementState(mesh, fes, "state"));
//          auto &state = fields.at("state");

//          auto &mesh_gf = *dynamic_cast<mfem::ParGridFunction *>(mesh.GetNodes());
//          auto *mesh_fespace = mesh_gf.ParFESpace();
//          /// create new state vector copying the mesh's fe space
//          fields.emplace("mesh_coords",
//                         mach::FiniteElementState(mesh, *mesh_fespace, "mesh_coords"));
//          auto &mesh_coords = fields.at("mesh_coords");
//          /// set the values of the new GF to those of the mesh's old nodes
//          mesh_coords.gridFunc() = mesh_gf;
//          /// tell the mesh to use this GF for its Nodes
//          /// (and that it doesn't own it)
//          mesh.NewNodes(mesh_coords.gridFunc(), false);


//          mfem::Vector state_tv(state.space().GetTrueVSize());
//          for (int i = 0; i < state_tv.Size(); ++i)
//          {
//             state_tv(i) = uniform_rand(gen);
//          }

//          mfem::Vector mesh_coords_tv(mesh_coords.space().GetTrueVSize());
//          mesh_coords.setTrueVec(mesh_coords_tv);

//          mfem::Vector res_bar(state.space().GetTrueVSize());
//          for (int i = 0; i < res_bar.Size(); ++i)
//          {
//             res_bar(i) = uniform_rand(gen);
//          }

//          mfem::Vector pert_vec(mesh_coords.space().GetTrueVSize());
//          for (int i = 0; i < pert_vec.Size(); ++i)
//          {
//             pert_vec(i) = uniform_rand(gen);
//          }

//          // res_bar = 0.0;
//          // res_bar(0) = 1.0;
//          // pert_vec = 0.0;
//          // pert_vec(0) = 1.0;

//          auto options = R"({
//             "lin-prec": {
//                "type": "hypreboomeramg",
//                "printlevel": 0
//             },       
//             "components": {
//                "box1": {
//                   "attrs": [1],
//                   "material": {
//                      "name": "box1",
//                      "mu_r": 795774.7154594767
//                   }
//                },
//                "box2": {
//                   "attrs": [2],
//                   "material": {
//                      "name": "box2",
//                      "mu_r": 795774.7154594767
//                   }
//                }
//             },
//             "current": {
//                "box1": {
//                   "box1": [1]
//                },
//                "box2": {
//                   "box2": [2]
//                }
//             },
//             "bcs": {
//                "essential": "all"
//             }
//          })"_json;

//          auto materials = R"({
//             "box1": {
//                "mu_r": 795774.715
//             },
//             "box2": {
//                "mu_r": 795774.715
//             }
//          })"_json;

//          auto &stack = mach::getDiffStack();
         
//          mach::MachLinearForm load(fes, fields);

//          mach::CurrentDensityCoefficient2D current_coeff(stack, options["current"]);
//          // mfem::ConstantCoefficient current_coeff(1.0); 
//          // mfem::FunctionCoefficient current_coeff(
//          // [](const mfem::Vector &x)
//          // {
//          //    return exp(-pow(x(0),2));
//          // },
//          // [](const mfem::Vector &x, const double q_bar, mfem::Vector &x_bar)
//          // {
//          //    x_bar(0) -= q_bar * 2 * x(0) * exp(-pow(x(0),2));
//          // });

//          load.addDomainIntegrator(new mach::DomainLFIntegrator(current_coeff));

//          // mach::MagnetostaticResidual res(stack, fes, fields, options, materials, nu);

//          double current_density = 1.0;
//          mach::MachInputs inputs{
//             {"state", state_tv},
//             {"mesh_coords", mesh_coords_tv},
//             {"current_density:box1", current_density},
//             {"current_density:box2", -current_density}
//          };

//          setInputs(load, inputs);
//          setInputs(current_coeff, inputs);

//          mfem::Vector res_dot(state.space().GetTrueVSize());
//          res_dot = 0.0;
//          jacobianVectorProduct(load, pert_vec, "mesh_coords", res_dot);
//          double drdp_fwd = res_bar * res_dot;

//          mfem::Vector wrt_bar(mesh_coords.space().GetTrueVSize());
//          wrt_bar = 0.0;
//          vectorJacobianProduct(load, res_bar, "mesh_coords", wrt_bar);
//          double drdp_rev = wrt_bar * pert_vec;

//          // now compute the finite-difference approximation...
//          mesh_coords_tv.Add(delta, pert_vec);
//          mfem::Vector drdp_fd_p(state.space().GetTrueVSize());
//          drdp_fd_p = 0.0;
//          setInputs(load, inputs);
//          setInputs(current_coeff, inputs);
//          addLoad(load, drdp_fd_p);

//          mesh_coords_tv.Add(-2 * delta, pert_vec);
//          mfem::Vector drdp_fd_m(state.space().GetTrueVSize());
//          drdp_fd_m = 0.0;
//          setInputs(load, inputs);
//          setInputs(current_coeff, inputs);
//          addLoad(load, drdp_fd_m);

//          mfem::Vector scratch(state.space().GetTrueVSize());
//          scratch = 0.0;
//          scratch += drdp_fd_p;
//          scratch -= drdp_fd_m;
//          scratch /= (2 * delta);

//          double drdp_fd = res_bar * scratch;

//          std::cout << "drdp_rev: " << drdp_rev << " drdp_fd: " << drdp_fd << "\n";
//          // REQUIRE(drdp_fwd == Approx(drdp_fd).margin(1e-8));
//          REQUIRE(drdp_rev == Approx(drdp_fd).margin(1e-8));
//          mesh_coords_tv.Add(delta, pert_vec);
//       }
//    }
// }

TEST_CASE("MagnetostaticResidual sensitivity wrt current_density")
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

   NonLinearCoefficient nu;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         mfem::H1_FECollection fec(p, dim);
         mfem::ParFiniteElementSpace fes(&mesh, &fec);

         std::map<std::string, mach::FiniteElementState> fields;
         fields.emplace("state", mach::FiniteElementState(mesh, fes, "state"));
         auto &state = fields.at("state");

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

         fields.emplace("temperature",
                        mach::FiniteElementState(mesh, {.order=p, .name="temperature"}));
         auto &temp = fields.at("temperature");

         mfem::Vector state_tv(state.space().GetTrueVSize());
         for (int i = 0; i < state_tv.Size(); ++i)
         {
            state_tv(i) = uniform_rand(gen);
         }

         mfem::Vector mesh_coords_tv(mesh_coords.space().GetTrueVSize());
         mesh_coords.setTrueVec(mesh_coords_tv);

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         for (int i = 0; i < temp_tv.Size(); ++i)
         {
            temp_tv(i) = uniform_rand(gen);
         }

         mfem::Vector res_bar(state.space().GetTrueVSize());
         for (int i = 0; i < res_bar.Size(); ++i)
         {
            res_bar(i) = uniform_rand(gen);
         }

         auto options = R"({
            "lin-prec": {
               "type": "hypreboomeramg",
               "printlevel": 0
            },       
            "components": {
               "box1": {
                  "attrs": [1],
                  "material": {
                     "name": "box1",
                     "mu_r": 795774.7154594767
                  }
               },
               "box2": {
                  "attrs": [2],
                  "material": {
                     "name": "box2",
                     "mu_r": 795774.7154594767
                  }
               }
            },
            "current": {
               "box1": {
                  "box1": [1]
               },
               "box2": {
                  "box2": [2]
               }
            },
            "bcs": {
               "essential": "all"
            }
         })"_json;

         auto materials = R"({
            "box1": {
               "mu_r": 795774.715
            },
            "box2": {
               "mu_r": 795774.715
            }
         })"_json;

         auto &stack = mach::getDiffStack();
         mach::MagnetostaticResidual res(stack, fes, fields, options, materials, nu);

         double current_density = 1.0;
         mach::MachInputs inputs{
            {"state", state_tv},
            {"mesh_coords", mesh_coords_tv},
            {"current_density:box1", current_density},
            {"current_density:box2", -current_density},
            {"temperature", temp_tv}
         };

         double pert = uniform_rand(gen);
         mfem::Vector pert_vec(&pert, 1);

         setInputs(res, inputs);

         mfem::Vector res_dot(state.space().GetTrueVSize());
         res_dot = 0.0;
         jacobianVectorProduct(res, pert_vec, "current_density:box1", res_dot);
         double drdp_fwd = res_bar * res_dot;

         double drdp_rev = vectorJacobianProduct(res, res_bar, "current_density:box1") * pert;

         // now compute the finite-difference approximation...
         inputs["current_density:box1"] = current_density + pert * delta;
         mfem::Vector drdp_fd_p(state.space().GetTrueVSize());
         drdp_fd_p = 0.0;
         evaluate(res, inputs, drdp_fd_p);

         inputs["current_density:box1"] = current_density - pert * delta;
         mfem::Vector drdp_fd_m(state.space().GetTrueVSize());
         drdp_fd_m = 0.0;
         evaluate(res, inputs, drdp_fd_m);

         mfem::Vector scratch(state.space().GetTrueVSize());
         scratch = 0.0;
         scratch += drdp_fd_p;
         scratch -= drdp_fd_m;
         scratch /= (2 * delta);

         double drdp_fd = res_bar * scratch;

         inputs["current_density:box1"] = current_density;
         REQUIRE(drdp_fwd == Approx(drdp_fd).margin(1e-8));
         REQUIRE(drdp_rev == Approx(drdp_fd).margin(1e-8));

         res_dot = 0.0;
         jacobianVectorProduct(res, pert_vec, "current_density:box2", res_dot);
         drdp_fwd = res_bar * res_dot;

         drdp_rev = vectorJacobianProduct(res, res_bar, "current_density:box2") * pert;

         // now compute the finite-difference approximation...
         inputs["current_density:box2"] = -current_density + pert * delta;
         drdp_fd_p = 0.0;
         evaluate(res, inputs, drdp_fd_p);

         inputs["current_density:box2"] = -current_density - pert * delta;
         drdp_fd_m = 0.0;
         evaluate(res, inputs, drdp_fd_m);

         scratch = 0.0;
         scratch += drdp_fd_p;
         scratch -= drdp_fd_m;
         scratch /= (2 * delta);

         drdp_fd = res_bar * scratch;

         REQUIRE(drdp_fwd == Approx(drdp_fd).margin(1e-8));
         REQUIRE(drdp_rev == Approx(drdp_fd).margin(1e-8));
      }
   }
}

TEST_CASE("MagnetostaticResidual sensitivity wrt mesh_coords")
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

   NonLinearCoefficient nu;

   for (int p = 1; p <= 1; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         mfem::H1_FECollection fec(p, dim);
         mfem::ParFiniteElementSpace fes(&mesh, &fec);

         std::map<std::string, mach::FiniteElementState> fields;
         fields.emplace("state", mach::FiniteElementState(mesh, fes, "state"));
         auto &state = fields.at("state");

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

         fields.emplace("temperature",
                        mach::FiniteElementState(mesh, {.order=p, .name="temperature"}));
         auto &temp = fields.at("temperature");

         mfem::Vector state_tv(state.space().GetTrueVSize());
         for (int i = 0; i < state_tv.Size(); ++i)
         {
            state_tv(i) = uniform_rand(gen);
         }

         mfem::Vector mesh_coords_tv(mesh_coords.space().GetTrueVSize());
         mesh_coords.setTrueVec(mesh_coords_tv);

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         for (int i = 0; i < temp_tv.Size(); ++i)
         {
            temp_tv(i) = uniform_rand(gen);
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
            "lin-prec": {
               "type": "hypreboomeramg",
               "printlevel": 0
            },       
            "components": {
               "box1": {
                  "attrs": [1],
                  "material": {
                     "name": "box1",
                     "mu_r": 795774.7154594767
                  }
               },
               "box2": {
                  "attrs": [2],
                  "material": {
                     "name": "box2",
                     "mu_r": 795774.7154594767
                  }
               }
            },
            "bcs": {
               "essential": "all"
            }
         })"_json;

         auto materials = R"({
            "box1": {
               "mu_r": 795774.715
            },
            "box2": {
               "mu_r": 795774.715
            },
            "Nd2Fe14B": {
               "B_r": 1.2
            }
         })"_json;

         auto &stack = mach::getDiffStack();
         mach::MagnetostaticResidual res(stack, fes, fields, options, materials, nu);

         double current_density = 1.0;
         mach::MachInputs inputs{
            {"state", state_tv},
            {"mesh_coords", mesh_coords_tv},
            {"current_density:box1", current_density},
            {"current_density:box2", -current_density},
            {"temperature", temp_tv}
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

mfem::Mesh buildMesh(int nxy)
{
   // generate a simple tet mesh
   auto mesh = mfem::Mesh::MakeCartesian2D(nxy, nxy,
                                           mfem::Element::TRIANGLE);

   // assign attributes to top and bottom sides
   for (int i = 0; i < mesh.GetNE(); ++i)
   {
      auto *elem = mesh.GetElement(i);

      mfem::Array<int> verts;
      elem->GetVertices(verts);

      bool below = true;
      for (int i = 0; i < verts.Size(); ++i)
      {
         auto *vtx = mesh.GetVertex(verts[i]);
         // std::cout << "mesh vtx: " << vtx[0] << ", " << vtx[1] << "\n";
         if (vtx[1] <= 0.5)
         {
            below = below;
         }
         else
         {
            below = false;
         }
      }
      if (below)
      {
         elem->SetAttribute(1);
      }
      else
      {
         elem->SetAttribute(2);
      }
   }
   mesh.SetAttributes();

   return mesh;
}