#include <iostream>
#include <memory>
#include <string>
#include <variant>

#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "coefficient.hpp"
#include "common_outputs.hpp"
#include "data_logging.hpp"
#include "electromag_integ.hpp"
#include "electromag_outputs.hpp"
#include "finite_element_state.hpp"
#include "functional_output.hpp"
#include "l2_transfer_operator.hpp"
#include "pde_solver.hpp"

#include "magnetostatic.hpp"

namespace
{
// miso::MeshDependentCoefficient constructSigma(const nlohmann::json &options,
//                                               const nlohmann::json
//                                               &materials)
// {
//    miso::MeshDependentCoefficient sigma;
//    /// loop over all components, construct conductivity for each
//    for (auto &component : options["components"])
//    {
//       int attr = component.value("attr", -1);

//       const auto &material = component["material"].get<std::string>();
//       double sigma_val = materials[material].value("sigma", 0.0);

//       if (-1 != attr)
//       {
//          auto coeff = std::make_unique<mfem::ConstantCoefficient>(sigma_val);
//          sigma.addCoefficient(attr, move(coeff));
//       }
//       else
//       {
//          for (auto &attribute : component["attrs"])
//          {
//             auto coeff =
//             std::make_unique<mfem::ConstantCoefficient>(sigma_val);
//             sigma.addCoefficient(attribute, move(coeff));
//          }
//       }
//    }
//    return sigma;
// }

std::vector<int> getCurrentAttributes(nlohmann::json &options)
{
   std::vector<int> attributes;
   for (const auto &group : options["current"])
   {
      for (const auto &source : group)
      {
         auto attrs = source.get<std::vector<int>>();
         attributes.insert(attributes.end(), attrs.begin(), attrs.end());
      }
   }
   return attributes;
}

}  // anonymous namespace

namespace miso
{
MagnetostaticSolver::MagnetostaticSolver(MPI_Comm comm,
                                         const nlohmann::json &solver_options,
                                         std::unique_ptr<mfem::Mesh> smesh)
 : PDESolver(comm, solver_options, 1, std::move(smesh)),
   nu(options, materials),
   // rho(constructMaterialCoefficient("rho", options["components"],
   // materials)),
   sigma(options, materials)
// mag_coeff(diff_stack, options["magnets"], materials, 2),
// B_knee(options, materials)
{
   std::cout << "state size: " << state().gridFunc().Size() << "\n";

   options["time-dis"]["type"] = "steady";

   if (auto temp_iter = fields.find("temperature"); temp_iter == fields.end())
   {
      fields.emplace(
          "temperature",
          FiniteElementState(
              mesh(), FiniteElementVector::Options{.name = "temperature"}));
   }
   auto &temp = fields.at("temperature");
   temp.gridFunc() = 273.15;

   spatial_res = std::make_unique<MISOResidual>(MagnetostaticResidual(
       diff_stack, fes(), fields, options, materials, nu));
   miso::setOptions(*spatial_res, options);

   auto *prec = getPreconditioner(*spatial_res);
   auto lin_solver_opts = options["lin-solver"];
   linear_solver = miso::constructLinearSolver(comm, lin_solver_opts, prec);
   auto nonlin_solver_opts = options["nonlin-solver"];
   nonlinear_solver =
       miso::constructNonlinearSolver(comm, nonlin_solver_opts, *linear_solver);
   nonlinear_solver->SetOperator(*spatial_res);

   auto state_degree =
       AbstractSolver2::options["space-dis"]["degree"].get<int>();
   nlohmann::json dg_field_options{{"degree", state_degree},
                                   {"basis-type", "DG"}};
   fields.emplace("flux_magnitude",
                  FiniteElementState(mesh(), dg_field_options));
   fields.emplace("peak_flux", FiniteElementState(mesh(), dg_field_options));

   auto paraview = options["paraview"]["log"].get<bool>();
   if (paraview)
   {
      auto paraview_dir_name = options["paraview"]["directory"];
      miso::ParaViewLogger logger(paraview_dir_name, &mesh());
      const auto &logged_fields = options["paraview"]["fields"];
      for (const auto &field : logged_fields)
      {
         if (fields.count(field) > 0)
         {
            logger.registerField(field, fields.at(field).gridFunc());
         }
         else if (duals.count(field) > 0)
         {
            logger.registerField(field,
                                 dynamic_cast<mfem::ParGridFunction &>(
                                     duals.at(field).localVec()));
         }
         else
         {
            std::cerr << "attempting to log unknown field \"" << field
                      << "\"!\n";
         }
      }
      addLogger(std::move(logger), {});
   }
   // miso::ParaViewLogger paraview("magnetostatic", &mesh());
   // paraview.registerField("state", fields.at("state").gridFunc());
   // paraview.registerField("adjoint", fields.at("adjoint").gridFunc());
   // paraview.registerField(
   //     "residual",
   //     dynamic_cast<mfem::ParGridFunction
   //     &>(duals.at("residual").localVec()));

   // paraview.registerField("peak_flux", fields.at("peak_flux").gridFunc());

   // const auto &temp_field_iter = fields.find("temperature");
   // if (temp_field_iter != fields.end())
   // {
   //    auto &temp_field = temp_field_iter->second;
   //    paraview.registerField("temperature", temp_field.gridFunc());
   // }

   // // Adding a field for the pm demag constraint field
   // paraview.registerField("pm_demag_field",
   // fields.at("pm_demag_field").gridFunc());

   // addLogger(std::move(paraview), {});
}

void MagnetostaticSolver::addOutput(const std::string &fun,
                                    const nlohmann::json &options)
{
   /// if fun starts with prefix "energy"
   if (fun.rfind("energy", 0) == 0)
   {
      FunctionalOutput out(fes(), fields);
      if (options.contains("attributes"))
      {
         auto attributes = options["attributes"].get<std::vector<int>>();
         out.addOutputDomainIntegrator(new MagneticEnergyIntegrator(nu),
                                       attributes);
      }
      else
      {
         out.addOutputDomainIntegrator(new MagneticEnergyIntegrator(nu));
      }
      outputs.emplace(fun, std::move(out));
   }
   else if (fun.rfind("force", 0) == 0)
   {
      /// create displacement field V that uses the same FES as the mesh
      auto &mesh_fes = fields.at("mesh_coords").space();
      fields.emplace(std::piecewise_construct,
                     std::make_tuple("v" + fun),
                     std::forward_as_tuple(mesh(), mesh_fes));

      ForceFunctional out(fes(), fields, options, nu);
      outputs.emplace(fun, std::move(out));
   }
   else if (fun.rfind("torque", 0) == 0)
   {
      /// create displacement field V that uses the same FES as the mesh
      auto &mesh_fes = fields.at("mesh_coords").space();
      fields.emplace(std::piecewise_construct,
                     std::make_tuple("v" + fun),
                     std::forward_as_tuple(mesh(), mesh_fes));

      TorqueFunctional out(fes(), fields, options, nu);
      outputs.emplace(fun, std::move(out));
   }
   else if (fun.rfind("flux_density", 0) == 0)
   {
      auto state_degree =
          AbstractSolver2::options["space-dis"]["degree"].get<int>();
      nlohmann::json dg_field_options{{"degree", state_degree},
                                      {"basis-type", "DG"}};
      fields.emplace(std::piecewise_construct,
                     std::forward_as_tuple(fun),
                     std::forward_as_tuple(
                         mesh(), dg_field_options, mesh().SpaceDimension()));

      // auto &[logger, logger_opts] = loggers.back();
      // if (std::holds_alternative<miso::ParaViewLogger>(logger))
      // {
      //    auto &paraview = std::get<miso::ParaViewLogger>(logger);
      //    paraview.registerField("flux_density",
      //                           fields.at("flux_density").gridFunc());
      // }

      auto &dg_field = fields.at(fun);
      L2CurlProjection out(state(), fields.at("mesh_coords"), dg_field);
      outputs.emplace(fun, std::move(out));
   }
   else if (fun.rfind("flux_magnitude", 0) == 0)
   {
      // auto state_degree =
      //     AbstractSolver2::options["space-dis"]["degree"].get<int>();
      // nlohmann::json dg_field_options{{"degree", state_degree},
      //                                 {"basis-type", "DG"}};
      // fields.emplace(std::piecewise_construct,
      //                std::forward_as_tuple(fun),
      //                std::forward_as_tuple(mesh(), dg_field_options));

      // miso::ParaViewLogger paraview("flux_magnitude", &mesh());
      // paraview.registerField("flux_magnitude",
      // fields.at("flux_magnitude").gridFunc());
      // addLogger(std::move(paraview), {});

      // auto &[logger, logger_opts] = loggers.back();
      // if (std::holds_alternative<miso::ParaViewLogger>(logger))
      // {
      //    auto &paraview = std::get<miso::ParaViewLogger>(logger);
      //    paraview.registerField("flux_magnitude",
      //                           fields.at("flux_magnitude").gridFunc());
      // }

      auto &dg_field = fields.at(fun);
      L2CurlMagnitudeProjection out(
          state(), fields.at("mesh_coords"), dg_field);
      outputs.emplace(fun, std::move(out));
   }
   else if (fun.rfind("average_flux_magnitude", 0) == 0)
   {
      AverageMagnitudeCurlState out(fes(), fields, options);
      outputs.emplace(fun, std::move(out));
   }
   else if (fun.rfind("max_flux_magnitude", 0) == 0)
   {
      IECurlMagnitudeAggregateFunctional out(fes(), fields, options);
      outputs.emplace(fun, std::move(out));
   }
   else if (fun.rfind("max_state", 0) == 0)
   {
      // auto state_degree =
      //     AbstractSolver2::options["space-dis"]["degree"].get<int>();
      // nlohmann::json dg_field_options{{"degree", state_degree},
      //                                 {"basis-type", "DG"}};
      // fields.emplace(std::piecewise_construct,
      //                std::forward_as_tuple("peak_flux"),
      //                std::forward_as_tuple(mesh(), dg_field_options));

      mfem::ParFiniteElementSpace *fes = nullptr;
      if (options.contains("state"))
      {
         auto field_name = options["state"].get<std::string>();
         fes = &fields.at(field_name).space();
      }
      else
      {
         fes = &PDESolver::fes();
      }
      IEAggregateFunctional out(*fes, fields, options);
      outputs.emplace(fun, std::move(out));
   }
   else if (fun.rfind("dc_loss", 0) == 0)
   {
      auto dc_loss_options = options;
      dc_loss_options["attributes"] =
          getCurrentAttributes(AbstractSolver2::options);
      DCLossFunctional out(fields, sigma, dc_loss_options);
      outputs.emplace(fun, std::move(out));
   }
   else if (fun.rfind("ac_loss", 0) == 0)
   {
      // auto state_degree =
      //     AbstractSolver2::options["space-dis"]["degree"].get<int>();
      // nlohmann::json dg_field_options{{"degree", state_degree},
      //                                 {"basis-type", "DG"}};
      // fields.emplace(std::piecewise_construct,
      //                std::forward_as_tuple("peak_flux"),
      //                std::forward_as_tuple(mesh(), dg_field_options));

      // auto &[logger, logger_opts] = loggers.back();
      // if (std::holds_alternative<miso::ParaViewLogger>(logger))
      // {
      //    std::cout << "adding peak flux to logger!\n";
      //    auto &paraview = std::get<miso::ParaViewLogger>(logger);
      //    paraview.registerField("peak_flux",
      //    fields.at("peak_flux").gridFunc());
      // }

      auto ac_loss_options = options;
      ac_loss_options["attributes"] =
          getCurrentAttributes(AbstractSolver2::options);
      ACLossFunctional out(fields, sigma, ac_loss_options);
      outputs.emplace(fun, std::move(out));
   }
   else if (fun.rfind("core_loss", 0) == 0)
   {
      // auto state_degree =
      //     AbstractSolver2::options["space-dis"]["degree"].get<int>();
      // nlohmann::json dg_field_options{{"degree", state_degree},
      //                                 {"basis-type", "DG"}};
      // fields.emplace(std::piecewise_construct,
      //                std::forward_as_tuple("peak_flux"),
      //                std::forward_as_tuple(mesh(), dg_field_options));

      // auto &[logger, logger_opts] = loggers.back();
      // if (std::holds_alternative<miso::ParaViewLogger>(logger))
      // {
      //    std::cout << "adding peak flux to logger!\n";
      //    auto &paraview = std::get<miso::ParaViewLogger>(logger);
      //    paraview.registerField("peak_flux",
      //    fields.at("peak_flux").gridFunc());
      // }

      CoreLossFunctional out(
          fields, AbstractSolver2::options["components"], materials, options);
      outputs.emplace(fun, std::move(out));
   }
   else if (fun.rfind("mass", 0) == 0)
   {
      MassFunctional out(
          fields, AbstractSolver2::options["components"], materials, options);
      outputs.emplace(fun, std::move(out));
   }
   else if (fun.rfind("volume", 0) == 0)
   {
      VolumeFunctional out(fields, options);
      outputs.emplace(fun, std::move(out));
   }
   else if (fun.rfind("heat_source", 0) == 0)
   {
      // auto state_degree =
      //     AbstractSolver2::options["space-dis"]["degree"].get<int>();
      // nlohmann::json dg_field_options{{"degree", state_degree},
      //                                 {"basis-type", "DG"}};
      // fields.emplace(std::piecewise_construct,
      //                std::forward_as_tuple("peak_flux"),
      //                std::forward_as_tuple(mesh(), dg_field_options));

      // auto &[logger, logger_opts] = loggers.back();
      // if (std::holds_alternative<miso::ParaViewLogger>(logger))
      // {
      //    std::cout << "adding peak flux to logger!\n";
      //    auto &paraview = std::get<miso::ParaViewLogger>(logger);
      //    paraview.registerField("peak_flux",
      //    fields.at("peak_flux").gridFunc());
      // }

      EMHeatSourceOutput out(fields,
                             sigma,
                             AbstractSolver2::options["components"],
                             materials,
                             options);
      outputs.emplace(fun, std::move(out));
   }
   // else if (fun.rfind("pm_demag", 0) == 0)
   // {
   //    // Make the pm demag constraint field
   //    auto state_degree =
   //        AbstractSolver2::options["space-dis"]["degree"].get<int>();
   //    nlohmann::json dg_field_options{{"degree", state_degree},
   //                                    {"basis-type", "DG"}};
   //    fields.emplace(std::piecewise_construct,
   //                   std::forward_as_tuple(fun),
   //                   std::forward_as_tuple(mesh(), dg_field_options));

   //    ///TODO: Just need regular flux density, which is assumed to be the
   //    elfun passed in to integrator. Look into making this change
   //    // Adding the peak flux field so can visualize it
   //    fields.emplace(std::piecewise_construct,
   //                   std::forward_as_tuple("peak_flux"),
   //                   std::forward_as_tuple(mesh(), dg_field_options));
   //    /*
   //    auto state_degree =
   //        AbstractSolver2::options["space-dis"]["degree"].get<int>();
   //    nlohmann::json dg_field_options{{"degree", state_degree},
   //                                    {"basis-type", "DG"}};
   //    fields.emplace(std::piecewise_construct,
   //                   std::forward_as_tuple("peak_flux"),
   //                   std::forward_as_tuple(mesh(), dg_field_options));
   //    */

   //    // Adding the pm demag field so can visualize it
   //    fields.emplace(std::piecewise_construct,
   //                   std::forward_as_tuple("pm_demag_field"),
   //                   std::forward_as_tuple(mesh(), dg_field_options));

   //    // Add the pm demag constraint to paraview
   //    auto &[logger, logger_opts] = loggers.back();
   //    if (std::holds_alternative<miso::ParaViewLogger>(logger))
   //    {
   //       auto &paraview = std::get<miso::ParaViewLogger>(logger);
   //       paraview.registerField("pm_demag",
   //                              fields.at("pm_demag").gridFunc());
   //       paraview.registerField("pm_demag_field",
   //                              fields.at("pm_demag_field").gridFunc());
   //    }

   //    ///TODO: If needed, emplace the temperature field as was done for heat
   //    source outputs

   //    // std::cout << "magnetostatic.cpp, pre PMDemagOutput call\n";
   //    PMDemagOutput out(
   //        fields, AbstractSolver2::options["components"], materials,
   //        options);
   //    // std::cout << "magnetostatic.cpp, post PMDemagOutput call\n";
   //    outputs.emplace(fun, std::move(out));
   //    // std::cout << "magnetostatic.cpp, post output emplace\n";
   // }
   // else if (fun.rfind("demag_proximity", 0) == 0)
   // {
   //    auto state_degree =
   //        AbstractSolver2::options["space-dis"]["degree"].get<int>();
   //    nlohmann::json dg_field_options{{"degree", state_degree},
   //                                    {"basis-type", "DG"}};
   //    fields.emplace(std::piecewise_construct,
   //                   std::forward_as_tuple("flux_density"),
   //                   std::forward_as_tuple(
   //                       mesh(), dg_field_options, mesh().SpaceDimension()));

   //    auto &[logger, logger_opts] = loggers.back();
   //    if (std::holds_alternative<miso::ParaViewLogger>(logger))
   //    {
   //       auto &paraview = std::get<miso::ParaViewLogger>(logger);
   //       paraview.registerField("flux_density",
   //                              fields.at("flux_density").gridFunc());
   //    }

   //    /* Pseudo-code
   //    // Get the B field from L2CurlProjection then turn it into a grid
   //    function
   //    // Get the temperature field and turn it into grid function

   //    // For a pointwise evaluation (integrator logic):
   //    Get trans, ip, etc.
   //    double temperature -> shape functions dotted with temp_elfun
   //    mfem::Vector B -> shape functions dotted with B_elfun
   //    double B_demag = B_knee.Eval(trans, ip, temperature);
   //    VectorStateCoefficient M;
   //    magnetization.Eval(M, trans, ip, temperature)
   //    double demag_prox = B_demag - (B * M)/M.Norml2();

   //    // The above is for one single point in space
   //    */

   //    // Obtain the flux density field
   //    auto &dg_B_field = fields.at("flux_density");
   //    L2CurlProjection out(state(), fields.at("mesh_coords"), dg_B_field);
   //    outputs.emplace(fun, std::move(out));
   // }
   else
   {
      throw MISOException("Output with name " + fun +
                          " not supported by "
                          "MagnetostaticSolver!\n");
   }
}

void MagnetostaticSolver::derivedPDETerminalHook(int iter,
                                                 double t_final,
                                                 const mfem::Vector &state)
{
   // work.SetSize(state.Size());
   // calcResidual(state, work);
   // res_vec().distributeSharedDofs(work);
}

}  // namespace miso

// using namespace std;
// using namespace mfem;

// using adept::adouble;

// namespace
// {
// /// permeability of free space
// constexpr double mu_0 = 4e-7 * M_PI;

// std::unique_ptr<mfem::Coefficient> constructReluctivityCoeff(
//     nlohmann::json &component,
//     nlohmann::json &materials)
// {
//    std::unique_ptr<mfem::Coefficient> temp_coeff;
//    std::string material = component["material"].get<std::string>();
//    if (!component["linear"].get<bool>())
//    {
//       std::unique_ptr<mfem::Coefficient> lin_coeff;
//       std::unique_ptr<miso::StateCoefficient> nonlin_coeff;

//       auto mu_r = materials[material]["mu_r"].get<double>();
//       lin_coeff =
//           std::make_unique<mfem::ConstantCoefficient>(1.0 / (mu_r *
//           mu_0));

//       // if (material == "team13")
//       // {
//       //    nonlin_coeff.reset(new miso::team13ReluctivityCoefficient());
//       // }
//       // else
//       // {
//       auto b = materials[material]["B"].get<std::vector<double>>();
//       auto h = materials[material]["H"].get<std::vector<double>>();
//       nonlin_coeff =
//           std::make_unique<miso::NonlinearReluctivityCoefficient>(b, h);
//       // }

//       temp_coeff =
//       std::make_unique<miso::ParameterContinuationCoefficient>(
//           move(lin_coeff), move(nonlin_coeff));

//       // if (material == "team13")
//       // {
//       //    temp_coeff.reset(new miso::team13ReluctivityCoefficient());
//       // }
//       // else
//       // {
//       // auto b = materials[material]["B"].get<std::vector<double>>();
//       // auto h = materials[material]["H"].get<std::vector<double>>();
//       // temp_coeff.reset(new miso::ReluctivityCoefficient(b, h));
//       // }
//    }
//    else
//    {
//       auto mu_r = materials[material]["mu_r"].get<double>();
//       temp_coeff =
//           std::make_unique<mfem::ConstantCoefficient>(1.0 / (mu_r *
//           mu_0));
//       // std::cout << "new coeff with mu_r: " << mu_r << "\n";
//    }
//    return temp_coeff;
// }

// // // define the random-number generator; uniform between -1 and 1
// // static std::default_random_engine gen;
// // static std::uniform_real_distribution<double> uniform_rand(-1.0,1.0);

// // // double randState(const mfem::Vector &x)
// // // {
// // //    return 2.0 * uniform_rand(gen) - 1.0;
// // // }

// // void randState(const mfem::Vector &x, mfem::Vector &u)
// // {
// //    // std::cout << "u size: " << u.Size() << std::endl;
// //    for (int i = 0; i < u.Size(); ++i)
// //    {
// //       // std::cout << i << std::endl;
// //       u(i) = uniform_rand(gen);
// //    }
// // }

// template <typename xdouble = double>
// void phase_a_current(const xdouble &n_slots,
//                      const xdouble &stack_length,
//                      const xdouble *x,
//                      xdouble *J)
// {
//    J[0] = 0.0;
//    J[1] = 0.0;
//    J[2] = 0.0;

//    xdouble zb = -stack_length / 2;  // bottom of stator
//    xdouble zt = stack_length / 2;   // top of stator

//    // compute theta from x and y
//    xdouble tha = atan2(x[1], x[0]);
//    xdouble thw = 2 * M_PI / n_slots;  // total angle of slot

//    // check which winding we're in
//    xdouble w = round(tha / thw);  // current slot
//    xdouble th = tha - w * thw;

//    // check if we're in the stator body
//    if (x[2] >= zb && x[2] <= zt)
//    {
//       // check if we're in left or right half
//       if (th > 0)
//       {
//          J[2] = -1;  // set to 1 for now, and direction depends on current
//                      // direction
//       }
//       if (th < 0)
//       {
//          J[2] = 1;
//       }
//    }
//    else  // outside of the stator body, check if above or below
//    {
//       // 'subtract' z position to 0 depending on if above or below
//       xdouble rx[] = {x[0], x[1], x[2]};
//       if (x[2] > zt)
//       {
//          rx[2] -= zt;
//       }
//       if (x[2] < zb)
//       {
//          rx[2] -= zb;
//       }

//       // draw top rotation axis
//       xdouble ax[] = {0.0, 0.0, 0.0};
//       ax[0] = cos(w * thw);
//       ax[1] = sin(w * thw);

//       // take x cross ax, normalize
//       J[0] = rx[1] * ax[2] - rx[2] * ax[1];
//       J[1] = rx[2] * ax[0] - rx[0] * ax[2];
//       J[2] = rx[0] * ax[1] - rx[1] * ax[0];
//       xdouble norm_J = sqrt(J[0] * J[0] + J[1] * J[1] + J[2] * J[2]);
//       J[0] /= norm_J;
//       J[1] /= norm_J;
//       J[2] /= norm_J;
//    }
//    J[0] *= -1.0;
//    J[1] *= -1.0;
//    J[2] *= -1.0;

//    auto jmag = sqrt(J[0] * J[0] + J[1] * J[1] + J[2] * J[2]);
//    if (abs(jmag - 1.0) >= 1e-14)
//    {
//       std::cout << "J mag: " << sqrt(J[0] * J[0] + J[1] * J[1] + J[2] *
//       J[2])
//                 << "\n";
//    }
// }

// template <typename xdouble = double>
// void phase_b_current(const xdouble &n_slots,
//                      const xdouble &stack_length,
//                      const xdouble *x,
//                      xdouble *J)
// {
//    J[0] = 0.0;
//    J[1] = 0.0;
//    J[2] = 0.0;

//    xdouble zb = -stack_length / 2;  // bottom of stator
//    xdouble zt = stack_length / 2;   // top of stator

//    // compute theta from x and y
//    xdouble tha = atan2(x[1], x[0]);
//    xdouble thw = 2 * M_PI / n_slots;  // total angle of slot

//    // check which winding we're in
//    xdouble w = round(tha / thw);  // current slot
//    xdouble th = tha - w * thw;

//    // check if we're in the stator body
//    if (x[2] >= zb && x[2] <= zt)
//    {
//       // check if we're in left or right half
//       if (th > 0)
//       {
//          J[2] = -1;  // set to 1 for now, and direction depends on current
//                      // direction
//       }
//       if (th < 0)
//       {
//          J[2] = 1;
//       }
//    }
//    else  // outside of the stator body, check if above or below
//    {
//       // 'subtract' z position to 0 depending on if above or below
//       xdouble rx[] = {x[0], x[1], x[2]};
//       if (x[2] > zt)
//       {
//          rx[2] -= zt;
//       }
//       if (x[2] < zb)
//       {
//          rx[2] -= zb;
//       }

//       // draw top rotation axis
//       xdouble ax[] = {0.0, 0.0, 0.0};
//       ax[0] = cos(w * thw);
//       ax[1] = sin(w * thw);

//       // take x cross ax, normalize
//       J[0] = rx[1] * ax[2] - rx[2] * ax[1];
//       J[1] = rx[2] * ax[0] - rx[0] * ax[2];
//       J[2] = rx[0] * ax[1] - rx[1] * ax[0];
//       xdouble norm_J = sqrt(J[0] * J[0] + J[1] * J[1] + J[2] * J[2]);
//       J[0] /= norm_J;
//       J[1] /= norm_J;
//       J[2] /= norm_J;
//    }
// }

// template <typename xdouble = double>
// void phase_c_current(const xdouble &n_slots,
//                      const xdouble &stack_length,
//                      const xdouble *x,
//                      xdouble *J)
// {
//    J[0] = 0.0;
//    J[1] = 0.0;
//    J[2] = 0.0;
// }

// template <typename xdouble = double>
// void north_magnetization(const xdouble &remnant_flux,
//                          const xdouble *x,
//                          xdouble *M)
// {
//    xdouble r[] = {0.0, 0.0, 0.0};
//    r[0] = x[0];
//    r[1] = x[1];
//    xdouble norm_r = sqrt(r[0] * r[0] + r[1] * r[1]);
//    M[0] = r[0] * remnant_flux / norm_r;
//    M[1] = r[1] * remnant_flux / norm_r;
//    M[2] = 0.0;
// }

// template <typename xdouble = double>
// void south_magnetization(const xdouble &remnant_flux,
//                          const xdouble *x,
//                          xdouble *M)
// {
//    xdouble r[] = {0.0, 0.0, 0.0};
//    r[0] = x[0];
//    r[1] = x[1];
//    xdouble norm_r = sqrt(r[0] * r[0] + r[1] * r[1]);
//    M[0] = -r[0] * remnant_flux / norm_r;
//    M[1] = -r[1] * remnant_flux / norm_r;
//    M[2] = 0.0;
// }

// template <typename xdouble = double>
// void cw_magnetization(const xdouble &remnant_flux, const xdouble *x,
// xdouble *M)
// {
//    xdouble r[] = {0.0, 0.0, 0.0};
//    r[0] = x[0];
//    r[1] = x[1];
//    xdouble norm_r = sqrt(r[0] * r[0] + r[1] * r[1]);
//    M[0] = -r[1] * remnant_flux / norm_r;
//    M[1] = r[0] * remnant_flux / norm_r;
//    M[2] = 0.0;
// }

// template <typename xdouble = double>
// void ccw_magnetization(const xdouble &remnant_flux,
//                        const xdouble *x,
//                        xdouble *M)
// {
//    xdouble r[] = {0.0, 0.0, 0.0};
//    r[0] = x[0];
//    r[1] = x[1];
//    xdouble norm_r = sqrt(r[0] * r[0] + r[1] * r[1]);
//    M[0] = r[1] * remnant_flux / norm_r;
//    M[1] = -r[0] * remnant_flux / norm_r;
//    M[2] = 0.0;
// }

// template <typename xdouble = double, int sign = 1>
// void x_axis_current(const xdouble *x, xdouble *J)
// {
//    J[0] = sign;
//    J[1] = 0.0;
//    J[2] = 0.0;
// }

// template <typename xdouble = double, int sign = 1>
// void y_axis_current(const xdouble *x, xdouble *J)
// {
//    J[0] = 0.0;
//    J[1] = sign;
//    J[2] = 0.0;
// }

// template <typename xdouble = double, int sign = 1>
// void z_axis_current(const xdouble *x, xdouble *J)
// {
//    J[0] = 0.0;
//    J[1] = 0.0;
//    J[2] = sign;
// }

// template <typename xdouble = double>
// void ring_current(const xdouble *x, xdouble *J)
// {
//    for (int i = 0; i < 3; ++i)
//    {
//       J[i] = 0.0;
//    }
//    xdouble r[] = {0.0, 0.0, 0.0};
//    r[0] = x[0];
//    r[1] = x[1];
//    xdouble norm_r = sqrt(r[0] * r[0] + r[1] * r[1]);
//    r[0] /= norm_r;
//    r[1] /= norm_r;
//    J[0] = -r[1];
//    J[1] = r[0];
// }

// template <typename xdouble = double>
// void x_axis_magnetization(const xdouble &remnant_flux,
//                           const xdouble *x,
//                           xdouble *M)
// {
//    M[0] = remnant_flux;
//    M[1] = 0.0;
//    M[2] = 0.0;
// }

// template <typename xdouble = double>
// void y_axis_magnetization(const xdouble &remnant_flux,
//                           const xdouble *x,
//                           xdouble *M)
// {
//    M[0] = 0.0;
//    M[1] = remnant_flux;
//    M[2] = 0.0;
// }

// template <typename xdouble = double>
// void z_axis_magnetization(const xdouble &remnant_flux,
//                           const xdouble *x,
//                           xdouble *M)
// {
//    M[0] = 0.0;
//    M[1] = 0.0;
//    M[2] = remnant_flux;
// }

// template <typename xdouble = double>
// void box1_current(const xdouble *x, xdouble *J)
// {
//    for (int i = 0; i < 3; ++i)
//    {
//       J[i] = 0.0;
//    }

//    xdouble y = x[1] - .5;

//    // J[2] = -current_density*6*y*(1/(M_PI*4e-7)); // for real scaled
//    problem J[2] = -6 * y;
// }

// template <typename xdouble = double>
// void box2_current(const xdouble *x, xdouble *J)
// {
//    for (int i = 0; i < 3; ++i)
//    {
//       J[i] = 0.0;
//    }

//    xdouble y = x[1] - .5;

//    // J[2] = current_density*6*y*(1/(M_PI*4e-7)); // for real scaled
//    problem J[2] = 6 * y;
// }

// /// function to get the sign of a number
// template <typename T>
// int sgn(T val)
// {
//    return (T(0) < val) - (val < T(0));
// }

// template <typename xdouble = double>
// void team13_current(const xdouble *X, xdouble *J)
// {
//    for (int i = 0; i < 3; ++i)
//    {
//       J[i] = 0.0;
//    }

//    auto x = X[0];
//    auto y = X[1];

//    if (y >= -0.075 && y <= 0.075)
//    {
//       J[1] = sgn(x);
//    }
//    else if (x >= -0.075 && x <= 0.075)
//    {
//       J[0] = -sgn(y);
//    }
//    else if (x > 0.075 && y > 0.075)
//    {
//       J[0] = -(y - 0.075);
//       J[1] = (x - 0.075);
//    }
//    else if (x < 0.075 && y > 0.075)
//    {
//       J[0] = -(y - 0.075);
//       J[1] = (x + 0.075);
//    }
//    else if (x < 0.075 && y < 0.075)
//    {
//       J[0] = -(y + 0.075);
//       J[1] = (x + 0.075);
//    }
//    else if (x > 0.075 && y < 0.075)
//    {
//       J[0] = -(y + 0.075);
//       J[1] = (x - 0.075);
//    }

//    auto norm = sqrt(J[0] * J[0] + J[1] * J[1]);

//    J[0] /= norm;
//    J[1] /= norm;
// }

// // void func(const mfem::Vector &x, mfem::Vector &y)
// // {
// //    y.SetSize(3);
// //    y(0) = x(0)*x(0) - x(1);
// //    y(1) = x(0) * exp(x(1));
// //    y(2) = x(2)*x(0) - x(1);
// // }

// // void funcRevDiff(const mfem::Vector &x, const mfem::Vector &v_bar,
// // mfem::Vector &x_bar)
// // {
// //    x_bar(0) = v_bar(0) * 2*x(0) + v_bar(1) * exp(x(1)) + v_bar(2)*x(2);
// //    x_bar(1) = -v_bar(0) + v_bar(1) * x(0) * exp(x(1)) - v_bar(2);
// //    x_bar(2) = v_bar(2) * x(0);
// // }

// }  // anonymous namespace

// namespace miso
// {
// MagnetostaticSolver::MagnetostaticSolver(const nlohmann::json
// &json_options,
//                                          std::unique_ptr<mfem::Mesh>
//                                          smesh, MPI_Comm comm)
//  : AbstractSolver(json_options, move(smesh), comm), dim(mesh->Dimension())
// {
//    mesh->RemoveInternalBoundaries();

//    // /// Create the H(Div) finite element collection
//    // // h_div_coll.reset(new RT_FECollection(order, dim));
//    // /// Create the H1 finite element collection
//    // h1_coll.reset(new H1_FECollection(order, dim));
//    // /// Create the L2 finite element collection
//    // l2_coll.reset(new L2_FECollection(order, dim));

//    // /// Create the H(Div) finite element space
//    // // h_div_space.reset(new SpaceType(mesh.get(), h_div_coll.get()));
//    // /// Create the H1 finite element space
//    // h1_space.reset(new SpaceType(mesh.get(), h1_coll.get()));
//    // /// Create the L2 finite element space
//    // l2_space.reset(new SpaceType(mesh.get(), l2_coll.get()));

//    /// Create magnetic flux density grid function
//    // B.reset(new GridFunType(h_div_space.get()));
//    dim = mesh->Dimension();
//    int order = options["space-dis"]["degree"].get<int>();
//    // if (dim == 3)
//    // {
//    auto *h_div_coll = new RT_FECollection(order, dim);
//    auto *h_div_space = new ParFiniteElementSpace(mesh.get(), h_div_coll);
//    res_fields.emplace("B", h_div_space);
//    res_fields.at("B").MakeOwner(h_div_coll);
//    // }
//    // else
//    // {
//    //    auto *h_curl_coll = new ND_FECollection(order, dim);
//    //    auto *h_curl_space = new ParFiniteElementSpace(mesh.get(),
//    //    h_curl_coll); res_fields.emplace("B", h_curl_space);
//    //    res_fields.at("B").MakeOwner(h_curl_coll);
//    // }

//    B = &res_fields.at("B");
// }

// void MagnetostaticSolver::calcCurl(const HypreParVector &A, HypreParVector
// &B)
// {
//    auto &B_gf = res_fields.at("B");
//    DiscreteCurlOperator curl(fes.get(), B_gf.ParFESpace());
//    curl.Assemble();
//    curl.Finalize();
//    std::unique_ptr<HypreParMatrix> curl_mat(curl.ParallelAssemble());
//    curl_mat->Mult(A, B);
// }

// void MagnetostaticSolver::printSolution(const std::string &file_name,
//                                         int refine)
// {
//    printFields(
//        file_name, {u.get(), B}, {"MVP", "Magnetic_Flux_Density"}, refine);
// }

// void MagnetostaticSolver::setEssentialBoundaries()
// {
//    // /// apply zero tangential boundary condition everywhere
//    // ess_bdr.SetSize(mesh->bdr_attributes.Max());
//    // ess_bdr = 1;

//    // // Array<int> internal_bdr;
//    // // mesh->GetInternalBoundaries(internal_bdr);
//    // // for (int i = 0; i < internal_bdr.Size(); ++i)
//    // // {
//    // //    ess_bdr[internal_bdr[i]] = 0;
//    // // }

//    // Array<int> ess_tdof_list;
//    // fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
//    // res->SetEssentialTrueDofs(ess_tdof_list);
//    // /// set current vector's ess_tdofs to zero
//    // load->SetSubVector(ess_tdof_list, 0.0);

//    AbstractSolver::setEssentialBoundaries();
// }

// void MagnetostaticSolver::setFieldValue(
//     HypreParVector &field,
//     const std::function<void(const Vector &, Vector &)> &u_init)
// {
//    const int dimc = (dim == 3) ? 3 : 1;
//    VectorFunctionCoefficient u0(dimc, u_init);
//    *scratch = field;
//    scratch->ProjectCoefficient(u0);
//    scratch->GetTrueDofs(field);
// }

// void MagnetostaticSolver::_solveUnsteady(ParGridFunction &state)
// {
//    double t = 0.0;
//    evolver->SetTime(t);
//    ode_solver->Init(*evolver);

//    // output the mesh and initial condition
//    // TODO: need to swtich to vtk for SBP
//    int precision = 8;
//    {
//       ofstream omesh("initial.mesh");
//       omesh.precision(precision);
//       mesh->Print(omesh);
//       ofstream osol("initial-sol.gf");
//       osol.precision(precision);
//       state.Save(osol);
//    }

//    /// TODO: put this in options
//    // bool paraview = !options["time-dis"]["steady"].get<bool>();
//    // bool paraview = true;
//    // std::unique_ptr<ParaViewDataCollection> pd;
//    // if (paraview)
//    // {
//    //    pd.reset(new ParaViewDataCollection("time_hist", mesh.get()));
//    //    pd->SetPrefixPath("ParaView");
//    //    pd->RegisterField("state", &state);
//    //    pd->RegisterField("B", B);
//    //    pd->SetLevelsOfDetail(options["space-dis"]["degree"].get<int>() +
//    1);
//    //    pd->SetDataFormat(VTKFormat::BINARY);
//    //    pd->SetHighOrderOutput(true);
//    //    pd->SetCycle(0);
//    //    pd->SetTime(t);
//    //    pd->Save();
//    // }

//    // std::cout.precision(16);
//    // std::cout << "res norm: " << calcResidualNorm(state) << "\n";

//    // auto residual = ParGridFunction(fes.get());
//    // calcResidual(state, residual);
//    // printFields("init", {&residual, &state}, {"Residual", "Solution"});

//    auto t_final = options["time-dis"]["t-final"].template get<double>();
//    *out << "t_final is " << t_final << '\n';

//    int ti = 0;
//    // bool done = false;
//    double dt = 1.0;
//    initialHook(state);

//    // int max_iter = options["time-dis"]["max-iter"].get<int>();
//    // double dlambda = 1.0/(max_iter-1);
//    // std::vector<double> lambda =
//    // {0.0, 1./8, 1./3, 2./3, 3./4, 7./8, 15./16, 31./32, 63./64,
//    // 127./128, 1.0}; std::vector<double> lambda = {0.0, 0.67, 1.0}; //
//    just for
//    // a test
//    std::vector<double> lambda = {1.0};  // just for a test
//    // for (ti = 0; ti < options["time-dis"]["max-iter"].get<int>(); ++ti)
//    for (ti = 0; ti < lambda.size(); ++ti)
//    {
//       ParameterContinuationCoefficient::setLambda(lambda[ti]);
//       // dt = calcStepSize(ti, t, t_final, dt, state);
//       *out << "iter " << ti << ": time = " << t << ": dt = " << dt;
//       if (!options["time-dis"]["steady"].get<bool>())
//       {
//          *out << " (" << round(100 * t / t_final) << "% complete)";
//       }
//       *out << endl;
//       iterationHook(ti, t, dt, state);
//       // computeSecondaryFields(state);
//       auto &u_true = state.GetTrueVector();
//       ode_solver->Step(u_true, t, dt);
//       state.SetFromTrueDofs(u_true);

//       // if (paraview)
//       // {
//       //    pd->SetCycle(ti);
//       //    pd->SetTime(t);
//       //    pd->Save();
//       // }
//       // std::cout << "res norm: " << calcResidualNorm(state) << "\n";

//       // if (iterationExit(ti, t, t_final, dt, state)) break;
//    }
//    // {
//    //    ofstream osol("final_before_TH.gf");
//    //    osol.precision(std::numeric_limits<long double>::digits10 + 1);
//    //    state.Save(osol);
//    // }
//    terminalHook(ti, t, state);

//    // Save the final solution. This output can be viewed later using
//    GLVis:
//    // glvis -m unitGridTestMesh.msh -g adv-final.gf".
//    {
//       ofstream osol("final.gf");
//       osol.precision(std::numeric_limits<long double>::digits10 + 1);
//       state.Save(osol);
//    }
//    // write the solution to vtk file
//    if (options["space-dis"]["basis-type"].template get<string>() ==
//    "csbp")
//    {
//       ofstream sol_ofs("final_cg.vtk");
//       sol_ofs.precision(14);
//       mesh->PrintVTK(sol_ofs,
//                      options["space-dis"]["degree"].template get<int>() +
//                      1);
//       state.SaveVTK(sol_ofs,
//                     "Solution",
//                     options["space-dis"]["degree"].template get<int>() +
//                     1);
//       sol_ofs.close();
//       printField("final", state, "Solution");
//    }
//    else if (options["space-dis"]["basis-type"].template get<string>() ==
//    "dsbp")
//    {
//       ofstream sol_ofs("final_dg.vtk");
//       sol_ofs.precision(14);
//       mesh->PrintVTK(sol_ofs,
//                      options["space-dis"]["degree"].template get<int>() +
//                      1);
//       state.SaveVTK(sol_ofs,
//                     "Solution",
//                     options["space-dis"]["degree"].template get<int>() +
//                     1);
//       sol_ofs.close();
//       printField("final", state, "Solution");
//    }
//    // TODO: These mfem functions do not appear to be parallelized
// }

// void MagnetostaticSolver::solveUnsteady(ParGridFunction &state)
// {
//    // auto old_comps = options["components"];
//    // // std::cout << "old cops: " << setw(3) << old_comps << "\n";
//    // for (auto& component : options["components"])
//    // {
//    //    if (!component["linear"].get<bool>())
//    //    {
//    //       component["linear"] = true;
//    //    }
//    // }
//    // // std::cout << "old cops: " << setw(3) << old_comps << "\n";
//    // constructReluctivity();
//    // Array<NonlinearFormIntegrator*> &dnfi = *res->GetDNFI();
//    // delete dnfi[0];
//    // dnfi[0] = new CurlCurlNLFIntegrator(nu.get());
//    // // AbstractSolver::solveUnsteady(state);
//    // auto old_time_dis = options["time-dis"];
//    // options["time-dis"]["dt"] = 1e12;
//    // options["time-dis"]["max-iter"] = 1;
//    // options["time-dis"]["steady-reltol"] = 1e-8;
//    // auto old_newton_opt = options["nonlin-solver"];
//    // options["nonlin-solver"]["reltol"] = 1e-8;

//    // _solveUnsteady(state);

//    // options["time-dis"] = old_time_dis;
//    // options["components"] = old_comps;
//    // options["nonlin-solver"] = old_newton_opt;
//    // constructReluctivity();
//    // dnfi = *res->GetDNFI();
//    // delete dnfi[0];
//    // dnfi[0] = new CurlCurlNLFIntegrator(nu.get());
//    // AbstractSolver::solveUnsteady(state);

//    // HypreParVector new_load(fes.get());
//    // new_load = 0.0;
//    // addLoad(*load, new_load);

//    _solveUnsteady(state);
//    // AbstractSolver::solveUnsteady(state);
// }
// //    *out << "Tucker: please check if the code below is needed" << endl;
// //    // if (newton_solver == nullptr)
// //    //    constructNewtonSolver();

// // setEssentialBoundaries();

// // Vector Zero(3);
// // Zero = 0.0;
// // bc_coef.reset(new VectorConstantCoefficient(Zero)); // for motor
// // // bc_coef.reset(new VectorFunctionCoefficient(3, a_exact)); // for box
// // problem

// // *u = 0.0;
// // u->ProjectBdrCoefficientTangent(*bc_coef, ess_bdr);

// // HypreParVector *u_true = u->GetTrueDofs();
// // auto load_gf = dynamic_cast<ParGridFunction*>(load.get());
// // HypreParVector *current_true = load_gf->GetTrueDofs();
// // newton_solver->Mult(*current_true, *u_true);
// // MFEM_VERIFY(newton_solver->GetConverged(), "Newton solver did not
// // converge."); u->SetFromTrueDofs(*u_true); Vector Zero(3); Zero = 0.0;
// bool
// // box_prob = options["problem-opts"].value("box", false);

// // if (!box_prob)
// //    bc_coef.reset(new VectorConstantCoefficient(Zero)); // for motor
// // else
// //    bc_coef.reset(new VectorFunctionCoefficient(3, a_exact)); // for box
// //    problem

// // computeSecondaryFields();
// // }

// // void MagnetostaticSolver::initialHook(const mfem::ParGridFunction
// &state)
// // {
// /// TODO!!
// // Vector Zero(3);
// // Zero = 0.0;
// // bc_coef.reset(new VectorConstantCoefficient(Zero)); // for motor
// // // bc_coef.reset(new VectorFunctionCoefficient(3, a_exact)); // for box
// // problem

// // state = 0.0;
// // state.ProjectBdrCoefficientTangent(*bc_coef, ess_bdr);
// // }

// void MagnetostaticSolver::addOutput(const std::string &fun,
//                                     const nlohmann::json &options)
// {
//    if (fun == "energy")
//    {
//       FunctionalOutput out(*fes, res_fields);
//       out.addOutputDomainIntegrator(new MagneticEnergyIntegrator(*nu));
//       // MISOOutput mout(std::move(out));
//       outputs.emplace(fun, std::move(out));
//    }
//    else if (fun == "ac_loss")
//    {
//       FunctionalOutput out(*fes, res_fields);
//       out.addOutputDomainIntegrator(
//           // new HybridACLossFunctionalIntegrator(*sigma, 1.0, 1.0, 1.0));
//           new ACLossFunctionalIntegrator(*sigma, 1.0, 1.0));
//       outputs.emplace(fun, std::move(out));
//    }
//    else if (fun == "dc_loss")
//    {
//       FunctionalOutput out(*fes, res_fields);
//       out.addOutputDomainIntegrator(
//           new DCLossFunctionalIntegrator(*sigma, *current_coeff, 1.0));
//       outputs.emplace(fun, std::move(out));
//    }
//    else if (fun == "force")
//    {
//       /// create displacement field V that uses the same FES as the mesh
//       auto &mesh_gf = *dynamic_cast<ParGridFunction *>(mesh->GetNodes());
//       res_fields.emplace("v" + fun, mesh_gf.ParFESpace());

//       ForceFunctional out(*fes, res_fields, options, *nu);
//       outputs.emplace(fun, std::move(out));
//    }
//    else if (fun == "torque")
//    {
//       /// create displacement field V that uses the same FES as the mesh
//       auto &mesh_gf = *dynamic_cast<ParGridFunction *>(mesh->GetNodes());
//       res_fields.emplace("v" + fun, mesh_gf.ParFESpace());

//       TorqueFunctional out(*fes, res_fields, options, *nu);
//       outputs.emplace(fun, std::move(out));
//    }
//    else if (fun == "max_flux")
//    {
//       IECurlMagnitudeAggregateFunctional out(*fes, res_fields, options);
//       outputs.emplace(fun, std::move(out));
//    }
//    else if (fun == "flux_squared_avg")
//    {
//       BNormSquaredAverageFunctional out(*fes, res_fields, options);
//       outputs.emplace(fun, std::move(out));
//    }
//    else
//    {
//       throw MISOException("Output with name " + fun +
//                           " not supported by "
//                           "MagnetostaticSolver!\n");
//    }
// }

// std::vector<GridFunType *> MagnetostaticSolver::getFields()
// {
//    return {u.get(), B};
// }

// void MagnetostaticSolver::constructForms()
// {
//    // MagnetostaticResidual mres(
//    //     *fes, res_fields, *current_coeff, *mag_coeff, *nu);
//    // new_res = std::make_unique<MISOResidual>(std::move(mres));
//    // mass.reset(new BilinearFormType(fes.get()));
//    res = std::make_unique<NonlinearFormType>(fes.get());
//    magnetostatic_load = std::make_unique<MagnetostaticLoad>(
//        *fes, *current_coeff, *mag_coeff, *nu);
//    load = std::make_unique<MISOLoad>(*magnetostatic_load);
//    // old_load.reset(new ParGridFunction(fes.get()));
//    ent = std::make_unique<ParNonlinearForm>(fes.get());
// }

// // GridFunction *MagnetostaticSolver::getMeshSensitivities()
// // {
// // /// assign mesh node space to forms
// // mesh->EnsureNodes();
// // SpaceType *mesh_fes =
// // static_cast<SpaceType*>(mesh->GetNodes()->FESpace());

// // dLdX.reset(new GridFunType(mesh_fes));
// // *dLdX = 0.0;

// // /// Add mesh sensitivities of functional
// // LinearFormType dJdX(mesh_fes);
// // dJdX.AddDomainIntegrator(
// //    new MagneticCoenergyIntegrator(*u, nu.get()));
// // dJdX.Assemble();
// // std::cout << "dJdX norm: " << dJdX.Norml2() << "\n";
// // /// TODO I don't know if this works in parallel / when we need to use
// tdof
// // vectors
// // // *dLdX -= dJdX;

// // res_mesh_sens_l.reset(new LinearFormType(mesh_fes));

// // /// compute \psi_A
// // solveForAdjoint("co-energy");

// // /// add integrators R = CurlCurl(A) + Cm + Mj = 0
// // /// \psi^T CurlCurl(A)
// // res_mesh_sens_l->AddDomainIntegrator(
// //    new CurlCurlNLFIntegrator(nu.get(), u.get(), adj.get()));
// // /// \psi^T C m
// // res_mesh_sens_l->AddDomainIntegrator(
// //    new VectorFECurldJdXIntegerator(nu.get(), M.get(), adj.get(),
// //                                    mag_coeff.get(), -1.0));

// // /// Compute the derivatives and accumulate the result
// // res_mesh_sens_l->Assemble();

// // ParGridFunction j_mesh_sens(mesh_fes);
// // j_mesh_sens = 0.0;
// // auto *j_mesh_sens_true = j_mesh_sens.GetTrueDofs();
// // getCurrentSourceMeshSens(*adj, *j_mesh_sens_true);
// // std::cout << "residual dJdX norm: " << res_mesh_sens_l->Norml2() <<
// "\n";
// // std::cout << "current source dJdX norm: " << j_mesh_sens_true->Norml2()
// <<
// // "\n";
// // /// dJdX = \partialJ / \partial X + \psi^T \partial R / \partial X
// // dLdX->Add(1, *res_mesh_sens_l);
// // dLdX->Add(-1, *j_mesh_sens_true);

// // return dLdX.get();
// // }

// void MagnetostaticSolver::verifyMeshSensitivities()
// {
//    // std::cout << "Verifying Mesh Sensitivities..." << std::endl;
//    // int dim = mesh->SpaceDimension();
//    // double delta = 1e-7;
//    // double delta_cd = 1e-5;
//    // double dJdX_fd_v = -calcOutput("co-energy") / delta;
//    // double dJdX_cd_v = 0.0;

//    // VectorFunctionCoefficient v_rand(dim, randState);
//    // // GridFunction state(fes.get());
//    // // GridFunction adjoint(fes.get());
//    // // state.ProjectCoefficient(v_rand);
//    // // adjoint.ProjectCoefficient(v_rand);

//    // ess_bdr.SetSize(mesh->bdr_attributes.Max());
//    // ess_bdr = 0;

//    // Array<int> ess_tdof_list;
//    // fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
//    // res->SetEssentialTrueDofs(ess_tdof_list);

//    // Vector *dJdX_vect = getMeshSensitivities();

//    // // extract mesh nodes and get their finite-element space
//    // GridFunction *x_nodes = mesh->GetNodes();
//    // FiniteElementSpace *mesh_fes = x_nodes->FESpace();
//    // GridFunction dJdX(mesh_fes, dJdX_vect->GetData());
//    // GridFunction dJdX_fd(mesh_fes); GridFunction dJdX_cd(mesh_fes);
//    // GridFunction dJdX_fd_err(mesh_fes); GridFunction
//    dJdX_cd_err(mesh_fes);
//    // // initialize the vector that we use to perturb the mesh nodes
//    // GridFunction v(mesh_fes);
//    // v.ProjectCoefficient(v_rand);

//    // /// only perturb the inside dofs not on model faces
//    // // for (int i = 0; i < mesh_fes_surface_dofs.Size(); ++i)
//    // // {
//    // //    v(mesh_fes_surface_dofs[i]) = 0.0;
//    // // }

//    // // contract dJ/dX with v
//    // double dJdX_v = (dJdX) * v;

//    // if (options["verify-full"].get<bool>())
//    // {
//    //    for(int k = 0; k < x_nodes->Size(); k++)
//    //    {
//    //       GridFunction x_pert(*x_nodes);
//    //       x_pert(k) += delta; mesh->SetNodes(x_pert);
//    //       std::cout << "Solving Forward Step..." << std::endl;
//    //       Update();
//    //       solveForState();
//    //       std::cout << "Solver Done" << std::endl;
//    //       dJdX_fd(k) = calcOutput("co-energy")/delta + dJdX_fd_v;
//    //       x_pert(k) -= delta; mesh->SetNodes(x_pert);

//    //    }
//    //    // central difference
//    //    for(int k = 0; k < x_nodes->Size(); k++)
//    //    {
//    //       //forward
//    //       GridFunction x_pert(*x_nodes);
//    //       x_pert(k) += delta_cd; mesh->SetNodes(x_pert);
//    //       std::cout << "Solving Forward Step..." << std::endl;
//    //       Update();
//    //       solveForState();
//    //       std::cout << "Solver Done" << std::endl;
//    //       dJdX_cd(k) = calcOutput("co-energy")/(2*delta_cd);

//    //       //backward
//    //       x_pert(k) -= 2*delta_cd; mesh->SetNodes(x_pert);
//    //       std::cout << "Solving Backward Step..." << std::endl;
//    //       Update();
//    //       solveForState();
//    //       std::cout << "Solver Done" << std::endl;
//    //       dJdX_cd(k) -= calcOutput("co-energy")/(2*delta_cd);
//    //       x_pert(k) += delta_cd; mesh->SetNodes(x_pert);
//    //    }

//    //    dJdX_fd_v = dJdX_fd*v;
//    //    dJdX_cd_v = dJdX_cd*v;
//    //    dJdX_fd_err += dJdX_fd; dJdX_fd_err -= dJdX;
//    //    dJdX_cd_err += dJdX_cd; dJdX_cd_err -= dJdX;
//    //    std::cout << "FD L2:  " << dJdX_fd_err.Norml2() << std::endl;
//    //    std::cout << "CD L2:  " << dJdX_cd_err.Norml2() << std::endl;
//    //    for(int k = 0; k < x_nodes->Size(); k++)
//    //    {
//    //       dJdX_fd_err(k) = dJdX_fd_err(k)/dJdX(k);
//    //       dJdX_cd_err(k) = dJdX_cd_err(k)/dJdX(k);
//    //    }
//    //    stringstream fderrname;
//    //    fderrname << "dJdX_fd_err.gf";
//    //    ofstream fd(fderrname.str()); fd.precision(15);
//    //    dJdX_fd_err.Save(fd);

//    //    stringstream cderrname;
//    //    cderrname << "dJdX_cd_err.gf";
//    //    ofstream cd(cderrname.str()); cd.precision(15);
//    //    dJdX_cd_err.Save(cd);

//    //    stringstream analytic;
//    //    analytic << "dJdX.gf";
//    //    ofstream an(analytic.str()); an.precision(15);
//    //    dJdX.Save(an);
//    // }
//    // else
//    // {
//    //    // // compute finite difference approximation
//    //    GridFunction x_pert(*x_nodes);
//    //    // x_pert.Add(delta, v);
//    //    // mesh->SetNodes(x_pert);
//    //    // std::cout << "Solving Forward Step..." << std::endl;
//    //    // Update();
//    //    // residual.reset(new GridFunType(fes.get()));
//    //    // *residual = 0.0;
//    //    // res->Mult(state, *residual);
//    //    // *residual -= *load;
//    //    // dJdX_fd_v += *residual * adjoint;
//    //    // // solveForState();
//    //    // std::cout << "Solver Done" << std::endl;
//    //    // // dJdX_fd_v += calcOutput("co-energy")/delta;

//    //    // central difference approximation
//    //    std::cout << "Solving CD Backward Step..." << std::endl;
//    //    x_pert = *x_nodes; x_pert.Add(-delta_cd, v);
//    //    mesh->SetNodes(x_pert);
//    //    Update();
//    //    residual.reset(new GridFunType(fes.get()));
//    //    *residual = 0.0;
//    //    res->Mult(*u, *residual);
//    //    // *residual -= *load;
//    //    dJdX_cd_v -= (*residual * *adj )/(2*delta_cd);

//    //    // solveForState();
//    //    // std::cout << "Solver Done" << std::endl;
//    //    // dJdX_cd_v = -calcOutput("co-energy")/(2*delta_cd);

//    //    std::cout << "Solving CD Forward Step..." << std::endl;
//    //    x_pert.Add(2*delta_cd, v);
//    //    mesh->SetNodes(x_pert);
//    //    Update();
//    //    residual.reset(new GridFunType(fes.get()));
//    //    *residual = 0.0;
//    //    res->Mult(*u, *residual);
//    //    // *residual -= *load;
//    //    dJdX_cd_v += (*residual * *adj )/(2*delta_cd);

//    //    // solveForState();
//    //    // std::cout << "Solver Done" << std::endl;
//    //    // dJdX_cd_v += calcOutput("co-energy")/(2*delta_cd);
//    // }

//    // std::cout << "Volume Mesh Sensititivies:  " << std::endl;
//    // // std::cout << "Finite Difference:          " << dJdX_fd_v <<
//    std::endl;
//    // std::cout << "Central Difference:         " << dJdX_cd_v <<
//    std::endl;
//    // std::cout << "Analytic:                   " << dJdX_v << std::endl;
//    // // std::cout << "FD Relative:                " <<
//    // (dJdX_v-dJdX_fd_v)/dJdX_v << std::endl;
//    // // std::cout << "FD Absolute:                " << dJdX_v - dJdX_fd_v
//    <<
//    // std::endl; std::cout << "CD Relative:                " <<
//    // (dJdX_v-dJdX_cd_v)/dJdX_v << std::endl; std::cout << "CD Absolute: "
//    <<
//    // dJdX_v - dJdX_cd_v << std::endl;
// }

// void MagnetostaticSolver::Update()
// {
//    fes->Update();
//    // h_div_space->Update();
//    // h1_space->Update();

//    u->Update();
//    adj->Update();
//    // B->Update();
//    // M->Update();
//    // load->Update();
//    // auto load_gf = dynamic_cast<ParGridFunction*>(load.get());
//    // load_gf->Update();
//    div_free_current_vec->Update();

//    res->Update();
//    assembleCurrentSource();
//    assembleMagnetizationSource();
// }

// void MagnetostaticSolver::setInitialCondition(ParGridFunction &state,
//                                               const Vector &u_init)
// {
//    state = 0.0;
//    // auto initState = [](const mfem::Vector &x, mfem::Vector &A)
//    // {
//    //    A(0) = 0.5*x(1);
//    //    A(1) = -0.5*x(0);
//    //    A(2) = 0.0;
//    // };
//    // VectorFunctionCoefficient internalState(dim, initState);
//    // state.ProjectCoefficient(internalState);

//    VectorConstantCoefficient u0(u_init);
//    state.ProjectBdrCoefficientTangent(u0, ess_bdr);
// }

// void MagnetostaticSolver::setInitialCondition(
//     mfem::ParGridFunction &state,
//     const std::function<void(const mfem::Vector &, mfem::Vector &)>
//     &u_init)
// {
//    state = 0.0;

//    // auto initState = [](const mfem::Vector &x, mfem::Vector &A)
//    // {
//    //    A(0) = -0.01*x(1);
//    //    A(1) = -0.01*x(0);
//    //    A(2) = 0.0;
//    // };
//    // VectorFunctionCoefficient internalState(dim, initState);
//    // state.ProjectCoefficient(internalState);

//    VectorFunctionCoefficient u0(dim, u_init);
//    state.ProjectBdrCoefficientTangent(u0, ess_bdr);
//    // state.ProjectCoefficient(u0);
//    // printField("uinit", state, "solution");
//    // state = 100.0;
// }

// void MagnetostaticSolver::initialHook(const ParGridFunction &state)
// {
//    if (options["time-dis"]["steady"].template get<bool>())
//    {
//       // res_norm0 is used to compute the time step in PTC
//       res_norm0 = calcResidualNorm(state);
//    }
// }

// bool MagnetostaticSolver::iterationExit(int iter,
//                                         double t,
//                                         double t_final,
//                                         double dt,
//                                         const ParGridFunction &state)
//                                         const
// {
//    if (options["time-dis"]["steady"].template get<bool>())
//    {
//       // use tolerance options for Newton's method
//       double norm = calcResidualNorm(state);
//       return norm <= options["time-dis"]["steady-abstol"].get<double>() ||
//              norm <=
//                  res_norm0 *
//                  options["time-dis"]["steady-reltol"].get<double>();
//    }
//    else
//    {
//       throw MISOException("MagnetostaticSolver requires steady
//       time-dis!\n");
//    }
// }

// void MagnetostaticSolver::terminalHook(int iter,
//                                        double t_final,
//                                        const ParGridFunction &state)
// {
//    computeSecondaryFields(state);

//    // auto *state_gf = const_cast<ParGridFunction*>(&state);
//    // printFields("em_state", {state_gf, B}, {"mvp", "B"});
// }

// double MagnetostaticSolver::calcStepSize(int iter,
//                                          double t,
//                                          double t_final,
//                                          double dt_old,
//                                          const ParGridFunction &state)
//                                          const
// {
//    if (options["time-dis"]["steady"].template get<bool>())
//    {
//       // ramp up time step for pseudo-transient continuation
//       // TODO: the l2 norm of the weak residual is probably not ideal here
//       // A better choice might be the l1 norm
//       double res_norm = calcResidualNorm(state);
//       if (std::abs(res_norm) <= 1e-14)
//       {
//          return 1e14;
//       }
//       double exponent = options["time-dis"]["res-exp"];
//       double dt = options["time-dis"]["dt"].template get<double>() *
//                   pow(res_norm0 / res_norm, exponent);
//       return max(dt, dt_old);
//    }
//    else
//    {
//       throw MISOException("MagnetostaticSolver requires steady
//       time-dis!\n");
//    }
// }

// // unique_ptr<NewtonSolver> MagnetostaticSolver::constructNonlinearSolver(
// //    nlohmann::json &_options, mfem::Solver &_lin_solver)
// // {
// //    std::string solver_type = _options["type"].get<std::string>();
// //    double abstol = _options["abstol"].get<double>();
// //    double reltol = _options["reltol"].get<double>();
// //    int maxiter = _options["maxiter"].get<int>();
// //    int ptl = _options["printlevel"].get<int>();
// //    unique_ptr<NewtonSolver> nonlin_solver;
// //    if (solver_type == "newton")
// //    {
// //       nonlin_solver.reset(new mfem::NewtonSolver(comm));
// //    }
// //    else if (solver_type == "inexactnewton")
// //    {
// //    nonlin_solver.reset(new NewtonSolver(comm));
// //    NewtonSolver *newton =
// dynamic_cast<NewtonSolver*>(nonlin_solver.get());

// //    /// use defaults from SetAdaptiveLinRtol unless specified
// //    int type = _options.value("inexacttype", 2);
// //    double rtol0 = _options.value("rtol0", 0.5);
// //    double rtol_max = _options.value("rtol_max", 0.9);
// //    double alpha = _options.value("alpha", (0.5) * ((1.0) +
// sqrt((5.0))));
// //    double gamma = _options.value("gamma", 1.0);
// //    newton->SetAdaptiveLinRtol(type, rtol0, rtol_max, alpha, gamma);
// //    }
// //    else
// //    {
// //    throw MISOException("Unsupported nonlinear solver type!\n"
// //    "\tavilable options are: newton, inexactnewton\n");
// //    }

// //    //double eta = 1e-1;
// //    //newton_solver.reset(new InexactNewton(comm, eta));

// //    nonlin_solver->iterative_mode = true;
// //    nonlin_solver->SetSolver(dynamic_cast<Solver&>(_lin_solver));
// //    nonlin_solver->SetPrintLevel(ptl);
// //    nonlin_solver->SetRelTol(reltol);
// //    nonlin_solver->SetAbsTol(abstol);
// //    nonlin_solver->SetMaxIter(maxiter);

// //    return nonlin_solver;
// // }

// void MagnetostaticSolver::constructCoefficients()
// {
//    div_free_current_vec = std::make_unique<GridFunType>(fes.get());

//    /// read options file to set the proper values of static member
//    variables setStaticMembers();
//    /// Construct current source coefficient
//    constructCurrent();
//    /// Construct magnetization coefficient
//    constructMagnetization();
//    /// Construct reluctivity coefficient
//    constructReluctivity();
//    /// Construct electircal conductivity coefficient
//    constructSigma();
// }

// void MagnetostaticSolver::addMassIntegrators(double alpha)
// {
//    mass->AddDomainIntegrator(new VectorFEMassIntegrator());
// }

// void MagnetostaticSolver::addResVolumeIntegrators(double alpha)
// {
//    addResidualDomainIntegrator(new CurlCurlNLFIntegrator(*nu));
// }

// void MagnetostaticSolver::assembleLoadVector(double alpha)
// {
//    /// Assemble current source vector
//    assembleCurrentSource();
//    /// Assemble magnetization source vector and add it into current
//    assembleMagnetizationSource();
// }

// void MagnetostaticSolver::addEntVolumeIntegrators()
// {
//    ent->AddDomainIntegrator(new MagneticEnergyIntegrator(*nu));
// }

// void MagnetostaticSolver::setStaticMembers()
// {
//    if (options["components"].contains("magnets"))
//    {
//       auto &magnets = options["components"]["magnets"];
//       std::string material = magnets["material"].get<std::string>();
//       remnant_flux = materials[material]["B_r"].get<double>();
//       mag_mu_r = materials[material]["mu_r"].get<double>();
//       std::cout << "B_r = " << remnant_flux << "\n";
//    }
//    // fill_factor = options["problem-opts"].value("fill-factor", 1.0);
//    // current_density =
//    options["problem-opts"].value("current_density", 1.0);
// }

// void MagnetostaticSolver::constructReluctivity()
// {
//    /// set up default reluctivity to be that of free space
//    // const double mu_0 = 4e-7*M_PI;
//    std::unique_ptr<Coefficient> nu_free_space(
//        new ConstantCoefficient(1.0 / mu_0));
//    nu = std::make_unique<MeshDependentCoefficient>(move(nu_free_space));

//    /// loop over all components, construct either a linear or nonlinear
//    ///    reluctivity coefficient for each
//    for (auto &component : options["components"])
//    {
//       int attr = component.value("attr", -1);
//       if (-1 != attr)
//       {
//          std::unique_ptr<mfem::Coefficient> temp_coeff;
//          temp_coeff = constructReluctivityCoeff(component, materials);
//          nu->addCoefficient(attr, move(temp_coeff));
//       }
//       else
//       {
//          auto attrs = component["attrs"].get<std::vector<int>>();
//          for (auto &attribute : attrs)
//          {
//             std::unique_ptr<mfem::Coefficient> temp_coeff;
//             temp_coeff = constructReluctivityCoeff(component, materials);
//             nu->addCoefficient(attribute, move(temp_coeff));
//          }
//       }
//    }
// }

// /// TODO - this approach cannot support general magnet topologies where
// the
// ///        magnetization cannot be described by a single vector function
// void MagnetostaticSolver::constructMagnetization()
// {
//    mag_coeff = std::make_unique<VectorMeshDependentCoefficient>(dim);

//    if (options["problem-opts"].contains("magnets"))
//    {
//       auto &magnets = options["problem-opts"]["magnets"];
//       if (magnets.contains("north"))
//       {
//          auto attrs = magnets["north"].get<std::vector<int>>();
//          for (auto &attr : attrs)
//          {
//             std::unique_ptr<mfem::VectorCoefficient> temp_coeff(
//                 new VectorFunctionCoefficient(dim,
//                                               northMagnetizationSource,
//                                               northMagnetizationSourceRevDiff));
//             mag_coeff->addCoefficient(attr, move(temp_coeff));
//          }
//       }
//       if (magnets.contains("south"))
//       {
//          auto attrs = magnets["south"].get<std::vector<int>>();

//          for (auto &attr : attrs)
//          {
//             std::unique_ptr<mfem::VectorCoefficient> temp_coeff(
//                 new VectorFunctionCoefficient(dim,
//                                               southMagnetizationSource,
//                                               southMagnetizationSourceRevDiff));
//             mag_coeff->addCoefficient(attr, move(temp_coeff));
//          }
//       }
//       if (magnets.contains("cw"))
//       {
//          auto attrs = magnets["cw"].get<std::vector<int>>();

//          for (auto &attr : attrs)
//          {
//             std::unique_ptr<mfem::VectorCoefficient> temp_coeff(
//                 new VectorFunctionCoefficient(
//                     dim, cwMagnetizationSource,
//                     cwMagnetizationSourceRevDiff));
//             mag_coeff->addCoefficient(attr, move(temp_coeff));
//          }
//       }
//       if (magnets.contains("ccw"))
//       {
//          auto attrs = magnets["ccw"].get<std::vector<int>>();

//          for (auto &attr : attrs)
//          {
//             std::unique_ptr<mfem::VectorCoefficient> temp_coeff(
//                 new VectorFunctionCoefficient(dim,
//                                               ccwMagnetizationSource,
//                                               ccwMagnetizationSourceRevDiff));
//             mag_coeff->addCoefficient(attr, move(temp_coeff));
//          }
//       }
//       if (magnets.contains("x"))
//       {
//          auto attrs = magnets["x"].get<std::vector<int>>();
//          for (auto &attr : attrs)
//          {
//             std::unique_ptr<mfem::VectorCoefficient> temp_coeff(
//                 new VectorFunctionCoefficient(dim,
//                                               xAxisMagnetizationSource,
//                                               xAxisMagnetizationSourceRevDiff));
//             mag_coeff->addCoefficient(attr, move(temp_coeff));
//          }
//       }
//       if (magnets.contains("y"))
//       {
//          auto attrs = magnets["y"].get<std::vector<int>>();
//          for (auto &attr : attrs)
//          {
//             std::unique_ptr<mfem::VectorCoefficient> temp_coeff(
//                 new VectorFunctionCoefficient(dim,
//                                               yAxisMagnetizationSource,
//                                               yAxisMagnetizationSourceRevDiff));
//             mag_coeff->addCoefficient(attr, move(temp_coeff));
//          }
//       }
//       if (magnets.contains("z"))
//       {
//          auto attrs = magnets["z"].get<std::vector<int>>();
//          for (auto &attr : attrs)
//          {
//             std::unique_ptr<mfem::VectorCoefficient> temp_coeff(
//                 new VectorFunctionCoefficient(dim,
//                                               zAxisMagnetizationSource,
//                                               zAxisMagnetizationSourceRevDiff));
//             mag_coeff->addCoefficient(attr, move(temp_coeff));
//          }
//       }
//    }
// }

// void MagnetostaticSolver::constructCurrent()
// {
//    current_coeff = std::make_unique<VectorMeshDependentCoefficient>();

//    if (options["problem-opts"].contains("current"))
//    {
//       auto &current = options["problem-opts"]["current"];
//       if (current.contains("Phase-A"))
//       {
//          auto attrs = current["Phase-A"].get<std::vector<int>>();
//          for (auto &attr : attrs)
//          {
//             std::unique_ptr<mfem::VectorCoefficient> temp_coeff(
//                 new VectorFunctionCoefficient(
//                     dim, phaseACurrentSource,
//                     phaseACurrentSourceRevDiff));
//             current_coeff->addCoefficient(attr, move(temp_coeff));
//          }
//       }
//       if (current.contains("Phase-B"))
//       {
//          auto attrs = current["Phase-B"].get<std::vector<int>>();
//          for (auto &attr : attrs)
//          {
//             std::unique_ptr<mfem::VectorCoefficient> temp_coeff(
//                 new VectorFunctionCoefficient(
//                     dim, phaseBCurrentSource,
//                     phaseBCurrentSourceRevDiff));
//             current_coeff->addCoefficient(attr, move(temp_coeff));
//          }
//       }
//       if (current.contains("Phase-C"))
//       {
//          auto attrs = current["Phase-C"].get<std::vector<int>>();
//          for (auto &attr : attrs)
//          {
//             std::unique_ptr<mfem::VectorCoefficient> temp_coeff(
//                 new VectorFunctionCoefficient(
//                     dim, phaseCCurrentSource,
//                     phaseCCurrentSourceRevDiff));
//             current_coeff->addCoefficient(attr, move(temp_coeff));
//          }
//       }
//       if (current.contains("x"))
//       {
//          auto attrs = current["x"].get<std::vector<int>>();
//          for (auto &attr : attrs)
//          {
//             std::unique_ptr<mfem::VectorCoefficient> temp_coeff(
//                 new VectorFunctionCoefficient(
//                     dim, xAxisCurrentSource, xAxisCurrentSourceRevDiff));
//             current_coeff->addCoefficient(attr, move(temp_coeff));
//          }
//       }
//       if (current.contains("y"))
//       {
//          auto attrs = current["y"].get<std::vector<int>>();
//          for (auto &attr : attrs)
//          {
//             std::unique_ptr<mfem::VectorCoefficient> temp_coeff(
//                 new VectorFunctionCoefficient(
//                     dim, yAxisCurrentSource, yAxisCurrentSourceRevDiff));
//             current_coeff->addCoefficient(attr, move(temp_coeff));
//          }
//       }
//       if (current.contains("z"))
//       {
//          auto attrs = current["z"].get<std::vector<int>>();
//          for (auto &attr : attrs)
//          {
//             std::unique_ptr<mfem::VectorCoefficient> temp_coeff(
//                 new VectorFunctionCoefficient(
//                     dim, zAxisCurrentSource, zAxisCurrentSourceRevDiff));
//             current_coeff->addCoefficient(attr, move(temp_coeff));
//          }
//       }
//       if (current.contains("-z"))
//       {
//          auto attrs = current["-z"].get<std::vector<int>>();
//          for (auto &attr : attrs)
//          {
//             std::unique_ptr<mfem::VectorCoefficient> temp_coeff(
//                 new VectorFunctionCoefficient(
//                     dim, nzAxisCurrentSource,
//                     nzAxisCurrentSourceRevDiff));
//             current_coeff->addCoefficient(attr, move(temp_coeff));
//          }
//       }
//       if (current.contains("ring"))
//       {
//          auto attrs = current["ring"].get<std::vector<int>>();
//          for (auto &attr : attrs)
//          {
//             std::unique_ptr<mfem::VectorCoefficient> temp_coeff(
//                 new VectorFunctionCoefficient(
//                     dim, ringCurrentSource, ringCurrentSourceRevDiff));
//             current_coeff->addCoefficient(attr, move(temp_coeff));
//          }
//       }
//       if (current.contains("box1"))
//       {
//          auto attrs = current["box1"].get<std::vector<int>>();
//          for (auto &attr : attrs)
//          {
//             std::unique_ptr<mfem::VectorCoefficient> temp_coeff(
//                 new VectorFunctionCoefficient(
//                     dim, box1CurrentSource, box1CurrentSourceRevDiff));
//             current_coeff->addCoefficient(attr, move(temp_coeff));
//          }
//       }
//       if (current.contains("box2"))
//       {
//          auto attrs = current["box2"].get<std::vector<int>>();
//          for (auto &attr : attrs)
//          {
//             std::unique_ptr<mfem::VectorCoefficient> temp_coeff(
//                 new VectorFunctionCoefficient(
//                     dim, box2CurrentSource, box2CurrentSourceRevDiff));
//             current_coeff->addCoefficient(attr, move(temp_coeff));
//          }
//       }
//       if (current.contains("team13"))
//       {
//          auto attrs = current["team13"].get<std::vector<int>>();
//          for (auto &attr : attrs)
//          {
//             std::unique_ptr<mfem::VectorCoefficient> temp_coeff(
//                 new VectorFunctionCoefficient(
//                     dim, team13CurrentSource,
//                     team13CurrentSourceRevDiff));
//             current_coeff->addCoefficient(attr, move(temp_coeff));
//          }
//       }
//    }
// }

// void MagnetostaticSolver::constructSigma()
// {
//    sigma = std::make_unique<MeshDependentCoefficient>();

//    /// loop over all components, construct conductivity for each
//    for (auto &component : options["components"])
//    {
//       int attr = component.value("attr", -1);

//       const auto &material = component["material"].get<std::string>();
//       double sigma_val = materials[material].value("sigma", 0.0);

//       if (-1 != attr)
//       {
//          std::unique_ptr<mfem::Coefficient> temp_coeff;
//          temp_coeff = std::make_unique<ConstantCoefficient>(sigma_val);
//          sigma->addCoefficient(attr, move(temp_coeff));
//       }
//       else
//       {
//          auto attrs = component["attrs"].get<std::vector<int>>();
//          for (auto &attribute : attrs)
//          {
//             std::unique_ptr<mfem::Coefficient> temp_coeff;
//             temp_coeff = std::make_unique<ConstantCoefficient>(sigma_val);
//             sigma->addCoefficient(attribute, move(temp_coeff));
//          }
//       }
//    }
// }

// void MagnetostaticSolver::assembleCurrentSource()
// {
//    // int fe_order = options["space-dis"]["degree"].get<int>();

//    // /// get int rule (approach followed my MFEM Tesla Miniapp)
//    // int irOrder = h1_space->GetElementTransformation(0)->OrderW()
//    //             + 2 * fe_order;
//    // int geom = h1_space->GetFE(0)->GetGeomType();
//    // const IntegrationRule *ir = &IntRules.Get(geom, irOrder);

//    // /// Create a H(curl) mass matrix for integrating grid functions
//    // BilinearFormIntegrator *h_curl_mass_integ = new
//    VectorFEMassIntegrator;
//    // h_curl_mass_integ->SetIntRule(ir);
//    // ParBilinearForm h_curl_mass(fes.get());
//    // h_curl_mass.AddDomainIntegrator(h_curl_mass_integ);
//    // // assemble mass matrix
//    // h_curl_mass.Assemble();
//    // h_curl_mass.Finalize();

//    // ParLinearForm J(fes.get());
//    // J.AddDomainIntegrator(new
//    VectorFEDomainLFIntegrator(*current_coeff));
//    // J.Assemble();

//    // ParGridFunction j(fes.get());
//    // // j = 0.0;
//    // j.ProjectCoefficient(*current_coeff);
//    // ParGridFunction jhdiv(h_div_space.get());
//    // jhdiv.ProjectCoefficient(*current_coeff);
//    // // printFields("jproj", {&j, &jhdiv}, {"jhcurl", "jhdiv"});
//    // {
//    //    HypreParMatrix M;
//    //    Vector X, RHS;
//    //    Array<int> ess_tdof_list;
//    //    // fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
//    //    h_curl_mass.FormLinearSystem(ess_tdof_list, j, J, M, X, RHS);
//    //    *out << "solving for j in H(curl)\n";
//    //    HypreBoomerAMG amg(M);
//    //    amg.SetPrintLevel(-1);
//    //    // HypreGMRES gmres(M);
//    //    // gmres.SetTol(1e-12);
//    //    // gmres.SetMaxIter(200);
//    //    // gmres.SetPrintLevel(3);
//    //    // gmres.SetPreconditioner(amg);
//    //    // gmres.Mult(RHS, X);

//    //    HyprePCG pcg(M);
//    //    pcg.SetTol(1e-12);
//    //    pcg.SetMaxIter(200);
//    //    pcg.SetPrintLevel(2);
//    //    pcg.SetPreconditioner(amg);
//    //    pcg.Mult(RHS, X);

//    //    h_curl_mass.RecoverFEMSolution(X, J, j);
//    // }
//    // h_curl_mass.Assemble();
//    // h_curl_mass.Finalize();

//    // /// compute the divergence free current source
//    // auto div_free_proj =
//    mfem::common::DivergenceFreeProjector(*h1_space,
//    // *fes,
//    //                                           irOrder, NULL, NULL,
//    NULL);

//    // // Compute the discretely divergence-free portion of j
//    // *div_free_current_vec = 0.0;
//    // div_free_proj.Mult(j, *div_free_current_vec);

//    // std::cout << "div free norm old load: " <<
//    div_free_current_vec->Norml2()
//    // << "\n\n";

//    // // printFields("current", {&j, div_free_current_vec.get()},
//    {"jhcurl",
//    // "jdivfree"});

//    // // *old_load = 0.0;
//    // // h_curl_mass.AddMult(*div_free_current_vec, *old_load);
//    // // *old_load *= -1.0;
//    // // printField("current_source",
//    // *dynamic_cast<ParGridFunction*>(old_load.get()), "current", 5); *out
//    <<
//    // "below h_curl add mult\n";
// }

// void MagnetostaticSolver::getCurrentSourceMeshSens(
//     const mfem::GridFunction &psi_a,
//     mfem::Vector &mesh_sens)
// {
//    // Array<int> ess_bdr, ess_bdr_tdofs;
//    // ess_bdr.SetSize(h1_space->GetParMesh()->bdr_attributes.Max());
//    // ess_bdr = 1;
//    // h1_space->GetEssentialTrueDofs(ess_bdr, ess_bdr_tdofs);

//    // /// compute \psi_k
//    // /// D \psi_k = G^T M^T \psi_A (\psi_j_hat = -M^T \psi_A)
//    // ParBilinearForm h_curl_mass(fes.get());
//    // h_curl_mass.AddDomainIntegrator(new VectorFEMassIntegrator);
//    // // assemble mass matrix
//    // h_curl_mass.Assemble();
//    // h_curl_mass.Finalize();

//    // /// compute \psi_j_hat
//    // /// \psi_j_hat = -\psi_A
//    // ParGridFunction psi_j_hat(fes.get());
//    // psi_j_hat = 0.0;
//    // h_curl_mass.MultTranspose(psi_a, psi_j_hat);
//    // psi_j_hat *= -1.0; // (\psi_j_hat = -M^T \psi_A)

//    // mfem::common::ParDiscreteGradOperator grad(h1_space.get(),
//    fes.get());
//    // grad.Assemble();
//    // grad.Finalize();

//    // ParGridFunction GTMTpsi_a(h1_space.get());
//    // GTMTpsi_a = 0.0;
//    // grad.MultTranspose(*adj, GTMTpsi_a);

//    // ParBilinearForm D(h1_space.get());
//    // D.AddDomainIntegrator(new DiffusionIntegrator);
//    // D.Assemble();
//    // D.Finalize();

//    // auto *Dmat = new HypreParMatrix;

//    // ParGridFunction psi_k(h1_space.get());
//    // psi_k = 0.0;
//    // {
//    //    Vector PSIK;
//    //    Vector RHS;
//    //    D.FormLinearSystem(ess_bdr_tdofs, psi_k, GTMTpsi_a, *Dmat, PSIK,
//    RHS);
//    //    /// Diffusion matrix is symmetric, no need to transpose
//    //    // auto *DmatT = Dmat->Transpose();
//    //    HypreBoomerAMG amg(*Dmat);
//    //    amg.SetPrintLevel(0);
//    //    HypreGMRES gmres(*Dmat);
//    //    gmres.SetTol(1e-14);
//    //    gmres.SetMaxIter(200);
//    //    gmres.SetPrintLevel(-1);
//    //    gmres.SetPreconditioner(amg);
//    //    gmres.Mult(RHS, PSIK);

//    //    D.RecoverFEMSolution(PSIK, GTMTpsi_a, psi_k);
//    // }

//    // /// compute psi_j
//    // /// M^T \psi_j = W^T \psi_k - M \psi_a
//    // ParMixedBilinearForm weakDiv(fes.get(), h1_space.get());
//    // weakDiv.AddDomainIntegrator(new VectorFEWeakDivergenceIntegrator);
//    // weakDiv.Assemble();
//    // weakDiv.Finalize();

//    // ParGridFunction WTpsik(fes.get());
//    // WTpsik = 0.0;
//    // weakDiv.MultTranspose(psi_k, WTpsik);

//    // ParGridFunction Mpsia(fes.get());
//    // Mpsia = 0.0;
//    // h_curl_mass.Mult(*adj, Mpsia);
//    // WTpsik.Add(-1.0, Mpsia);

//    // ParGridFunction psi_j(fes.get());
//    // psi_j = 0.0;
//    // {
//    //    Vector PSIJ;
//    //    Vector RHS;
//    //    auto *M = new HypreParMatrix;
//    //    Array<int> ess_tdof_list;
//    //    h_curl_mass.FormLinearSystem(ess_tdof_list, psi_j, WTpsik,
//    //                                 *M, PSIJ, RHS);

//    //    HypreBoomerAMG amg(*M);
//    //    amg.SetPrintLevel(0);
//    //    HypreGMRES gmres(*M);
//    //    gmres.SetTol(1e-14);
//    //    gmres.SetMaxIter(200);
//    //    gmres.SetPrintLevel(-1);
//    //    gmres.SetPreconditioner(amg);
//    //    gmres.Mult(RHS, PSIJ);

//    //    h_curl_mass.RecoverFEMSolution(PSIJ, WTpsik, psi_j);
//    // }

//    // /// compute j
//    // LinearFormType J(fes.get());
//    // J.AddDomainIntegrator(new
//    VectorFEDomainLFIntegrator(*current_coeff));
//    // J.Assemble();

//    // GridFunType j(fes.get());
//    // j = 0.0;
//    // // j.ProjectCoefficient(*current_coeff);
//    // {
//    //    auto *M = new HypreParMatrix;
//    //    Vector X, RHS;
//    //    Array<int> ess_tdof_list;
//    //    h_curl_mass.FormLinearSystem(ess_tdof_list, j, J, *M, X, RHS);
//    //    HypreBoomerAMG amg(*M);
//    //    amg.SetPrintLevel(0);
//    //    HypreGMRES gmres(*M);
//    //    gmres.SetTol(1e-12);
//    //    gmres.SetMaxIter(200);
//    //    gmres.SetPrintLevel(0);
//    //    gmres.SetPreconditioner(amg);
//    //    gmres.Mult(RHS, X);

//    //    h_curl_mass.RecoverFEMSolution(X, J, j);
//    // }

//    // /// compute k
//    // ParGridFunction Wj(h1_space.get());
//    // Wj = 0.0;
//    // weakDiv.Mult(j, Wj);

//    // ParGridFunction k(h1_space.get());
//    // k = 0.0;
//    // {
//    //    Vector K;
//    //    Vector RHS;
//    //    D.FormLinearSystem(ess_bdr_tdofs, k, Wj, *Dmat, K, RHS);

//    //    HypreBoomerAMG amg(*Dmat);
//    //    amg.SetPrintLevel(0);
//    //    HypreGMRES gmres(*Dmat);
//    //    gmres.SetTol(1e-14);
//    //    gmres.SetMaxIter(200);
//    //    gmres.SetPrintLevel(-1);
//    //    gmres.SetPreconditioner(amg);
//    //    gmres.Mult(RHS, K);

//    //    D.RecoverFEMSolution(K, Wj, k);
//    // }

//    // SpaceType *mesh_fes =
//    // static_cast<SpaceType*>(mesh->GetNodes()->FESpace());

//    // ParLinearForm Rk_mesh_sens(mesh_fes);
//    // /// add integrators R_k = Dk - Wj = 0
//    // /// \psi_k^T Dk
//    // ConstantCoefficient one(1.0);
//    // Rk_mesh_sens.AddDomainIntegrator(
//    //    new DiffusionResIntegrator(one, &k, &psi_k));
//    // /// -\psi_k^T W j
//    // Rk_mesh_sens.AddDomainIntegrator(
//    //    new VectorFEWeakDivergencedJdXIntegrator(&j, &psi_k, nullptr,
//    -1.0));
//    // Rk_mesh_sens.Assemble();

//    // /// Add integrators R_{\hat{j}} = \hat{j} - MGk - Mj = 0
//    // ParLinearForm Rjhat_mesh_sens(mesh_fes);
//    // ParGridFunction Gk(fes.get());
//    // Gk = 0.0;
//    // grad.Mult(k, Gk);

//    // /// NOTE: Not using -1.0 here even though there are - signs in the
//    // residual
//    // /// because we're using adj, not psi_j_hat, which would be -adj
//    // Rjhat_mesh_sens.AddDomainIntegrator(
//    //    new VectorFEMassdJdXIntegerator(&Gk, &psi_a));
//    // Rjhat_mesh_sens.AddDomainIntegrator(
//    //    new VectorFEMassdJdXIntegerator(&j, &psi_a));
//    // Rjhat_mesh_sens.Assemble();

//    // /// add integrators R_j = Mj - J = 0
//    // ParLinearForm Rj_mesh_sens(mesh_fes);

//    // Rj_mesh_sens.AddDomainIntegrator(
//    //    new VectorFEMassdJdXIntegerator(&j, &psi_j));
//    // Rj_mesh_sens.AddDomainIntegrator(
//    //    new VectorFEDomainLFMeshSensInteg(&psi_j, *current_coeff, -1.0));
//    // Rj_mesh_sens.Assemble();

//    // mesh_sens.Add(1.0, *Rk_mesh_sens.ParallelAssemble());
//    // mesh_sens.Add(1.0, *Rjhat_mesh_sens.ParallelAssemble());
//    // mesh_sens.Add(1.0, *Rj_mesh_sens.ParallelAssemble());
// }

// Vector *MagnetostaticSolver::getResidual()
// {
//    residual = std::make_unique<GridFunType>(fes.get());
//    *residual = 0.0;
//    /// state needs to be the same as the current density changes, zero is
//    /// arbitrary
//    *u = 0.0;
//    res->Mult(*u, *residual);
//    // *residual -= *load;
//    return residual.get();
// }

// // Vector *MagnetostaticSolver::getResidualCurrentDensitySensitivity()
// // {
// //    current_density = 1.0;
// //    // *load = 0.0;
// //    constructCurrent();
// //    assembleCurrentSource();
// //    // *load *= -1.0;

// //    Array<int> ess_bdr(mesh->bdr_attributes.Size());
// //    Array<int> ess_tdof_list;
// //    ess_bdr = 1;
// //    fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
// //    /// set current vector's ess_tdofs to zero
// //    // load->SetSubVector(ess_tdof_list, 0.0);

// //    // return load.get();
// // }

// // double MagnetostaticSolver::getFunctionalCurrentDensitySensitivity(
// //     const std::string &fun)
// // {
// //    Array<int> ess_bdr(mesh->bdr_attributes.Size());
// //    ess_bdr = 1;
// //    res->SetEssentialBC(ess_bdr);
// //    solveForAdjoint(fun);

// //    double derivative = *adj * *getResidualCurrentDensitySensitivity();

// //    setStaticMembers();
// //    constructCurrent();
// //    constructMagnetization();
// //    assembleCurrentSource();
// //    assembleMagnetizationSource();

// //    return derivative;
// // }

// void MagnetostaticSolver::assembleMagnetizationSource()
// {
//    // M.reset(new GridFunType(h_div_space.get()));

//    // auto weakCurlMuInv_ = new ParMixedBilinearForm(h_div_space.get(),
//    // fes.get()); weakCurlMuInv_->AddDomainIntegrator(new
//    // VectorFECurlIntegrator(*nu));

//    // weakCurlMuInv_->Assemble();
//    // weakCurlMuInv_->Finalize();

//    // M->ProjectCoefficient(*mag_coeff);
//    // // weakCurlMuInv_->AddMult(*M, *old_load, -1.0);

//    // delete weakCurlMuInv_;
// }

// void MagnetostaticSolver::computeSecondaryFields(const ParGridFunction
// &state)
// {
//    // *out << "before curl constructed\n";
//    DiscreteCurlOperator curl(fes.get(), B->ParFESpace());
//    // *out << "curl constructed\n";
//    curl.Assemble();
//    curl.Finalize();
//    curl.Mult(state, *B);
//    *out << "secondary quantities computed\n";
// }

// // /// TODO: Find a better way to handle solving the simple box problem
// // void MagnetostaticSolver::phase_a_source(const Vector &x,
// //                                        Vector &J)
// // {
// //    // example of needed geometric parameters, this should be all you
// need
// //    int n_s = 12; //number of slots
// //    double zb = 0.0; //bottom of stator
// //    double zt = 0.25; //top of stator

// //    // compute r and theta from x and y
// //    // double r = sqrt(x(0)*x(0) + x(1)*x(1)); (r not needed)
// //    double tha = atan2(x(1), x(0));
// //    double th;

// //    double thw = 2*M_PI/n_s; //total angle of slot
// //    int w; //current slot
// //    J = 0.0;

// //    // check which winding we're in
// //    th = remquo(tha, thw, &w);

// //    // check if we're in the stator body
// //    if(x(2) >= zb && x(2) <= zt)
// //    {
// //       // check if we're in left or right half
// //       if(th > 0)
// //       {
// //          J(2) = -1; // set to 1 for now, and direction depends on
// current
// //          direction
// //       }
// //       if(th < 0)
// //       {
// //          J(2) = 1;
// //       }
// //    }
// //    else  // outside of the stator body, check if above or below
// //    {
// //       // 'subtract' z position to 0 depending on if above or below
// //       mfem::Vector rx(x);
// //       if(x(2) > zt)
// //       {
// //          rx(2) -= zt;
// //       }
// //       if(x(2) < zb)
// //       {
// //          rx(2) -= zb;
// //       }

// //       // draw top rotation axis
// //       mfem::Vector ax(3);
// //       mfem::Vector Jr(3);
// //       ax = 0.0;
// //       ax(0) = cos(w*thw);
// //       ax(1) = sin(w*thw);

// //       // take x cross ax, normalize
// //       Jr(0) = rx(1)*ax(2) - rx(2)*ax(1);
// //       Jr(1) = rx(2)*ax(0) - rx(0)*ax(2);
// //       Jr(2) = rx(0)*ax(1) - rx(1)*ax(0);
// //       Jr /= Jr.Norml2();
// //       J = Jr;
// //    }
// //    J *= current_density * fill_factor;
// // }

// // void MagnetostaticSolver::phase_b_source(const Vector &x,
// //                                        Vector &J)
// // {
// //    // example of needed geometric parameters, this should be all you
// need
// //    int n_s = 12; //number of slots
// //    double zb = 0.0; //bottom of stator
// //    double zt = 0.25; //top of stator

// //    // compute r and theta from x and y
// //    // double r = sqrt(x(0)*x(0) + x(1)*x(1)); (r not needed)
// //    double tha = atan2(x(1), x(0));
// //    double th;

// //    double thw = 2*M_PI/n_s; //total angle of slot
// //    int w; //current slot
// //    J = 0.0;

// //    // check which winding we're in
// //    th = remquo(tha, thw, &w);

// //    // check if we're in the stator body
// //    if(x(2) >= zb && x(2) <= zt)
// //    {
// //       // check if we're in left or right half
// //       if(th > 0)
// //       {
// //          J(2) = -1; // set to 1 for now, and direction depends on
// current
// //          direction
// //       }
// //       if(th < 0)
// //       {
// //          J(2) = 1;
// //       }
// //    }
// //    else  // outside of the stator body, check if above or below
// //    {
// //       // 'subtract' z position to 0 depending on if above or below
// //       mfem::Vector rx(x);
// //       if(x(2) > zt)
// //       {
// //          rx(2) -= zt;
// //       }
// //       if(x(2) < zb)
// //       {
// //          rx(2) -= zb;
// //       }

// //       // draw top rotation axis
// //       mfem::Vector ax(3);
// //       mfem::Vector Jr(3);
// //       ax = 0.0;
// //       ax(0) = cos(w*thw);
// //       ax(1) = sin(w*thw);

// //       // take x cross ax, normalize
// //       Jr(0) = rx(1)*ax(2) - rx(2)*ax(1);
// //       Jr(1) = rx(2)*ax(0) - rx(0)*ax(2);
// //       Jr(2) = rx(0)*ax(1) - rx(1)*ax(0);
// //       Jr /= Jr.Norml2();
// //       J = Jr;
// //    }
// //    J *= -current_density * fill_factor;
// // }

// // void MagnetostaticSolver::phase_c_source(const Vector &x,
// //                                        Vector &J)
// // {
// //    J.SetSize(3);
// //    J = 0.0;
// //    // Vector r = x;
// //    // r(2) = 0.0;
// //    // r /= r.Norml2();
// //    // J(0) = -r(1);
// //    // J(1) = r(0);
// //    // J *= current_density;
// // }

// // /// TODO: Find a better way to handle solving the simple box problem
// // /// TODO: implement other kinds of sources
// // void MagnetostaticSolver::magnetization_source_north(const Vector &x,
// //                                                    Vector &M)
// // {
// //    Vector plane_vec = x;
// //    plane_vec(2) = 0;
// //    M = plane_vec;
// //    M /= M.Norml2();
// //    M *= remnant_flux;
// // }

// // void MagnetostaticSolver::magnetization_source_south(const Vector &x,
// //                                                    Vector &M)
// // {
// //    Vector plane_vec = x;
// //    plane_vec(2) = 0;
// //    M = plane_vec;
// //    M /= M.Norml2();
// //    M *= -remnant_flux;

// //    // M = 0.0;
// //    // M(2) = remnant_flux;
// // }

// // void MagnetostaticSolver::x_axis_current_source(const Vector &x,
// //                                                 Vector &J)
// // {
// //    J.SetSize(3);
// //    J = 0.0;
// //    J(0) = current_density;
// // }

// // void MagnetostaticSolver::y_axis_current_source(const Vector &x,
// //                                                 Vector &J)
// // {
// //    J.SetSize(3);
// //    J = 0.0;
// //    J(1) = current_density;
// // }

// // void MagnetostaticSolver::z_axis_current_source(const Vector &x,
// //                                                 Vector &J)
// // {
// //    J.SetSize(3);
// //    J = 0.0;
// //    J(2) = current_density;
// // }

// // void MagnetostaticSolver::ring_current_source(const Vector &x,
// //                                               Vector &J)
// // {
// //    J.SetSize(3);
// //    J = 0.0;
// //    Vector r = x;
// //    r(2) = 0.0;
// //    r /= r.Norml2();
// //    J(0) = -r(1);
// //    J(1) = r(0);
// //    J *= current_density;
// // }

// // void MagnetostaticSolver::x_axis_magnetization_source(const Vector &x,
// //                                                       Vector &M)
// // {
// //    M.SetSize(3);
// //    M = 0.0;
// //    M(0) = remnant_flux;
// // }

// // void MagnetostaticSolver::y_axis_magnetization_source(const Vector &x,
// //                                                       Vector &M)
// // {
// //    M.SetSize(3);
// //    M = 0.0;
// //    M(1) = remnant_flux;
// // }

// // void MagnetostaticSolver::z_axis_magnetization_source(const Vector &x,
// //                                                       Vector &M)
// // {
// //    M.SetSize(3);
// //    M = 0.0;
// //    M(2) = remnant_flux;
// // }

// void MagnetostaticSolver::phaseACurrentSource(const Vector &x, Vector &J)
// {
//    phase_a_current(24.0, 0.03450, x.GetData(), J.GetData());
// }

// void MagnetostaticSolver::phaseACurrentSourceRevDiff(const Vector &x,
//                                                      const Vector &V_bar,
//                                                      Vector &x_bar)
// {
//    DenseMatrix source_jac(3);
//    // declare vectors of active input variables
//    std::vector<adouble> x_a(x.Size());
//    // copy data from mfem::Vector
//    adept::set_values(x_a.data(), x.Size(), x.GetData());
//    // start recording
//    diff_stack.new_recording();
//    // the depedent variable must be declared after the recording
//    std::vector<adouble> J_a(x.Size());
//    phase_a_current<adouble>(24.0, 0.03450, x_a.data(), J_a.data());
//    // set the independent and dependent variable
//    diff_stack.independent(x_a.data(), x.Size());
//    diff_stack.dependent(J_a.data(), x.Size());
//    // calculate the jacobian w.r.t position
//    diff_stack.jacobian(source_jac.GetData());
//    source_jac.MultTranspose(V_bar, x_bar);
// }

// void MagnetostaticSolver::phaseBCurrentSource(const Vector &x, Vector &J)
// {
//    phase_b_current(24.0, 0.03450, x.GetData(), J.GetData());
// }

// void MagnetostaticSolver::phaseBCurrentSourceRevDiff(const Vector &x,
//                                                      const Vector &V_bar,
//                                                      Vector &x_bar)
// {
//    DenseMatrix source_jac(3);
//    // declare vectors of active input variables
//    std::vector<adouble> x_a(x.Size());
//    // copy data from mfem::Vector
//    adept::set_values(x_a.data(), x.Size(), x.GetData());
//    // start recording
//    diff_stack.new_recording();
//    // the depedent variable must be declared after the recording
//    std::vector<adouble> J_a(x.Size());
//    phase_b_current<adouble>(24.0, 0.03450, x_a.data(), J_a.data());
//    // set the independent and dependent variable
//    diff_stack.independent(x_a.data(), x.Size());
//    diff_stack.dependent(J_a.data(), x.Size());
//    // calculate the jacobian w.r.t position
//    diff_stack.jacobian(source_jac.GetData());
//    source_jac.MultTranspose(V_bar, x_bar);
// }

// void MagnetostaticSolver::phaseCCurrentSource(const Vector &x, Vector &J)
// {
//    phase_c_current(24.0, 0.03450, x.GetData(), J.GetData());
// }

// void MagnetostaticSolver::phaseCCurrentSourceRevDiff(const Vector &x,
//                                                      const Vector &V_bar,
//                                                      Vector &x_bar)
// {
//    DenseMatrix source_jac(3);
//    // declare vectors of active input variables
//    std::vector<adouble> x_a(x.Size());
//    // copy data from mfem::Vector
//    adept::set_values(x_a.data(), x.Size(), x.GetData());
//    // start recording
//    diff_stack.new_recording();
//    // the depedent variable must be declared after the recording
//    std::vector<adouble> J_a(x.Size());
//    phase_c_current<adouble>(24.0, 0.03450, x_a.data(), J_a.data());
//    // set the independent and dependent variable
//    diff_stack.independent(x_a.data(), x.Size());
//    diff_stack.dependent(J_a.data(), x.Size());
//    // calculate the jacobian w.r.t position
//    diff_stack.jacobian(source_jac.GetData());
//    source_jac.MultTranspose(V_bar, x_bar);
// }

// void MagnetostaticSolver::northMagnetizationSource(const Vector &x, Vector
// &M)
// {
//    north_magnetization(remnant_flux, x.GetData(), M.GetData());
// }

// void MagnetostaticSolver::northMagnetizationSourceRevDiff(const Vector &x,
//                                                           const Vector
//                                                           &V_bar, Vector
//                                                           &x_bar)
// {
//    DenseMatrix source_jac(3);
//    // declare vectors of active input variables
//    std::vector<adouble> x_a(x.Size());
//    // copy data from mfem::Vector
//    adept::set_values(x_a.data(), x.Size(), x.GetData());
//    // start recording
//    diff_stack.new_recording();
//    // the depedent variable must be declared after the recording
//    std::vector<adouble> M_a(x.Size());
//    north_magnetization<adouble>(remnant_flux, x_a.data(), M_a.data());
//    // set the independent and dependent variable
//    diff_stack.independent(x_a.data(), x.Size());
//    diff_stack.dependent(M_a.data(), x.Size());
//    // calculate the jacobian w.r.t position
//    diff_stack.jacobian(source_jac.GetData());
//    source_jac.MultTranspose(V_bar, x_bar);
// }

// void MagnetostaticSolver::southMagnetizationSource(const Vector &x, Vector
// &M)
// {
//    south_magnetization(remnant_flux, x.GetData(), M.GetData());
// }

// void MagnetostaticSolver::southMagnetizationSourceRevDiff(const Vector &x,
//                                                           const Vector
//                                                           &V_bar, Vector
//                                                           &x_bar)
// {
//    DenseMatrix source_jac(3);
//    // declare vectors of active input variables
//    std::vector<adouble> x_a(x.Size());
//    // copy data from mfem::Vector
//    adept::set_values(x_a.data(), x.Size(), x.GetData());
//    // start recording
//    diff_stack.new_recording();
//    // the depedent variable must be declared after the recording
//    std::vector<adouble> M_a(x.Size());
//    south_magnetization<adouble>(remnant_flux, x_a.data(), M_a.data());
//    // set the independent and dependent variable
//    diff_stack.independent(x_a.data(), x.Size());
//    diff_stack.dependent(M_a.data(), x.Size());
//    // calculate the jacobian w.r.t position
//    diff_stack.jacobian(source_jac.GetData());
//    source_jac.MultTranspose(V_bar, x_bar);
// }

// void MagnetostaticSolver::cwMagnetizationSource(const Vector &x, Vector
// &M)
// {
//    cw_magnetization(remnant_flux, x.GetData(), M.GetData());
// }

// void MagnetostaticSolver::cwMagnetizationSourceRevDiff(const Vector &x,
//                                                        const Vector
//                                                        &V_bar, Vector
//                                                        &x_bar)
// {
//    DenseMatrix source_jac(3);
//    // declare vectors of active input variables
//    std::vector<adouble> x_a(x.Size());
//    // copy data from mfem::Vector
//    adept::set_values(x_a.data(), x.Size(), x.GetData());
//    // start recording
//    diff_stack.new_recording();
//    // the depedent variable must be declared after the recording
//    std::vector<adouble> M_a(x.Size());
//    cw_magnetization<adouble>(remnant_flux, x_a.data(), M_a.data());
//    // set the independent and dependent variable
//    diff_stack.independent(x_a.data(), x.Size());
//    diff_stack.dependent(M_a.data(), x.Size());
//    // calculate the jacobian w.r.t position
//    diff_stack.jacobian(source_jac.GetData());
//    source_jac.MultTranspose(V_bar, x_bar);
// }

// void MagnetostaticSolver::ccwMagnetizationSource(const Vector &x, Vector
// &M)
// {
//    ccw_magnetization(remnant_flux, x.GetData(), M.GetData());
// }

// void MagnetostaticSolver::ccwMagnetizationSourceRevDiff(const Vector &x,
//                                                         const Vector
//                                                         &V_bar, Vector
//                                                         &x_bar)
// {
//    DenseMatrix source_jac(3);
//    // declare vectors of active input variables
//    std::vector<adouble> x_a(x.Size());
//    // copy data from mfem::Vector
//    adept::set_values(x_a.data(), x.Size(), x.GetData());
//    // start recording
//    diff_stack.new_recording();
//    // the depedent variable must be declared after the recording
//    std::vector<adouble> M_a(x.Size());
//    ccw_magnetization<adouble>(remnant_flux, x_a.data(), M_a.data());
//    // set the independent and dependent variable
//    diff_stack.independent(x_a.data(), x.Size());
//    diff_stack.dependent(M_a.data(), x.Size());
//    // calculate the jacobian w.r.t position
//    diff_stack.jacobian(source_jac.GetData());
//    source_jac.MultTranspose(V_bar, x_bar);
// }

// void MagnetostaticSolver::xAxisCurrentSource(const Vector &x, Vector &J)
// {
//    x_axis_current(x.GetData(), J.GetData());
// }

// void MagnetostaticSolver::xAxisCurrentSourceRevDiff(const Vector &x,
//                                                     const Vector &V_bar,
//                                                     Vector &x_bar)
// {
//    DenseMatrix source_jac(3);
//    // declare vectors of active input variables
//    std::vector<adouble> x_a(x.Size());
//    // copy data from mfem::Vector
//    adept::set_values(x_a.data(), x.Size(), x.GetData());
//    // start recording
//    diff_stack.new_recording();
//    // the depedent variable must be declared after the recording
//    std::vector<adouble> J_a(x.Size());
//    x_axis_current<adouble>(x_a.data(), J_a.data());
//    // set the independent and dependent variable
//    diff_stack.independent(x_a.data(), x.Size());
//    diff_stack.dependent(J_a.data(), x.Size());
//    // calculate the jacobian w.r.t state vaiables
//    diff_stack.jacobian(source_jac.GetData());
//    source_jac.MultTranspose(V_bar, x_bar);
// }

// void MagnetostaticSolver::yAxisCurrentSource(const Vector &x, Vector &J)
// {
//    y_axis_current(x.GetData(), J.GetData());
// }

// void MagnetostaticSolver::yAxisCurrentSourceRevDiff(const Vector &x,
//                                                     const Vector &V_bar,
//                                                     Vector &x_bar)
// {
//    DenseMatrix source_jac(3);
//    // declare vectors of active input variables
//    std::vector<adouble> x_a(x.Size());
//    // copy data from mfem::Vector
//    adept::set_values(x_a.data(), x.Size(), x.GetData());
//    // start recording
//    diff_stack.new_recording();
//    // the depedent variable must be declared after the recording
//    std::vector<adouble> J_a(x.Size());
//    y_axis_current<adouble>(x_a.data(), J_a.data());
//    // set the independent and dependent variable
//    diff_stack.independent(x_a.data(), x.Size());
//    diff_stack.dependent(J_a.data(), x.Size());
//    // calculate the jacobian w.r.t state vaiables
//    diff_stack.jacobian(source_jac.GetData());
//    source_jac.MultTranspose(V_bar, x_bar);
// }

// void MagnetostaticSolver::zAxisCurrentSource(const Vector &x, Vector &J)
// {
//    z_axis_current(x.GetData(), J.GetData());
// }

// void MagnetostaticSolver::zAxisCurrentSourceRevDiff(const Vector &x,
//                                                     const Vector &V_bar,
//                                                     Vector &x_bar)
// {
//    DenseMatrix source_jac(3);
//    // declare vectors of active input variables
//    std::vector<adouble> x_a(x.Size());
//    // copy data from mfem::Vector
//    adept::set_values(x_a.data(), x.Size(), x.GetData());
//    // start recording
//    diff_stack.new_recording();
//    // the depedent variable must be declared after the recording
//    std::vector<adouble> J_a(x.Size());
//    z_axis_current<adouble>(x_a.data(), J_a.data());
//    // set the independent and dependent variable
//    diff_stack.independent(x_a.data(), x.Size());
//    diff_stack.dependent(J_a.data(), x.Size());
//    // calculate the jacobian w.r.t state vaiables
//    diff_stack.jacobian(source_jac.GetData());
//    source_jac.MultTranspose(V_bar, x_bar);
// }

// void MagnetostaticSolver::nzAxisCurrentSource(const Vector &x, Vector &J)
// {
//    z_axis_current<double, -1>(x.GetData(), J.GetData());
// }

// void MagnetostaticSolver::nzAxisCurrentSourceRevDiff(const Vector &x,
//                                                      const Vector &V_bar,
//                                                      Vector &x_bar)
// {
//    DenseMatrix source_jac(3);
//    // declare vectors of active input variables
//    std::vector<adouble> x_a(x.Size());
//    // copy data from mfem::Vector
//    adept::set_values(x_a.data(), x.Size(), x.GetData());
//    // start recording
//    diff_stack.new_recording();
//    // the depedent variable must be declared after the recording
//    std::vector<adouble> J_a(x.Size());
//    z_axis_current<adouble, -1>(x_a.data(), J_a.data());
//    // set the independent and dependent variable
//    diff_stack.independent(x_a.data(), x.Size());
//    diff_stack.dependent(J_a.data(), x.Size());
//    // calculate the jacobian w.r.t state vaiables
//    diff_stack.jacobian(source_jac.GetData());
//    source_jac.MultTranspose(V_bar, x_bar);
// }

// void MagnetostaticSolver::ringCurrentSource(const Vector &x, Vector &J)
// {
//    ring_current(x.GetData(), J.GetData());
// }

// void MagnetostaticSolver::ringCurrentSourceRevDiff(const Vector &x,
//                                                    const Vector &V_bar,
//                                                    Vector &x_bar)
// {
//    DenseMatrix source_jac(3);
//    // declare vectors of active input variables
//    std::vector<adouble> x_a(x.Size());
//    // copy data from mfem::Vector
//    adept::set_values(x_a.data(), x.Size(), x.GetData());
//    // start recording
//    diff_stack.new_recording();
//    // the depedent variable must be declared after the recording
//    std::vector<adouble> J_a(x.Size());
//    ring_current<adouble>(x_a.data(), J_a.data());
//    // set the independent and dependent variable
//    diff_stack.independent(x_a.data(), x.Size());
//    diff_stack.dependent(J_a.data(), x.Size());
//    // calculate the jacobian w.r.t state vaiables
//    diff_stack.jacobian(source_jac.GetData());
//    source_jac.MultTranspose(V_bar, x_bar);
// }

// void MagnetostaticSolver::xAxisMagnetizationSource(const Vector &x, Vector
// &M)
// {
//    x_axis_magnetization(remnant_flux, x.GetData(), M.GetData());
// }

// void MagnetostaticSolver::xAxisMagnetizationSourceRevDiff(const Vector &x,
//                                                           const Vector
//                                                           &V_bar, Vector
//                                                           &x_bar)
// {
//    DenseMatrix source_jac(3);
//    // declare vectors of active input variables
//    std::vector<adouble> x_a(x.Size());
//    // copy data from mfem::Vector
//    adept::set_values(x_a.data(), x.Size(), x.GetData());
//    // start recording
//    diff_stack.new_recording();
//    // the depedent variable must be declared after the recording
//    std::vector<adouble> M_a(x.Size());
//    x_axis_magnetization<adouble>(remnant_flux, x_a.data(), M_a.data());
//    // set the independent and dependent variable
//    diff_stack.independent(x_a.data(), x.Size());
//    diff_stack.dependent(M_a.data(), x.Size());
//    // calculate the jacobian w.r.t state vaiables
//    diff_stack.jacobian(source_jac.GetData());
//    source_jac.MultTranspose(V_bar, x_bar);
// }

// void MagnetostaticSolver::yAxisMagnetizationSource(const Vector &x, Vector
// &M)
// {
//    y_axis_magnetization(remnant_flux, x.GetData(), M.GetData());
// }

// void MagnetostaticSolver::yAxisMagnetizationSourceRevDiff(const Vector &x,
//                                                           const Vector
//                                                           &V_bar, Vector
//                                                           &x_bar)
// {
//    DenseMatrix source_jac(3);
//    // declare vectors of active input variables
//    std::vector<adouble> x_a(x.Size());
//    // copy data from mfem::Vector
//    adept::set_values(x_a.data(), x.Size(), x.GetData());
//    // start recording
//    diff_stack.new_recording();
//    // the depedent variable must be declared after the recording
//    std::vector<adouble> M_a(x.Size());
//    y_axis_magnetization<adouble>(remnant_flux, x_a.data(), M_a.data());
//    // set the independent and dependent variable
//    diff_stack.independent(x_a.data(), x.Size());
//    diff_stack.dependent(M_a.data(), x.Size());
//    // calculate the jacobian w.r.t state vaiables
//    diff_stack.jacobian(source_jac.GetData());
//    source_jac.MultTranspose(V_bar, x_bar);
// }

// void MagnetostaticSolver::zAxisMagnetizationSource(const Vector &x, Vector
// &M)
// {
//    z_axis_magnetization(remnant_flux, x.GetData(), M.GetData());
// }

// void MagnetostaticSolver::zAxisMagnetizationSourceRevDiff(const Vector &x,
//                                                           const Vector
//                                                           &V_bar, Vector
//                                                           &x_bar)
// {
//    DenseMatrix source_jac(3);
//    // declare vectors of active input variables
//    std::vector<adouble> x_a(x.Size());
//    // copy data from mfem::Vector
//    adept::set_values(x_a.data(), x.Size(), x.GetData());
//    // start recording
//    diff_stack.new_recording();
//    // the depedent variable must be declared after the recording
//    std::vector<adouble> M_a(x.Size());
//    z_axis_magnetization<adouble>(remnant_flux, x_a.data(), M_a.data());
//    // set the independent and dependent variable
//    diff_stack.independent(x_a.data(), x.Size());
//    diff_stack.dependent(M_a.data(), x.Size());
//    // calculate the jacobian w.r.t state vaiables
//    diff_stack.jacobian(source_jac.GetData());
//    source_jac.MultTranspose(V_bar, x_bar);
// }

// void MagnetostaticSolver::box1CurrentSource(const Vector &x, Vector &J)
// {
//    box1_current(x.GetData(), J.GetData());
// }

// void MagnetostaticSolver::box1CurrentSourceRevDiff(const Vector &x,
//                                                    const Vector &V_bar,
//                                                    Vector &x_bar)
// {
//    DenseMatrix source_jac(3);
//    // declare vectors of active input variables
//    std::vector<adouble> x_a(x.Size());
//    // copy data from mfem::Vector
//    adept::set_values(x_a.data(), x.Size(), x.GetData());
//    // start recording
//    diff_stack.new_recording();
//    // the depedent variable must be declared after the recording
//    std::vector<adouble> J_a(x.Size());
//    box1_current<adouble>(x_a.data(), J_a.data());
//    // set the independent and dependent variable
//    diff_stack.independent(x_a.data(), x.Size());
//    diff_stack.dependent(J_a.data(), x.Size());
//    // calculate the jacobian w.r.t state vaiables
//    diff_stack.jacobian(source_jac.GetData());
//    source_jac.MultTranspose(V_bar, x_bar);
// }

// void MagnetostaticSolver::box2CurrentSource(const Vector &x, Vector &J)
// {
//    box2_current(x.GetData(), J.GetData());
// }

// void MagnetostaticSolver::box2CurrentSourceRevDiff(const Vector &x,
//                                                    const Vector &V_bar,
//                                                    Vector &x_bar)
// {
//    DenseMatrix source_jac(3);
//    // declare vectors of active input variables
//    std::vector<adouble> x_a(x.Size());
//    // copy data from mfem::Vector
//    adept::set_values(x_a.data(), x.Size(), x.GetData());
//    // start recording
//    diff_stack.new_recording();
//    // the depedent variable must be declared after the recording
//    std::vector<adouble> J_a(x.Size());
//    box2_current<adouble>(x_a.data(), J_a.data());
//    // set the independent and dependent variable
//    diff_stack.independent(x_a.data(), x.Size());
//    diff_stack.dependent(J_a.data(), x.Size());
//    // calculate the jacobian w.r.t state vaiables
//    diff_stack.jacobian(source_jac.GetData());
//    source_jac.MultTranspose(V_bar, x_bar);
// }

// void MagnetostaticSolver::team13CurrentSource(const Vector &x, Vector &J)
// {
//    team13_current(x.GetData(), J.GetData());
// }

// void MagnetostaticSolver::team13CurrentSourceRevDiff(const Vector &x,
//                                                      const Vector &V_bar,
//                                                      Vector &x_bar)
// {
//    DenseMatrix source_jac(3);
//    // declare vectors of active input variables
//    std::vector<adouble> x_a(x.Size());
//    // copy data from mfem::Vector
//    adept::set_values(x_a.data(), x.Size(), x.GetData());
//    // start recording
//    diff_stack.new_recording();
//    // the depedent variable must be declared after the recording
//    std::vector<adouble> J_a(x.Size());
//    team13_current<adouble>(x_a.data(), J_a.data());
//    // set the independent and dependent variable
//    diff_stack.independent(x_a.data(), x.Size());
//    diff_stack.dependent(J_a.data(), x.Size());
//    // calculate the jacobian w.r.t state vaiables
//    diff_stack.jacobian(source_jac.GetData());
//    source_jac.MultTranspose(V_bar, x_bar);
// }

// void MagnetostaticSolver::a_exact(const Vector &x, Vector &A)
// {
//    A.SetSize(3);
//    A = 0.0;
//    double y = x(1) - .5;
//    if (x(1) <= .5)
//    {
//       A(2) = y * y * y;
//       // A(2) = y*y;
//    }
//    else
//    {
//       A(2) = -y * y * y;
//       // A(2) = -y*y;
//    }
// }

// void MagnetostaticSolver::b_exact(const Vector &x, Vector &B)
// {
//    B.SetSize(3);
//    B = 0.0;
//    double y = x(1) - .5;
//    if (x(1) <= .5)
//    {
//       B(0) = 3 * y * y;
//       // B(0) = 2*y;
//    }
//    else
//    {
//       B(0) = -3 * y * y;
//       // B(0) = -2*y;
//    }
// }

// double MagnetostaticSolver::remnant_flux = 0.0;
// double MagnetostaticSolver::mag_mu_r = 0.0;

// void setInputs(MagnetostaticLoad &load, const MISOInputs &inputs)
// {
//    setInputs(load.current_load, inputs);
//    setInputs(load.magnetic_load, inputs);
// }

// void setOptions(MagnetostaticLoad &load, const nlohmann::json &options)
// {
//    setOptions(load.current_load, options);
//    setOptions(load.magnetic_load, options);
// }

// void addLoad(MagnetostaticLoad &load, mfem::Vector &tv)
// {
//    addLoad(load.current_load, tv);
//    addLoad(load.magnetic_load, tv);
// }

// double vectorJacobianProduct(MagnetostaticLoad &load,
//                              const mfem::Vector &res_bar,
//                              const std::string &wrt)
// {
//    double wrt_bar = 0.0;
//    wrt_bar += vectorJacobianProduct(load.current_load, res_bar, wrt);
//    wrt_bar += vectorJacobianProduct(load.magnetic_load, res_bar, wrt);
//    return wrt_bar;
// }

// void vectorJacobianProduct(MagnetostaticLoad &load,
//                            const mfem::Vector &res_bar,
//                            const std::string &wrt,
//                            mfem::Vector &wrt_bar)
// {
//    vectorJacobianProduct(load.current_load, res_bar, wrt, wrt_bar);
//    vectorJacobianProduct(load.magnetic_load, res_bar, wrt, wrt_bar);
// }

// }  // namespace miso
