#include <cmath>
#include <memory>
#include "euler_dg_cut_sens_test.hpp"
#include "cut_quad_poly.hpp"
using namespace std::chrono;
using namespace mfem;
using namespace std;
#include <random>
static std::default_random_engine gen;
static std::uniform_real_distribution<double> uniform_rand(-1.0, 1.0);
namespace mach
{
template <int dim, bool entvar>
CutEulerDGSensitivityTestSolver<dim, entvar>::CutEulerDGSensitivityTestSolver(
    const nlohmann::json &json_options,
    unique_ptr<mfem::Mesh> smesh,
    MPI_Comm comm)
 : AbstractSolver(json_options, move(smesh), comm)
{
   if (entvar)
   {
      *out << "The state variables are the entropy variables." << endl;
   }
   else
   {
      *out << "The state variables are the conservative variables." << endl;
   }
   // define free-stream parameters; may or may not be used, depending on case
   mach_fs = options["flow-param"]["mach"].template get<double>();
   aoa_fs = options["flow-param"]["aoa"].template get<double>() * M_PI / 180;
   iroll = options["flow-param"]["roll-axis"].template get<int>();
   ipitch = options["flow-param"]["pitch-axis"].template get<int>();
   if (iroll == ipitch)
   {
      throw MachException("iroll and ipitch must be distinct dimensions!");
   }
   if ((iroll < 0) || (iroll > 2))
   {
      throw MachException("iroll axis must be between 0 and 2!");
   }
   if ((ipitch < 0) || (ipitch > 2))
   {
      throw MachException("ipitch axis must be between 0 and 2!");
   }
   /// int rules for cut elements
   cout << "#elements " << mesh->GetNE() << endl;
   ofstream sol_ofs("cart_mesh_dg_cut_airfoil.vtk");
   sol_ofs.precision(14);
   mesh->PrintVTK(sol_ofs, 0);
   int order = options["space-dis"]["degree"].template get<int>();
   int deg_vol = min((order + 2) * (order + 2), 10);
   int deg_surf = min(((order + 2) * (order + 2)), 10);
   // deg_vol = max(10, 2 * order);
   // deg_surf = max(10, 2 * order);
   // deg_vol = 2*order;
   // deg_surf = 2*order;
   deg_vol = max(10, 2 * order);
   deg_surf = max(10, 2 * order);
   int poly_deg = 3;
   double circ_rad = 0.50;
   delta = 1e-05;
   CutCell<duald, 2, 1> cutcell(circ_rad, mesh.get());
   CutCell<double, 2, 2> cutcell2(circ_rad, mesh.get());
   CutCell<double, 2, 1> cutcell_p(circ_rad + delta, mesh.get());
   CutCell<double, 2, 1> cutcell_m(circ_rad - delta, mesh.get());
   phi = cutcell.constructLevelSet<double>();
   phi_p = cutcell_p.constructLevelSet<double>();
   phi_m = cutcell_m.constructLevelSet<double>();
   vector<int> cutelems_inner;
   vector<int> solid_elements;
   /// find the elements for which we don't need to solve
   for (int i = 0; i < mesh->GetNE(); ++i)
   {
      /// at r
      if (cutcell.cutByGeom(i) == true)
      {
         cutelems.push_back(i);
         cutElements.push_back(true);
      }
      else
      {
         cutElements.push_back(false);
      }
      if ((cutcell.insideBoundary(i) == true))
      {
         embeddedElements.push_back(true); /*original*/
      }
      else
      {
         embeddedElements.push_back(false);
      }
      /// r + delta
      if (cutcell_p.cutByGeom(i) == true)
      {
         cutelems_p.push_back(i);
         cutElements_p.push_back(true);
      }
      else
      {
         cutElements_p.push_back(false);
      }
      if ((cutcell_p.insideBoundary(i) == true))
      {
         embeddedElements_p.push_back(true); /*original*/
      }
      else
      {
         embeddedElements_p.push_back(false);
      }
      /// r - delta
      if (cutcell_m.cutByGeom(i) == true)
      {
         cutelems_m.push_back(i);
         cutElements_m.push_back(true);
      }
      else
      {
         cutElements_m.push_back(false);
      }
      if ((cutcell_m.insideBoundary(i) == true))
      {
         embeddedElements_m.push_back(true); /*original*/
      }
      else
      {
         embeddedElements_m.push_back(false);
      }
   }
   /// find if given face is cut, or immersed

   /// find faces cut by inner circle
   for (int i = 0; i < mesh->GetNumFaces(); ++i)
   {
      FaceElementTransformations *tr;
      // tr = mesh->GetInteriorFaceTransformations(i);
      tr = mesh->GetFaceElementTransformations(i);
      if (tr->Elem2No >= 0)
      {
         /// r
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
         /// r + delta
         if ((find(cutelems_p.begin(), cutelems_p.end(), tr->Elem1No) !=
              cutelems_p.end()) &&
             (find(cutelems_p.begin(), cutelems_p.end(), tr->Elem2No) !=
              cutelems_p.end()))
         {
            if (!cutcell_p.findImmersedFace(tr->Face->ElementNo))
            {
               cutInteriorFaces_p.push_back(tr->Face->ElementNo);
            }
         }
         /// r - delta
         if ((find(cutelems_m.begin(), cutelems_m.end(), tr->Elem1No) !=
              cutelems_m.end()) &&
             (find(cutelems_m.begin(), cutelems_m.end(), tr->Elem2No) !=
              cutelems_m.end()))
         {
            if (!cutcell_m.findImmersedFace(tr->Face->ElementNo))
            {
               cutInteriorFaces_m.push_back(tr->Face->ElementNo);
            }
         }
      }
   }
   for (int i = 0; i < mesh->GetNumFaces(); ++i)
   {
      FaceElementTransformations *tr;
      tr = mesh->GetInteriorFaceTransformations(i);
      if (tr != NULL)
      {
         /// r
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
         /// r + delta
         if ((embeddedElements_p.at(tr->Elem1No) == true) ||
             (embeddedElements_p.at(tr->Elem2No)) == true)
         {
            immersedFaces_p[tr->Face->ElementNo] = true;
         }
         else if (cutcell_p.findImmersedFace(tr->Face->ElementNo))
         {
            immersedFaces_p[tr->Face->ElementNo] = true;
         }
         else
         {
            // cout << "immersed Face element is: " << tr->Elem1No << endl;
            immersedFaces_p[tr->Face->ElementNo] = false;
         }
         /// r - delta
         if ((embeddedElements_m.at(tr->Elem1No) == true) ||
             (embeddedElements_m.at(tr->Elem2No)) == true)
         {
            immersedFaces_m[tr->Face->ElementNo] = true;
         }
         else if (cutcell_m.findImmersedFace(tr->Face->ElementNo))
         {
            immersedFaces_m[tr->Face->ElementNo] = true;
         }
         else
         {
            // cout << "immersed Face element is: " << tr->Elem1No << endl;
            immersedFaces_m[tr->Face->ElementNo] = false;
         }
      }
   }
   double radius = 0.3;
   /// int rule for cut elements
   auto elint_start = high_resolution_clock::now();
   /// r
   cutcell.GetCutElementIntRule(cutelems,
                                deg_vol,
                                poly_deg,
                                cutSquareIntRules,
                                cutSquareIntRules_sens,
                                cutSegmentIntRules,
                                cutSegmentIntRules_sens);
   cutcell.GetCutInterfaceIntRule(cutelems,
                                  cutInteriorFaces,
                                  deg_vol,
                                  cutInteriorFaceIntRules,
                                  cutInteriorFaceIntRules_sens);
   /// r + delta
   cutcell_p.GetCutElementIntRule(cutelems_p,
                                  deg_vol,
                                  poly_deg,
                                  cutSquareIntRules_p,
                                  cutSquareIntRules_sens_p,
                                  cutSegmentIntRules_p,
                                  cutSegmentIntRules_sens_p);
   cutcell_p.GetCutInterfaceIntRule(cutelems_p,
                                    cutInteriorFaces_p,
                                    deg_vol,
                                    cutInteriorFaceIntRules_p,
                                    cutInteriorFaceIntRules_sens_p);
   /// r - delta
   cutcell_m.GetCutElementIntRule(cutelems_m,
                                  deg_vol,
                                  poly_deg,
                                  cutSquareIntRules_m,
                                  cutSquareIntRules_sens_m,
                                  cutSegmentIntRules_m,
                                  cutSegmentIntRules_sens_m);
   cutcell_m.GetCutInterfaceIntRule(cutelems_m,
                                    cutInteriorFaces_m,
                                    deg_vol,
                                    cutInteriorFaceIntRules_m,
                                    cutInteriorFaceIntRules_sens_m);

   auto elint_stop = high_resolution_clock::now();
   auto elint_duration = duration_cast<seconds>(elint_stop - elint_start);
   cout << " ---- Time taken to get cut elements int rules  ---- " << endl;
   cout << "      " << elint_duration.count() << "s " << endl;
   /// int rule for cut boundaries and interior faces
   // get interior face int rule that is cut by the embedded geometry
   auto segint_start = high_resolution_clock::now();
   cout << "#interior faces " << cutInteriorFaces.size() << endl;
   int fe_order = options["space-dis"]["degree"].template get<int>();
   gd = options["space-dis"].value("GD", false);
}
template <int dim, bool entvar>
void CutEulerDGSensitivityTestSolver<dim, entvar>::setGDSpace(int fe_order)
{
   fes_gd.reset(new GDSpaceType(mesh.get(),
                                fec.get(),
                                embeddedElements,
                                cutElements,
                                num_state,
                                Ordering::byVDIM,
                                fe_order,
                                comm));
   cout << "Inside gd stuff " << endl;
   fes_gd_p.reset(new mfem::ParGalerkinDifference(mesh.get(),
                                                  fec.get(),
                                                  embeddedElements_p,
                                                  cutElements_p,
                                                  num_state,
                                                  Ordering::byVDIM,
                                                  fe_order,
                                                  comm));

   cout << "num_state " << num_state << endl;
   fes_gd_m.reset(new GDSpaceType(mesh.get(),
                                  fec.get(),
                                  embeddedElements_m,
                                  cutElements_m,
                                  num_state,
                                  Ordering::byVDIM,
                                  fe_order,
                                  comm));
}

template <int dim, bool entvar>
void CutEulerDGSensitivityTestSolver<dim, entvar>::constructForms()
{
   if (gd)
   {
      res.reset(new NonlinearFormType(fes_gd.get()));
      res_sens_cut.reset(new NonlinearFormType(fes_gd.get()));
      res_p.reset(new NonlinearFormType(fes_gd_p.get()));
      res_m.reset(new NonlinearFormType(fes_gd_m.get()));
      double alpha = 1.0;
      if ((entvar) && (!options["time-dis"]["steady"].template get<bool>()))
      {
         nonlinear_mass.reset(new NonlinearFormType(fes_gd.get()));
         mass.reset();
      }
      else
      {
         mass.reset(new BilinearFormType(fes_gd.get()));
         nonlinear_mass.reset();
      }
      ent.reset(new NonlinearFormType(fes_gd.get()));
   }
   else
   {
      res.reset(new NonlinearFormType(fes.get()));
      res_sens_cut.reset(new NonlinearFormType(fes.get()));
      if ((entvar) && (!options["time-dis"]["steady"].template get<bool>()))
      {
         nonlinear_mass.reset(new NonlinearFormType(fes.get()));
         mass.reset();
      }
      else
      {
         mass.reset(new BilinearFormType(fes.get()));
         nonlinear_mass.reset();
      }
      ent.reset(new NonlinearFormType(fes.get()));
   }
}
template <int dim, bool entvar>
void CutEulerDGSensitivityTestSolver<dim, entvar>::addMassIntegrators(
    double alpha)
{
   mass->AddDomainIntegrator(
       new CutDGMassIntegrator(cutSquareIntRules, embeddedElements, num_state));
}

template <int dim, bool entvar>
void CutEulerDGSensitivityTestSolver<dim, entvar>::addResVolumeIntegrators(
    double alpha,
    double &diff_coeff)
{
#if 1

   res->AddDomainIntegrator(new CutEulerDGIntegrator<dim>(
       diff_stack, cutSquareIntRules, embeddedElements, alpha));
#if 1
   res_sens_cut->AddDomainIntegrator(
       new CutEulerDGSensitivityIntegrator<dim>(diff_stack,
                                                cutSquareIntRules,
                                                cutSquareIntRules_sens,
                                                embeddedElements,
                                                alpha));
   /// add volume integrators
   res_p->AddDomainIntegrator(new CutEulerDGIntegrator<dim>(
       diff_stack, cutSquareIntRules_p, embeddedElements_p, alpha));
   res_m->AddDomainIntegrator(new CutEulerDGIntegrator<dim>(
       diff_stack, cutSquareIntRules_m, embeddedElements_m, alpha));
   ParCentGridFunction x(fes_gd.get());
//    double area = res_sens_cut->GetEnergy(x);
//    cout << "area " << area << endl;
#endif

   // res->AddDomainIntegrator(new CutEulerDiffusionIntegrator<dim>(
   //     cutSquareIntRules, embeddedElements, diff_coeff, alpha));
   auto &bcs = options["bcs"];
   if (bcs.find("vortex") != bcs.end())
   {  // isentropic vortex BC
      if (dim != 2)
      {
         throw MachException(
             "CutEulerDGSensitivityTestSolver::addBoundaryIntegrators(alpha)\n"
             "\tisentropic vortex BC must use 2D mesh!");
      }
      res->AddDomainIntegrator(new CutDGIsentropicVortexBC<dim, entvar>(
          diff_stack, fec.get(), cutSegmentIntRules_outer, phi_outer, alpha));
   }
   if (bcs.find("slip-wall") != bcs.end())
   {  // slip-wall boundary condition
#if 1
      cout << "slip-wall bc are present " << endl;
      res->AddDomainIntegrator(new CutDGSlipWallBC<dim, entvar>(
          diff_stack, fec.get(), cutSegmentIntRules, phi, alpha));
      res_p->AddDomainIntegrator(new CutDGSlipWallBC<dim, entvar>(
          diff_stack, fec.get(), cutSegmentIntRules_p, phi_p, alpha));
      res_m->AddDomainIntegrator(new CutDGSlipWallBC<dim, entvar>(
          diff_stack, fec.get(), cutSegmentIntRules_m, phi_m, alpha));
      res_sens_cut->AddDomainIntegrator(
          new CutDGSensitivitySlipWallBC<dim, entvar>(diff_stack,
                                                      fec.get(),
                                                      cutSegmentIntRules,
                                                      cutSegmentIntRules_sens,
                                                      phi,
                                                      alpha));
#endif
   }
   //    double peri = res->GetEnergy(x);
   //    double peri_sens = res_sens_cut->GetEnergy(x);
   //    cout << "peri_sens " << peri_sens << endl;
   if (options["flow-param"]["potential-mms"].template get<bool>())
   {
      if (dim != 2)
      {
         throw MachException("Inviscid MMS problem only available for 2D!");
      }
      res->AddDomainIntegrator(new CutPotentialMMSIntegrator<dim, entvar>(
          diff_stack, cutSquareIntRules, embeddedElements, -alpha));
      res_p->AddDomainIntegrator(new CutPotentialMMSIntegrator<dim, entvar>(
          diff_stack, cutSquareIntRules_p, embeddedElements_p, -alpha));
      res_m->AddDomainIntegrator(new CutPotentialMMSIntegrator<dim, entvar>(
          diff_stack, cutSquareIntRules_m, embeddedElements_m, -alpha));
      res_sens_cut->AddDomainIntegrator(
          new CutSensitivityPotentialMMSIntegrator<dim, entvar>(
              diff_stack,
              cutSquareIntRules,
              cutSquareIntRules_sens,
              embeddedElements,
              -alpha));
   }

#endif
/// use this for testing purposes
#if 0
   //GridFunction x(fes.get());
   ParCentGridFunction x(fes_gd.get());
   res->AddDomainIntegrator(new CutEulerDGIntegrator<dim>(
       diff_stack, cutSquareIntRules, embeddedElements, alpha));
   double area;
   cout << "before GetEnergy() " << endl;
   area = res->GetEnergy(x);
   cout << "after GetEnergy() " << endl;
   //  double exact_area = 400 - 0.0817073;  // airfoil
   double exact_area = 100.0 - M_PI * 0.25;
   //double exact_area = 2 * M_PI;
   cout << "correct area: " << (exact_area) << endl;
   cout << "calculated area: " << area << endl;
   cout << "area err = " << abs(area - exact_area) << endl;
   double eop_c, eip_c;
   auto &bcs = options["bcs"];
   if (bcs.find("vortex") != bcs.end())
   {  // isentropic vortex BC
      if (dim != 2)
      {
         throw MachException(
             "CutEulerDGSensitivityTestSolver::addBoundaryIntegrators(alpha)\n"
             "\tisentropic vortex BC must use 2D mesh!");
      }
      res->AddDomainIntegrator(new CutDGIsentropicVortexBC<dim, entvar>(
          diff_stack, fec.get(), cutSegmentIntRules_outer, phi_outer, alpha));
      double eop = 3.0 * M_PI / 2.0;
      eop_c = res->GetEnergy(x) - area;
      cout << "exact outer perimeter: " << eop << endl;
      cout << "calcualted outer perimeter: " << eop_c << endl;
      cout << "outer peri error:  " << abs(eop - eop_c) << endl;
   }
   if (bcs.find("slip-wall") != bcs.end())
   {  // slip-wall boundary condition
      cout << "slip-wall bc are present " << endl;
      res->AddDomainIntegrator(new CutDGSlipWallBC<dim, entvar>(
          diff_stack, fec.get(), cutSegmentIntRules, phi, alpha));
   }
   double eip = 2.0*M_PI*0.5;
   eip_c = res->GetEnergy(x)  - area;
   cout << "exact inner perimeter: " << eip << endl;
   cout << "calcualted bdr perimeter: " << eip_c << endl;
   cout << "bdr peri error:  "  << abs(eip-eip_c) << endl;
   int idx = 0.0;
   if (bcs.find("vortex") != bcs.end())
   {
      if (dim != 2)
      {
         throw MachException(
             "EulerDGSolver::addBoundaryIntegrators(alpha)\n"
             "\tisentropic vortex BC must use 2D mesh!");
      }
      vector<int> tmp = bcs["vortex"].template get<vector<int>>();
      bndry_marker[idx].SetSize(tmp.size(), 0);
      bndry_marker[idx].Assign(tmp.data());
      // isentropic vortex BC
      res->AddBdrFaceIntegrator(
          new CutDGVortexBC<dim, entvar>(diff_stack,
                                         fec.get(),
                                         cutBdrFaceIntRules,
                                         embeddedElements,
                                         alpha),
          bndry_marker[idx]);
      idx++;
      double farp = 4.0;
      double farp_c = res->GetEnergy(x) - area - eip_c - eop_c;
      cout << "exact farfield perimeter: " << farp << endl;
      cout << "calcualted farfield perimeter: " << farp_c << endl;
      cout << "farfield peri error:  " << abs(farp - farp_c) << endl;
   }

   if (bcs.find("far-field") != bcs.end())
   {
      // far-field boundary conditions on immersed boundary
      mfem::Vector qfar(dim + 2);
      getFreeStreamState(qfar);
      res->AddDomainIntegrator(new CutDGSlipFarFieldBC<dim, entvar>(
          diff_stack, fec.get(), cutSegmentIntRules, phi, qfar, alpha));
   }
#endif
}

template <int dim, bool entvar>
void CutEulerDGSensitivityTestSolver<dim, entvar>::addResBoundaryIntegrators(
    double alpha)
{
   auto &bcs = options["bcs"];
   int idx = 0;
   // ParCentGridFunction x(fes_gd.get());
   // GridFunction x(fes.get());
   if (bcs.find("vortex") != bcs.end())
   {
      if (dim != 2)
      {
         throw MachException(
             "EulerDGSolver::addBoundaryIntegrators(alpha)\n"
             "\tisentropic vortex BC must use 2D mesh!");
      }
      vector<int> tmp = bcs["vortex"].template get<vector<int>>();
      bndry_marker[idx].SetSize(tmp.size(), 0);
      bndry_marker[idx].Assign(tmp.data());
      // isentropic vortex BC
      res->AddBdrFaceIntegrator(
          new CutDGVortexBC<dim, entvar>(diff_stack,
                                         fec.get(),
                                         cutBdrFaceIntRules,
                                         embeddedElements,
                                         alpha),
          bndry_marker[idx]);
      idx++;
   }
   if (bcs.find("potential-flow") != bcs.end())
   {
      // far-field boundary conditions
      vector<int> tmp = bcs["potential-flow"].template get<vector<int>>();
      bndry_marker[idx].SetSize(tmp.size(), 0);
      bndry_marker[idx].Assign(tmp.data());
      res->AddBdrFaceIntegrator(
          new DGPotentialFlowBC<dim, entvar>(diff_stack, fec.get(), alpha),
          bndry_marker[idx]);
      idx++;
   }
}

template <int dim, bool entvar>
void CutEulerDGSensitivityTestSolver<dim, entvar>::addResInterfaceIntegrators(
    double alpha)
{
   // add the integrators based on if discretization is continuous or
   // discrete
   auto diss_coeff = options["space-dis"]["iface-coeff"].template get<double>();
   res->AddInteriorFaceIntegrator(
       new CutDGInterfaceIntegrator<dim, entvar>(diff_stack,
                                                 diss_coeff,
                                                 fec.get(),
                                                 immersedFaces,
                                                 cutInteriorFaceIntRules,
                                                 alpha));
#if 1
   res_p->AddInteriorFaceIntegrator(
       new CutDGInterfaceIntegrator<dim, entvar>(diff_stack,
                                                 diss_coeff,
                                                 fec.get(),
                                                 immersedFaces_p,
                                                 cutInteriorFaceIntRules_p,
                                                 alpha));
   res_m->AddInteriorFaceIntegrator(
       new CutDGInterfaceIntegrator<dim, entvar>(diff_stack,
                                                 diss_coeff,
                                                 fec.get(),
                                                 immersedFaces_m,
                                                 cutInteriorFaceIntRules_m,
                                                 alpha));
   res_sens_cut->AddInteriorFaceIntegrator(
       new CutDGSensitivityInterfaceIntegrator<dim, entvar>(
           diff_stack,
           diss_coeff,
           fec.get(),
           immersedFaces,
           cutInteriorFaceIntRules,
           cutInteriorFaceIntRules_sens,
           alpha));
#endif
   //    ParCentGridFunction x(fes_gd.get());
   //    res_sens_cut->GetEnergy(x);
}

template <int dim, bool entvar>
void CutEulerDGSensitivityTestSolver<dim, entvar>::addOutput(
    const std::string &fun,
    const nlohmann::json &options)
{
   if (fun == "drag")
   {
      // drag on the specified boundaries (not used for cut-cell mesh)
      auto bdrs = options["boundaries"].template get<vector<int>>();

      mfem::Vector drag_dir(dim);
      drag_dir = 0.0;
      if (dim == 1)
      {
         drag_dir(0) = 1.0;
      }
      else
      {
         drag_dir(iroll) = cos(aoa_fs);
         drag_dir(ipitch) = sin(aoa_fs);
      }
      drag_dir *= 1.0 / pow(mach_fs, 2.0);  // to get non-dimensional Cd

      FunctionalOutput out(*fes, res_fields);
      out.addOutputDomainIntegrator(new CutDGPressureForce<dim, entvar>(
          diff_stack, fec.get(), drag_dir, cutSegmentIntRules, phi));
      outputs.emplace(fun, std::move(out));
   }
   else if (fun == "lift")
   {
      cout << "calling lift integrator " << endl;
      // lift on the specified boundaries (not used for cut-cell mesh)
      auto bdrs = options["boundaries"].template get<vector<int>>();
      mfem::Vector lift_dir(dim);
      lift_dir = 0.0;
      if (dim == 1)
      {
         lift_dir(0) = 0.0;
      }
      else
      {
         lift_dir(iroll) = -sin(aoa_fs);
         lift_dir(ipitch) = cos(aoa_fs);
      }
      lift_dir *= 1.0 / pow(mach_fs, 2.0);  // to get non-dimensional Cl
      FunctionalOutput out(*fes, res_fields);
      // out.addOutputBdrFaceIntegrator(
      //     new DGPressureForce<dim, entvar>(diff_stack, fec.get(), lift_dir),
      //     std::move(bdrs));
      out.addOutputDomainIntegrator(new CutDGPressureForce<dim, entvar>(
          diff_stack, fec.get(), lift_dir, cutSegmentIntRules, phi));
      outputs.emplace(fun, std::move(out));
   }
   else if (fun == "entropy")
   {
      // integral of entropy over the entire volume domain
      // FunctionalOutput out(*fes, res_fields);
      // out.addOutputDomainIntegrator(
      //     new EntropyIntegrator<dim, entvar>(diff_stack));
      // outputs.emplace(fun, std::move(out));
   }
   else
   {
      throw MachException("Output with name " + fun +
                          " not supported by "
                          "CutEulerDGSensitivityTestSolver!\n");
   }
}
void randVectorState(const mfem::Vector &x, mfem::Vector &u)
{
   // std::cout << "u size: " << u.Size() << std::endl;
   for (int i = 0; i < u.Size(); ++i)
   {
      // std::cout << i << std::endl;
      u(i) = uniform_rand(gen);
      // u(i) = 1/sqrt(3);
      // u(i) = 1.0;
   }
}

template <int dim, bool entvar>
void CutEulerDGSensitivityTestSolver<dim, entvar>::testSensIntegrators(
    const ParCentGridFunction &u_gd)
{
   cout << "FD step-size: " << delta << endl;
   // initialize the vector that the Jacobian multiplies
   ParCentGridFunction v(fes_gd.get());
   VectorFunctionCoefficient v_rand(num_state, randVectorState);
   v.ProjectCoefficient(v_rand);
   std::unique_ptr<ParCentGridFunction> res_sens;
   std::unique_ptr<ParCentGridFunction> res_plus;
   std::unique_ptr<ParCentGridFunction> res_minus;
   res_sens.reset(new ParCentGridFunction(fes_gd.get()));
   res_plus.reset(new ParCentGridFunction(fes_gd_p.get()));
   res_minus.reset(new ParCentGridFunction(fes_gd_m.get()));
   *res_plus = 0.0;
   *res_minus = 0.0;
   *res_sens = 0.0;
   HypreParVector *res_p_true = res_plus->GetTrueDofs();
   HypreParVector *res_m_true = res_minus->GetTrueDofs();
   HypreParVector *res_sens_true = res_sens->GetTrueDofs();
   HypreParVector *u_true = u_gd.GetTrueDofs();
   cout << "res_p size " << res_p_true->Size() << endl;
   res_sens_cut->Mult(*u_true, *res_sens_true);
   *res_sens = *res_sens_true;
   double drda_v_auto = InnerProduct(*res_sens_true, v);
   res_p->Mult(*u_true, *res_p_true);
   *res_plus = *res_p_true;
   res_m->Mult(*u_true, *res_m_true);
   *res_minus = *res_m_true;
   subtract(1 / (2 * delta), *res_plus, *res_minus, *res_plus);
   double drda_v_fd = InnerProduct(*res_p_true, v);
   double abs_deriv_err = abs(drda_v_fd - drda_v_auto);
   double rel_deriv_err = abs_deriv_err / abs(drda_v_fd) * 100;
   cout << " ============================================ " << endl;
   cout << "drda_v_fd " << drda_v_fd << endl;
   cout << "drda_v_auto " << drda_v_auto << endl;
   cout << "|v.dRda_auto - v.dRda_FD|/|v.dRda_FD| x 100 " << endl;
   cout << rel_deriv_err << " %" << endl;
   cout << " ============================================ " << endl;
   //    cout << "check GetEnergy() " << endl;
   ParCentGridFunction x_p(fes_gd_p.get());
   ParCentGridFunction x_m(fes_gd_m.get());
   ParCentGridFunction x_sens(fes_gd.get());
   double area_p = res_p->GetEnergy(x_p);
   double area_m = res_m->GetEnergy(x_m);
   double dadr_auto = res_sens_cut->GetEnergy(x_sens);
   area_p -= area_m;
   area_p /= (2.0 * delta);
   double func_err = abs(area_p - dadr_auto) / abs(area_p);
   cout << " ============================================ " << endl;
   cout << " dAdr_FD " << endl;
   cout << area_p << endl;
   cout << " dAdr_auto " << endl;
   cout << dadr_auto << endl;
   cout << "functional error " << endl;
   cout << func_err * 100 << " % " << endl;
   cout << " ============================================ " << endl;
}

template <int dim, bool entvar>
double
CutEulerDGSensitivityTestSolver<dim, entvar>::calcConservativeVarsL2Error(
    void (*u_exact)(const mfem::Vector &, mfem::Vector &),
    int entry)
{
   // This lambda function computes the error at a node
   // Beware: this is not particularly efficient, given the conditionals
   // Also **NOT thread safe!**
   Vector qdiscrete(dim + 2);
   Vector qexact(dim + 2);  // define here to avoid reallocation
   auto node_error = [&](const Vector &discrete, const Vector &exact) -> double
   {
      if (entvar)
      {
         calcConservativeVars<double, dim>(discrete.GetData(),
                                           qdiscrete.GetData());
         calcConservativeVars<double, dim>(exact.GetData(), qexact.GetData());
      }
      else
      {
         qdiscrete = discrete;
         qexact = exact;
      }
      double err = 0.0;
      if (entry < 0)
      {
         for (int i = 0; i < dim + 2; ++i)
         {
            double dq = qdiscrete(i) - qexact(i);
            err += dq * dq;
         }
      }
      else
      {
         err = qdiscrete(entry) - qexact(entry);
         err = err * err;
      }
      return err;
   };

   VectorFunctionCoefficient exsol(num_state, u_exact);
   DenseMatrix vals;
   DenseMatrix exact_vals;
   Vector u_j;
   Vector exsol_j;
   double loc_norm = 0.0;
   for (int i = 0; i < fes->GetNE(); i++)
   {
      if (embeddedElements.at(i) == true)
      {
         loc_norm += 0.0;
      }
      else
      {
         const FiniteElement *fe = fes->GetFE(i);
         // const IntegrationRule *ir = &(fe->GetNodes());
         const IntegrationRule *ir;
         ir = cutSquareIntRules[i];
         if (ir == NULL)
         {
            int intorder = fe->GetOrder();
            ir = &(IntRules.Get(fe->GetGeomType(), intorder));
         }
         ElementTransformation *T = fes->GetElementTransformation(i);
         u->GetVectorValues(*T, *ir, vals);
         exsol.Eval(exact_vals, *T, *ir);
         for (int j = 0; j < ir->GetNPoints(); j++)
         {
            const IntegrationPoint &ip = ir->IntPoint(j);
            T->SetIntPoint(&ip);
            vals.GetColumnReference(j, u_j);
            exact_vals.GetColumnReference(j, exsol_j);
            loc_norm += ip.weight * T->Weight() * node_error(u_j, exsol_j);
            // std::cout << "ip.w " << ip.weight << std::endl;
         }
      }
      // std::cout << "loc_norm " << loc_norm << std::endl;
   }
   double norm = NAN;
   MPI_Allreduce(&loc_norm, &norm, 1, MPI_DOUBLE, MPI_SUM, comm);
   if (norm < 0.0)  // This was copied from mfem...should not happen for us
   {
      return -sqrt(-norm);
   }
   return sqrt(norm);
}
/// explicit instantiation
template class CutEulerDGSensitivityTestSolver<1, true>;
template class CutEulerDGSensitivityTestSolver<1, false>;
template class CutEulerDGSensitivityTestSolver<2, true>;
template class CutEulerDGSensitivityTestSolver<2, false>;
template class CutEulerDGSensitivityTestSolver<3, true>;
template class CutEulerDGSensitivityTestSolver<3, false>;
}  // namespace mach
