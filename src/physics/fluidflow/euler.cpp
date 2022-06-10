#include <memory>

#include "sbp_fe.hpp"
#include "euler.hpp"
#include "euler_fluxes.hpp"
#include "euler_integ.hpp"
#include <iostream>
using namespace mfem;
using namespace std;


namespace mach
{

template <int dim, bool entvar>
EulerSolver<dim, entvar>::EulerSolver(const string &opt_file_name,
                              unique_ptr<mfem::Mesh> smesh)
    : AbstractSolver(opt_file_name, move(smesh))
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
   aoa_fs = options["flow-param"]["aoa"].template get<double>()*M_PI/180;
   iroll = options["flow-param"]["roll-axis"].template get<int>();
   ipitch = options["flow-param"]["pitch-axis"].template get<int>();
   
   if (iroll == ipitch)
   {
      throw MachException("iroll and ipitch must be distinct dimensions!");
   }
   if ( (iroll < 0) || (iroll > 2) )
   {
      throw MachException("iroll axis must be between 0 and 2!");
   }
   if ( (ipitch < 0) || (ipitch > 2) )
   {
      throw MachException("ipitch axis must be between 0 and 2!");
   }
}


template <int dim, bool entvar>
void EulerSolver<dim, entvar>::addMassIntegrators(double alpha)
{
   nonlinear_mass->AddDomainIntegrator(new MassIntegrator<dim, entvar>(diff_stack, alpha)); 
}

template <int dim, bool entvar>
void EulerSolver<dim, entvar>::addVolumeIntegrators(double alpha)
{
   // TODO: if statement when using entropy variables as state variables

   // TODO: should decide between one-point and two-point fluxes using options
   res->AddDomainIntegrator(
       new IsmailRoeIntegrator<dim, entvar>(diff_stack, alpha));
   //res->AddDomainIntegrator(new EulerIntegrator<dim>(diff_stack, alpha));

   // add the LPS stabilization
   double lps_coeff = options["space-dis"]["lps-coeff"].template get<double>();
   res->AddDomainIntegrator(
       new EntStableLPSIntegrator<dim, entvar>(diff_stack, alpha, lps_coeff));
}

template <int dim, bool entvar>
void EulerSolver<dim, entvar>::addBoundaryIntegrators(double alpha)
{
   auto &bcs = options["bcs"];
   int idx = 0;
   if (bcs.find("vortex") != bcs.end())
   { // isentropic vortex BC
      if (dim != 2)
      {
         throw MachException("EulerSolver::addBoundaryIntegrators(alpha)\n"
                             "\tisentropic vortex BC must use 2D mesh!");
      }
      vector<int> tmp = bcs["vortex"].template get<vector<int>>();
      bndry_marker[idx].SetSize(tmp.size(), 0);
      bndry_marker[idx].Assign(tmp.data());
      res->AddBdrFaceIntegrator(
          new IsentropicVortexBC<dim, entvar>(diff_stack, fec.get(), alpha),
          bndry_marker[idx]);
      idx++;
   }
   if (bcs.find("slip-wall") != bcs.end())
   { // slip-wall boundary condition
      vector<int> tmp = bcs["slip-wall"].template get<vector<int>>();
      bndry_marker[idx].SetSize(tmp.size(), 0);
      bndry_marker[idx].Assign(tmp.data());
      res->AddBdrFaceIntegrator(
             new SlipWallBC<dim, entvar>(diff_stack, fec.get(), alpha),
             bndry_marker[idx]);
      idx++;
   }
   if (bcs.find("far-field") != bcs.end())
   { 
      // far-field boundary conditions
      vector<int> tmp = bcs["far-field"].template get<vector<int>>();
      mfem::Vector qfar(dim+2);
      getFreeStreamState(qfar);
      bndry_marker[idx].SetSize(tmp.size(), 0);
      bndry_marker[idx].Assign(tmp.data());
      res->AddBdrFaceIntegrator(
          new FarFieldBC<dim, entvar>(diff_stack, fec.get(), qfar, alpha),
          bndry_marker[idx]);
      idx++;
   }
   if (bcs.find("sod-shock-left") != bcs.end())
   {
      // 1d sod-shock boundary conditions
      vector<int> tmp = bcs["sod-shock-left"].template get<vector<int>>();
      mfem::Vector qfar(dim+2);
      qfar(0) =5.0; qfar(1) = 0.0; qfar(2) = 2.5;
      bndry_marker[idx].SetSize(tmp.size(), 0);
      bndry_marker[idx].Assign(tmp.data());
      res->AddBdrFaceIntegrator(
          new FarFieldBC<dim, entvar>(diff_stack, fec.get(), qfar, alpha),
          bndry_marker[idx]);
      idx++;
   }
   if (bcs.find("sod-shock-right") != bcs.end())
   {
      // 1d sod-shock boundary conditions
      vector<int> tmp = bcs["sod-shock-right"].template get<vector<int>>();
      mfem::Vector qfar(dim+2);
      qfar(0) = 0.5; qfar(1) = 0.0; qfar(2) = 0.25;
      bndry_marker[idx].SetSize(tmp.size(), 0);
      bndry_marker[idx].Assign(tmp.data());
      res->AddBdrFaceIntegrator(
          new FarFieldBC<dim, entvar>(diff_stack, fec.get(), qfar, alpha),
          bndry_marker[idx]);
      idx++;
   }
}

template <int dim, bool entvar>
void EulerSolver<dim, entvar>::addInterfaceIntegrators(double alpha)
{
   // add the integrators based on if discretization is continuous or discrete
   if (options["space-dis"]["basis-type"].template get<string>() == "dsbp")
   {
      double diss_coeff = options["space-dis"]["iface-coeff"].template get<double>();
      res->AddInteriorFaceIntegrator(
          new InterfaceIntegrator<dim, entvar>(diff_stack, diss_coeff,
                                               fec.get(), alpha));
   }
}

template <int dim, bool entvar>
void EulerSolver<dim, entvar>::addMassIntegrator(double alpha)
{
   double dt = options["time-dis"]["dt"].template get<double>();
   mass_integ.reset(new MassIntegrator<dim,entvar>(diff_stack, alpha));
   nonlinear_mass->AddDomainIntegrator(mass_integ.get());
}

template <int dim, bool entvar>
void EulerSolver<dim, entvar>::addOutputs()
{
   auto &fun = options["outputs"];
   int idx = 0;
   if (fun.find("drag") != fun.end())
   { 
      // drag on the specified boundaries
      vector<int> tmp = fun["drag"].template get<vector<int>>();
      output_bndry_marker[idx].SetSize(tmp.size(), 0);
      output_bndry_marker[idx].Assign(tmp.data());
      output.emplace("drag", fes.get());
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
      output.at("drag").AddBdrFaceIntegrator(
          new PressureForce<dim, entvar>(diff_stack, fec.get(), drag_dir),
          output_bndry_marker[idx]);
      idx++;
   }
   if (fun.find("lift") != fun.end())
   { 
      // lift on the specified boundaries
      vector<int> tmp = fun["lift"].template get<vector<int>>();
      output_bndry_marker[idx].SetSize(tmp.size(), 0);
      output_bndry_marker[idx].Assign(tmp.data());
      output.emplace("lift", fes.get());
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
      output.at("lift").AddBdrFaceIntegrator(
          new PressureForce<dim, entvar>(diff_stack, fec.get(), lift_dir),
          output_bndry_marker[idx]);
      idx++;
   }
   if (fun.find("entropy") != fun.end())
   {
      // integral of entropy over the entire volume domain
      output.emplace("entropy", fes.get());
      output.at("entropy").AddDomainIntegrator(
         new EntropyIntegrator<dim, entvar>(diff_stack));
   }

   // if (fun.find("mass" != fun.end())
   // {
   //    output.emplace("mass",fes.get());
   //    output.at("mass").AddDomainIntegrator(
   //       new MassIntegrator
   //    );
   // }
}

template <int dim, bool entvar>
double EulerSolver<dim, entvar>::calcStepSize(double cfl) const
{
   Vector q(dim+2);
   auto calcSpect = [&q](const double* dir, const double* u)
   {
      if (entvar)
      {
         calcConservativeVars<double, dim>(u, q);
         return calcSpectralRadius<double, dim>(dir, q);
      }
      else
      {
         return calcSpectralRadius<double, dim>(dir, u);
      }
   };
   double dt_local = 1e100;
   Vector xi(dim);
   Vector dxij(dim);
   Vector ui, dxidx;
   DenseMatrix uk;
   DenseMatrix adjJt(dim);
   for (int k = 0; k < fes->GetNE(); k++)
   {
      // get the element, its transformation, and the state values on element
      const FiniteElement *fe = fes->GetFE(k);
      const IntegrationRule *ir = &(fe->GetNodes());
      ElementTransformation *trans = fes->GetElementTransformation(k);
      u->GetVectorValues(*trans, *ir, uk);
      for (int i = 0; i < fe->GetDof(); ++i)
      {
         trans->SetIntPoint(&fe->GetNodes().IntPoint(i));
         trans->Transform(fe->GetNodes().IntPoint(i), xi);
         CalcAdjugateTranspose(trans->Jacobian(), adjJt);
         uk.GetColumnReference(i, ui);
         for (int j = 0; j < fe->GetDof(); ++j)
         {
            if (j == i)
               continue;
            trans->Transform(fe->GetNodes().IntPoint(j), dxij);
            dxij -= xi;
            double dx = dxij.Norml2();
            dt_local = min(dt_local, cfl * dx * dx / calcSpect(dxij, ui)); // extra dx is to normalize dxij
         }
      }
   }
   double dt_min;
#ifdef MFEM_USE_MPI
   MPI_Allreduce(&dt_local, &dt_min, 1, MPI_DOUBLE, MPI_MIN, comm);
#else
   dt_min = dt_local;
#endif
   return dt_min;
}

template <int dim, bool entvar>
void EulerSolver<dim, entvar>::getFreeStreamState(mfem::Vector &q_ref) 
{
   q_ref = 0.0;
   q_ref(0) = 1.0;
   if (dim == 1)
   {
      q_ref(1) = q_ref(0)*mach_fs; // ignore angle of attack
   }
   else
   {
      q_ref(iroll+1) = q_ref(0)*mach_fs*cos(aoa_fs);
      q_ref(ipitch+1) = q_ref(0)*mach_fs*sin(aoa_fs);
   }
   q_ref(dim+1) = 1/(euler::gamma*euler::gami) + 0.5*mach_fs*mach_fs;
}

template <int dim, bool entvar>
double EulerSolver<dim, entvar>::calcConservativeVarsL2Error(
   void (*u_exact)(const mfem::Vector &, mfem::Vector &), int entry)
{
   // This lambda function computes the error at a node
   // Beware: this is not particularly efficient, given the conditionals
   // Also **NOT thread safe!**
   Vector qdiscrete(dim+2), qexact(dim+2); // define here to avoid reallocation
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
         for (int i = 0; i < dim+2; ++i)
         {
            double dq = qdiscrete(i) - qexact(i);
            err += dq*dq;
         }
      }
      else
      {
         err = qdiscrete(entry) - qexact(entry);
         err = err*err;  
      }
      return err;
   };

   VectorFunctionCoefficient exsol(num_state, u_exact);
   DenseMatrix vals, exact_vals;
   Vector u_j, exsol_j;
   fes->GetProlongationMatrix()->Mult(*uc, *u);
   double loc_norm = 0.0;
   for (int i = 0; i < fes->GetNE(); i++)
   {
      const FiniteElement *fe = fes->GetFE(i);
      const IntegrationRule *ir = &(fe->GetNodes());
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
      }
   }
   double norm;
#ifdef MFEM_USE_MPI
   MPI_Allreduce(&loc_norm, &norm, 1, MPI_DOUBLE, MPI_SUM, comm);
#else
   norm = loc_norm;
#endif
   if (norm < 0.0) // This was copied from mfem...should not happen for us
   {
      return -sqrt(-norm);
   }
   return sqrt(norm);
}

// template <int dim, bool entvar>
// void EulerSolver<dim, entvar>::updateNonlinearMass(int ti, double dt, double alpha)
// {
//    fes->GetProlongationMatrix()->Mult(*uc, *u);
//    if(0 == ti)
//    {
//       mass_integ.reset(new MassIntegrator<dim, entvar>(diff_stack, *u, dt, alpha));
//       nonlinear_mass->AddDomainIntegrator(mass_integ.get()); 
//    }
//    dynamic_cast<mach::NonlinearMassIntegrator<MassIntegrator<dim,entvar>>*>
//                (mass_integ.get())->updateDeltat(dt);
// }

template<int dim, bool entvar>
void EulerSolver<dim, entvar>::convertToEntvar(mfem::Vector &state)
{
   if (entvar)
   {
      return ;
   }
   else
   {
      int num_nodes, offset;
      Array<int> vdofs(num_state);
      Vector el_con(num_state), el_ent(num_state);
      const FiniteElement *fe;
      for (int i = 0; i < fes->GetNE(); i++)
      {
         fe = fes->GetFE(i);
         num_nodes = fe->GetDof();
         for (int j = 0; j < num_nodes; j++)
         {
            offset = i * num_nodes * num_state + j * num_state;
            for (int k = 0; k < num_state; k++)
            {
               vdofs[k] = offset + k;
            }
            u->GetSubVector(vdofs, el_con);
            calcEntropyVars<double, dim>(el_con.GetData(), el_ent.GetData());
            state.SetSubVector(vdofs, el_ent);
         }
      }
   }
}

template<int dim, bool entvar>
void EulerSolver<dim, entvar>::convertToConserv(mfem::Vector &state)
{
   if (!entvar)
   {
      return ;
   }
   else
   {
      int num_nodes, offset;
      Array<int> vdofs(num_state);
      Vector el_con(num_state), el_ent(num_state);
      const FiniteElement *fe;
      for (int i = 0; i < fes->GetNE(); i++)
      {
         fe = fes->GetFE(i);
         num_nodes = fe->GetDof();
         for (int j = 0; j < num_nodes; j++)
         {
            offset = i * num_nodes * num_state + j * num_state;
            for (int k = 0; k < num_state; k++)
            {
               vdofs[k] = offset + k;
            }
            u->GetSubVector(vdofs, el_ent);
            // cout << "entropy variables is:";
            // el_ent.Print(cout, dim+2);
            calcConservativeVars<double, dim>(el_ent.GetData(), el_con.GetData());
            state.SetSubVector(vdofs, el_con);
         }
      }
   }
}

template<int dim, bool entvar>
void EulerSolver<dim, entvar>::convertToConservCent(mfem::Vector &state)
{
   if (!entvar)
   {

      return ;
   }
   else
   {
      int num_nodes, offset;
      Array<int> vdofs(num_state);
      Vector el_con(num_state), el_ent(num_state);
      const FiniteElement *fe;
      for (int i = 0; i < fes->GetNE(); i++)
      {
         fe = fes->GetFE(i);
         offset = i *num_state;
         for (int k = 0; k < num_state; k++)
         {
            vdofs[k] = offset + k;
         }
         uc->GetSubVector(vdofs, el_ent);
         calcConservativeVars<double, dim>(el_ent.GetData(), el_con.GetData());
         state.SetSubVector(vdofs, el_con);
      }
   }
}

template<int dim, bool entvar>
void EulerSolver<dim, entvar>::PrintSodShock(const std::string &file_name)
{
   cout << "In print sodshock\n";
   fes->GetProlongationMatrix()->Mult(*uc, *u);
   mfem::GridFunction state(fes_normal.get());
   state = *u;
   convertToConserv(state);
   ofstream write_value(file_name+"_u.txt");
   write_value.precision(14);
   ofstream write_coord(file_name+"_coord.txt");
   write_coord.precision(14);
   // ofstream write_error(file_name+"_error.txt");
   // write_error.precision(14);
   mfem::Vector quad_coord(1);
   mfem::Array<int> vdofs;
   ElementTransformation *eltransf;
   const FiniteElement *fe;
   int num_dofs;
   for (int i = 0; i < fes_normal->GetNE(); i++)
   {
      fe = fes_normal->GetFE(i);
      num_dofs = fe->GetDof();
      eltransf = mesh->GetElementTransformation(i);
      fes_normal->GetElementVDofs(i, vdofs);
      for(int j = 0; j < num_dofs; j++)
      {
         eltransf->Transform(fe->GetNodes().IntPoint(j), quad_coord);
         write_coord << quad_coord(0) << std::endl;

         for(int k = 0; k < num_state; k++)
         {
            write_value << state(vdofs[k*num_dofs + j]) << ' ';
         }
         write_value << std::endl;
      }
   }
   write_coord.close();
   write_value.close();
   // write_error.close();
}

template<int dim, bool entvar>
double EulerSolver<dim, entvar>::computeMass()
{
   fes->GetProlongationMatrix()->Mult(*uc,*u);
   mfem::GridFunction state(fes_normal.get());
   state = *u;
   int num_dofs;
   convertToConserv(state);
   mfem::Array<int> vdofs;
   
   const FiniteElement *fe;
   const SBPFiniteElement *sbp;
   ElementTransformation *eltransf;
   double mass = 0.0;
   for (int i = 0; i < fes_normal->GetNE(); i++)
   {
      fe = fes_normal->GetFE(i);
      sbp = dynamic_cast<const SBPFiniteElement*>(fe);
      eltransf = mesh->GetElementTransformation(i);
      fes_normal->GetElementVDofs(i,vdofs);
      num_dofs = fe->GetDof();
      for (int j = 0; j < num_dofs; j++)
      {
         eltransf->SetIntPoint(&fe->GetNodes().IntPoint(i));
         mass += state(vdofs[j]) * eltransf->Weight() * sbp->getDiagNormEntry(j); 
      }
   }
   return mass;
}

template<int dim, bool entvar>
void EulerSolver<dim, entvar>::PrintSodShockCenter(const std::string &file_name)
{
   cout << "In print sodshock center.\n";
   // prepare the file stream
   mfem::CentGridFunction state(fes.get());
   state = *uc;
   cout << "state size is " << state.Size() << endl;
   convertToConservCent(state);
   ofstream write_center(file_name+"_coord.txt");
   ofstream write_state(file_name+"_u.txt");
   write_state.precision(14);
   write_center.precision(14);
   // print the state
   mfem::Vector cent(1);
   int geom = mesh->GetElement(0)->GetGeometryType();
   ElementTransformation *eltransf;
   for (int i = 0; i < fes->GetNE(); i++)
   {
      eltransf = mesh->GetElementTransformation(i);
      eltransf->Transform(Geometries.GetCenter(geom), cent);
      write_center << cent(0) << std::endl;
      for (int j = 0; j < num_state; j++)
      {
         write_state << state( i * num_state + j) << ' ';
      }
      write_state << std::endl;
   }
   write_state.close();
   write_center.close();
}

template<int dim, bool entvar>
void EulerSolver<dim, entvar>::checkConversion(void (*u_exact)(const mfem::Vector &, mfem::Vector &))
{
   VectorFunctionCoefficient u_fun(num_state, u_exact);
   GridFunction state1(fes_normal.get());
   GridFunction state2(fes_normal.get());
   u->ProjectCoefficient(u_fun);

   if (entvar)
   {
      conToConservVars(*u, state1);
      conToEntropyVars(state1, state2);
   }
   else
   {
      conToEntropyVars(*u, state1);
      conToConservVars(state1, state2);
   }
   state2 -= *u;
   cout << "After two consersion " << state2.Norml2() << '\n';
}


template<int dim, bool entvar>
void EulerSolver<dim, entvar>::conToConservVars(const mfem::Vector &conserv, mfem::Vector &entropy)
{
   int num_nodes, offset;
   Array<int> vdofs(num_state);
   Vector el_con(num_state), el_ent(num_state);
   const FiniteElement *fe;
   for (int i = 0; i < fes_normal->GetNE(); i++)
   {
      fe = fes_normal->GetFE(i);
      num_nodes = fe->GetDof();
      for (int j = 0; j < num_nodes; j++)
      {
         offset = i * num_nodes * num_state + j * num_state;
         for (int k = 0; k < num_state; k++)
         {
            vdofs[k] = offset + k;
         }
         conserv.GetSubVector(vdofs, el_ent);
         // cout << "entropy variables is:";
         // el_ent.Print(cout, dim+2);
         calcConservativeVars<double, dim>(el_ent.GetData(), el_con.GetData());
         entropy.SetSubVector(vdofs, el_con);
      }
   }
}

template<int dim, bool entvar>
void EulerSolver<dim, entvar>::conToEntropyVars(const mfem::Vector &entropy, mfem::Vector &conserv)
{
   int num_nodes, offset;
   Array<int> vdofs(num_state);
   Vector el_con(num_state), el_ent(num_state);
   const FiniteElement *fe;
   for (int i = 0; i < fes_normal->GetNE(); i++)
   {
      fe = fes_normal->GetFE(i);
      num_nodes = fe->GetDof();
      for (int j = 0; j < num_nodes; j++)
      {
         offset = i * num_nodes * num_state + j * num_state;
         for (int k = 0; k < num_state; k++)
         {
            vdofs[k] = offset + k;
         }
         entropy.GetSubVector(vdofs, el_con);
         calcEntropyVars<double, dim>(el_con.GetData(), el_ent.GetData());
         conserv.SetSubVector(vdofs, el_ent);
      }
   }
}

// explicit instantiation
template class EulerSolver<1, true>;
template class EulerSolver<1, false>;
template class EulerSolver<2, true>;
template class EulerSolver<2, false>;
template class EulerSolver<3, true>;
template class EulerSolver<3, false>;

} // namespace mach
