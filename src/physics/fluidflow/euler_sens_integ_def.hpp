template <int dim, bool entvar>
void IsmailRoeMeshSensIntegrator<dim, entvar>::calcFlux(
    int di, const mfem::Vector &qL, const mfem::Vector &qR, mfem::Vector &flux)
{
   if (entvar)
   {
      calcIsmailRoeFluxUsingEntVars<double, dim>(di, qL.GetData(), qR.GetData(),
                                                 flux.GetData());
   }
   else
   {
      calcIsmailRoeFlux<double, dim>(di, qL.GetData(), qR.GetData(),
                                     flux.GetData());
   }
}

template <int dim, bool entvar>
void SlipWallBCMeshSens<dim, entvar>::calcFluxBar(
    const mfem::Vector &x, const mfem::Vector &dir, const mfem::Vector &u,
    const mfem::Vector &flux_bar, mfem::Vector &dir_bar)
{
   // create containers for active double objects for each input
   std::vector<adouble> x_a(x.Size());
   std::vector<adouble> dir_a(dir.Size());
   std::vector<adouble> u_a(u.Size());
   // initialize active double containers with data from inputs
   adept::set_values(x_a.data(), x.Size(), x.GetData());
   adept::set_values(dir_a.data(), dir.Size(), dir.GetData());
   adept::set_values(u_a.data(), u.Size(), u.GetData());
   // start new stack recording
   this->stack.new_recording();
   // create container for active double flux output
   std::vector<adouble> flux_a(u.Size());
   mach::calcSlipWallFlux<adouble, dim, entvar>(x_a.data(), dir_a.data(),
                                                u_a.data(), flux_a.data());
   adept::set_gradients(flux_a.data(), flux_bar.Size(), flux_bar.GetData());
   this->stack.compute_adjoint();
   adept::get_gradients(dir_a.data(), dir.Size(), dir_bar.GetData());
}

// template <int dim, bool entvar>
// double FarFieldBCDiff<dim, entvar>::GetFaceEnergy(const mfem::FiniteElement &el_bnd,
//                                             const mfem::FiniteElement &el_unused,
//                                             mfem::FaceElementTransformations &trans,
//                                             const mfem::Vector &elfun)
// {
// //    using namespace mfem;
// //    const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement&>(el_bnd);
// //    const int num_nodes = el_bnd.GetDof();
// //    //const int dim = sbp.GetDim();
// // #ifdef MFEM_THREAD_SAFE
// //    Vector u_face, x, nrm, flux_face;
// // #endif
// //    num_states = qfs.Size();
// //    u_face.SetSize(num_states);
// //    x.SetSize(dim);
// //    nrm.SetSize(dim);
// //    flux_face.SetSize(num_states);
// //    //Vector elvect(num_states*num_nodes);
// //    //elvect = 0.0;
// //     double energy = 0.0;
// //    // get element adjoint vector
// //    Vector eladj; Array<int> vdofs;
// //     adjoint->FESpace()->GetElementVDofs(trans.Elem1no, vdofs);
// //     adjoint->GetSubVector(vdofs, eladj);

// //    DenseMatrix u(elfun.GetData(), num_nodes, num_states);
// //    //DenseMatrix res(elvect.GetData(), num_nodes, num_states);

// //    const FiniteElement *sbp_face;
// //    switch (dim)
// //    {
// //       case 1: sbp_face = fec->FiniteElementForGeometry(Geometry::POINT);
// //               break;
// //       case 2: sbp_face = fec->FiniteElementForGeometry(Geometry::SEGMENT);
// //               break;
// //       default: throw mach::MachException(
// //          "InviscidBoundaryIntegrator::AssembleFaceVector())\n"
// //          "\tcannot handle given dimension");
// //    }
// //    IntegrationPoint el_ip;
// //    for (int i = 0; i < sbp_face->GetDof(); ++i)
// //    {
// //       const IntegrationPoint &face_ip = sbp_face->GetNodes().IntPoint(i);
// //       trans.Loc1.Transform(face_ip, el_ip);
// //       trans.Elem1->Transform(el_ip, x);
// //       int j = sbp.getIntegrationPointIndex(el_ip);
// //       u.GetRow(j, u_face);

// //       // get the normal vector and the flux on the face
// //       trans.Face->SetIntPoint(&face_ip);
// //       CalcOrtho(trans.Face->Jacobian(), nrm);
// //       flux(x, nrm, u_face, flux_face); //this function needs to return the derivatives w.r.t mach number
// //       flux_face *= face_ip.weight;

// //       // multiply by test function, contract with adjoint, and sum contributions
// //       for (int n = 0; n < num_states; ++n)
// //       {
// //          energy += eladj(i*num_states + n)*alpha*flux_face(n);
// //       }
// //    }

// //    return energy;
//    mfem::Vector eladj; mfem::Array<int> vdofs;
//    adjoint->FESpace()->GetElementVDofs(trans.Elem1No, vdofs);
//    adjoint->GetSubVector(vdofs, eladj);
//    mfem::Vector elvect(elfun.Size());
//    this->AssembleFaceVector(el_bnd, el_unused, trans, elfun, elvect);

//    return eladj*elvect;
// }
template <int dim, bool entvar>
void FarFieldBCDiff<dim, entvar>::AssembleRHSElementVect(
   const mfem::FiniteElement &el_bnd, mfem::FaceElementTransformations &trans,
   mfem::Vector &elvect)
{
   using namespace mfem;
   const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement&>(el_bnd);
   const int num_nodes = el_bnd.GetDof();
   //const int dim = sbp.GetDim();
#ifdef MFEM_THREAD_SAFE
   Vector u_face, x, nrm, flux_face;
#endif
   u_face.SetSize(this->num_states);
   this->x.SetSize(dim);
   this->nrm.SetSize(dim);
   flux_face.SetSize(this->num_states);
   elvect.SetSize(this->num_states*num_nodes);
   elvect = 0.0;

   Array<int> vdofs;
   this->state.FESpace()->GetElementVDofs(trans.Elem1No, vdofs); 
   Vector elfun(this->num_states*num_nodes);
   this->state.GetSubVector(vdofs, elfun);
   DenseMatrix u(elfun.GetData(), num_nodes, this->num_states);
   DenseMatrix res(elvect.GetData(), num_nodes, this->num_states);

   const FiniteElement *sbp_face;
   switch (dim)
   {
      case 1: sbp_face = this->state.FESpace()->FEColl()->
               FiniteElementForGeometry(Geometry::POINT);
              break;
      case 2: sbp_face = this->state.FESpace()->FEColl()->
               FiniteElementForGeometry(Geometry::SEGMENT);
              break;
      default: throw mach::MachException(
         "InviscidBoundaryIntegrator::AssembleFaceVector())\n"
         "\tcannot handle given dimension");
   }
   IntegrationPoint el_ip;
   for (int i = 0; i < sbp_face->GetDof(); ++i)
   {
      const IntegrationPoint &face_ip = sbp_face->GetNodes().IntPoint(i);
      trans.Loc1.Transform(face_ip, el_ip);
      trans.Elem1->Transform(el_ip, this->x);
      int j = sbp.getIntegrationPointIndex(el_ip);
      u.GetRow(j, this->u_face);

      // get the normal vector and the flux on the face
      trans.Face->SetIntPoint(&face_ip);
      CalcOrtho(trans.Face->Jacobian(), this->nrm);
      calcFlux(this->x, this->nrm, this->u_face, flux_face);
      flux_face *= face_ip.weight;

      // multiply by test function
      for (int n = 0; n < this->num_states; ++n)
      {
         res(j, n) += this->alpha*flux_face(n);
      }
   }
}


template <int dim, bool entvar>
void FarFieldBCDiff<dim, entvar>::calcFlux(const mfem::Vector &x,
                                          const mfem::Vector &dir,
                                          const mfem::Vector &q,
                                          mfem::Vector &flux_vec)
{
   // calcFarFieldMachDiff<double, dim, entvar>(dir.GetData(), qfs.GetData(),
   //                                       q.GetData(), work_vec.GetData(),
   //                                       flux_vec.GetData(), mach_fs, aoa_fs);
   // create containers for active double objects for each input
   std::vector<adouble> x_a(x.Size());
   std::vector<adouble> dir_a(dir.Size());
   std::vector<adouble> q_a(q.Size());
   std::vector<adouble> qfs_a(qfs.Size());
   std::vector<adouble> work_a(work_vec.Size());
   // initialize active double containers with data from inputs
   adept::set_values(x_a.data(), x.Size(), x.GetData());
   adept::set_values(dir_a.data(), dir.Size(), dir.GetData());
   adept::set_values(q_a.data(), q.Size(), q.GetData());
   adept::set_values(qfs_a.data(), qfs.Size(), qfs.GetData());
   adept::set_values(work_a.data(), work_vec.Size(), work_vec.GetData());
   // start new stack recording
   this->stack.new_recording();
   // create container for active double flux output
   std::vector<adouble> flux_a(q.Size());
   std::vector<double> flux_jac(qfs.Size()*q.Size());
   mach::calcFarFieldFlux<adouble, dim, entvar>(dir_a.data(), qfs_a.data(),
                                                q_a.data(), work_a.data(),
                                                 flux_a.data());
   stack.independent(qfs_a.data(), qfs.Size());
   stack.dependent(flux_a.data(), qfs.Size());
   stack.jacobian(flux_jac.data());
   // multiply by derivative of qfs w.r.t. mach number
   mfem::Vector ddM(dim+2); flux_vec.SetSize(dim+2);
   ddM(0) = 0;
   if(dim == 1)
      ddM(1) = 1;
   else
   {
      ddM(1) = cos(aoa_fs);
      ddM(2) = sin(aoa_fs);
   }
   ddM(3) = mach_fs;
   mfem::DenseMatrix Jac(flux_jac.data(), dim+2, dim+2);
   Jac.Mult(ddM, flux_vec);
}
