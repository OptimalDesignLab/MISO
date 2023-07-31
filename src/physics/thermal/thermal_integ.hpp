#ifndef MACH_THERMAL_INTEG
#define MACH_THERMAL_INTEG

#include <set>

#include "mfem.hpp"

#include "coefficient.hpp"
#include "electromag_integ.hpp"
#include "mach_input.hpp"
#include "mfem_common_integ.hpp"

namespace mach
{

class TestBCIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   void AssembleFaceVector(const mfem::FiniteElement &el1,
                           const mfem::FiniteElement &el2,
                           mfem::FaceElementTransformations &trans,
                           const mfem::Vector &elfun,
                           mfem::Vector &elvect) override;

#ifndef MFEM_THREAD_SAFE
   mfem::Vector shape;
#endif

   friend class TestBCIntegratorMeshRevSens;
};

class TestBCIntegratorMeshRevSens : public mfem::LinearFormIntegrator
{
public:
   /// \param[in] state - the state to use when evaluating d(psi^T R)/dX
   /// \param[in] adjoint - the adjoint to use when evaluating d(psi^T R)/dX
   /// \param[in] integ - reference to primal integrator
   TestBCIntegratorMeshRevSens(mfem::GridFunction &state,
                               mfem::GridFunction &adjoint,
                               TestBCIntegrator &integ)
    : state(state), adjoint(adjoint), integ(integ)
   { }

   /// \note Not used!
   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &mesh_coords_bar) override;

   /// the state to use when evaluating d(psi^T R)/dX
   mfem::GridFunction &state;
   /// the adjoint to use when evaluating d(psi^T R)/dX
   mfem::GridFunction &adjoint;
   /// reference to primal integrator
   TestBCIntegrator &integ;
#ifndef MFEM_THREAD_SAFE
   mfem::Vector shape_bar;
   mfem::DenseMatrix PointMat_bar;
   mfem::Array<int> vdofs;
   mfem::Vector elfun, psi;
#endif
};

class L2ProjectionIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   /// Construct the element local residual
   /// \param[in] el - the finite element whose residual we want
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - element local state vector
   /// \param[out] elvect - element local residual
   void AssembleElementVector(const mfem::FiniteElement &el,
                              mfem::ElementTransformation &trans,
                              const mfem::Vector &elfun,
                              mfem::Vector &elvect) override;

   /// Construct the element local Jacobian
   /// \param[in] el - the finite element whose Jacobian we want
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - element local state vector
   /// \param[out] elmat - element local Jacobian
   void AssembleElementGrad(const mfem::FiniteElement &el,
                            mfem::ElementTransformation &trans,
                            const mfem::Vector &elfun,
                            mfem::DenseMatrix &elmat) override;

   L2ProjectionIntegrator(mfem::Coefficient &g, double alpha = 1.0)
    : g(g), alpha(alpha)
   { }

private:
   /// Field value condition value
   mfem::Coefficient &g;
   /// scales the terms; can be used to move to rhs/lhs
   double alpha;

#ifndef MFEM_THREAD_SAFE
   mfem::Vector shape;
#endif

   friend class L2ProjectionIntegratorMeshRevSens;
};

class L2ProjectionIntegratorMeshRevSens : public mfem::LinearFormIntegrator
{
public:
   /// \param[in] state - the state to use when evaluating d(psi^T R)/dX
   /// \param[in] adjoint - the adjoint to use when evaluating d(psi^T R)/dX
   /// \param[in] integ - reference to primal integrator
   L2ProjectionIntegratorMeshRevSens(mfem::GridFunction &state,
                                     mfem::GridFunction &adjoint,
                                     L2ProjectionIntegrator &integ)
    : state(state), adjoint(adjoint), integ(integ)
   { }

   void AssembleRHSElementVect(const mfem::FiniteElement &mesh_el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &mesh_coords_bar) override;

private:
   /// the state to use when evaluating d(psi^T R)/dX
   mfem::GridFunction &state;
   /// the adjoint to use when evaluating d(psi^T R)/dX
   mfem::GridFunction &adjoint;
   /// reference to primal integrator
   L2ProjectionIntegrator &integ;

#ifndef MFEM_THREAD_SAFE
   mfem::Array<int> vdofs;
   mfem::Vector elfun;
   mfem::Vector psi;
   mfem::DenseMatrix PointMat_bar;
#endif
};

class ThermalContactResistanceIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   friend void setInputs(ThermalContactResistanceIntegrator &integ,
                         const MachInputs &inputs);

   void AssembleFaceVector(const mfem::FiniteElement &el1,
                           const mfem::FiniteElement &el2,
                           mfem::FaceElementTransformations &trans,
                           const mfem::Vector &elfun,
                           mfem::Vector &elvect) override;

   void AssembleFaceGrad(const mfem::FiniteElement &el1,
                         const mfem::FiniteElement &el2,
                         mfem::FaceElementTransformations &trans,
                         const mfem::Vector &elfun,
                         mfem::DenseMatrix &elmat) override;

   ThermalContactResistanceIntegrator(double h = 1.0,
                                      std::string name = "",
                                      double alpha = 1.0)
    : h(h), name(std::move(name)), alpha(alpha)
   { }

private:
   /// Thermal contact coefficient
   double h;
   /// name of the interface to apply the integrator to
   std::string name;
   /// scales the terms; can be used to move to rhs/lhs
   double alpha;

#ifndef MFEM_THREAD_SAFE
   mfem::Vector shape1;
   mfem::Vector shape2;

   mfem::DenseMatrix elmat11;
   mfem::DenseMatrix elmat12;
   mfem::DenseMatrix elmat22;
#endif
   friend class ThermalContactResistanceIntegratorMeshRevSens;
};

class ThermalContactResistanceIntegratorMeshRevSens
 : public mfem::LinearFormIntegrator
{
public:
   /// \param[in] mesh_fes - the mesh finite element space
   /// \param[in] state - the state to use when evaluating d(psi^T R)/dX
   /// \param[in] adjoint - the adjoint to use when evaluating d(psi^T R)/dX
   /// \param[in] integ - reference to primal integrator
   ThermalContactResistanceIntegratorMeshRevSens(
       mfem::FiniteElementSpace &mesh_fes,
       mfem::GridFunction &state,
       mfem::GridFunction &adjoint,
       ThermalContactResistanceIntegrator &integ)
    : mesh_fes(mesh_fes), state(state), adjoint(adjoint), integ(integ)
   { }

   void AssembleRHSElementVect(const mfem::FiniteElement &,
                               mfem::ElementTransformation &,
                               mfem::Vector &) override
   {
      mfem::mfem_error(
          "ThermalContactResistanceIntegratorMeshRevSens::"
          "AssembleRHSElementVect(...)");
   }

   /// \brief - assemble an element's contribution to d(psi^T R)/dX
   /// \param[in] mesh_el1 - the finite element that describes the mesh element
   /// \param[in] mesh_el2 - the finite element that describes the mesh element
   /// \param[in] trans - the transformation between reference and physical
   /// space
   /// \param[out] mesh_coords_bar - d(psi^T R)/dX for the element
   /// \note the LinearForm that assembles this integrator's FiniteElementSpace
   /// MUST be the mesh's nodal finite element space
   /// \note this signature is for sensitivity wrt mesh element
   void AssembleRHSElementVect(const mfem::FiniteElement &mesh_el1,
                               const mfem::FiniteElement &mesh_el2,
                               mfem::FaceElementTransformations &trans,
                               mfem::Vector &mesh_coords_bar) override;

private:
   /// The mesh finite element space used to assemble the sensitivity
   mfem::FiniteElementSpace &mesh_fes;
   /// the state to use when evaluating d(psi^T R)/dX
   mfem::GridFunction &state;
   /// the adjoint to use when evaluating d(psi^T R)/dX
   mfem::GridFunction &adjoint;
   /// reference to primal integrator
   ThermalContactResistanceIntegrator &integ;

#ifndef MFEM_THREAD_SAFE
   mfem::Array<int> vdofs1;
   mfem::Array<int> vdofs2;
   mfem::Vector elfun1;
   mfem::Vector elfun2;
   mfem::Vector psi1;
   mfem::Vector psi2;
   mfem::DenseMatrix PointMatFace_bar;
   mfem::Vector mesh_coords_face_bar;
#endif
};

class InternalConvectionInterfaceIntegrator
 : public mfem::NonlinearFormIntegrator
{
public:
   friend void setInputs(InternalConvectionInterfaceIntegrator &integ,
                         const MachInputs &inputs);

   void AssembleFaceVector(const mfem::FiniteElement &el1,
                           const mfem::FiniteElement &el2,
                           mfem::FaceElementTransformations &trans,
                           const mfem::Vector &elfun,
                           mfem::Vector &elvect) override;

   void AssembleFaceGrad(const mfem::FiniteElement &el1,
                         const mfem::FiniteElement &el2,
                         mfem::FaceElementTransformations &trans,
                         const mfem::Vector &elfun,
                         mfem::DenseMatrix &elmat) override;

   InternalConvectionInterfaceIntegrator(double h = 1.0,
                                         double theta_f = 1.0,
                                         std::string name = "",
                                         double alpha = 1.0)
    : h(h),
      theta_f(theta_f),
      name(std::move(name)),
      alpha(alpha)
   { }

private:
   /// Convection coefficient
   double h;
   /// Fluid temperature
   double theta_f;
   /// name of the interface to apply the integrator to
   std::string name;
   /// scales the terms; can be used to move to rhs/lhs
   double alpha;

#ifndef MFEM_THREAD_SAFE
   mfem::Vector shape1;
   mfem::Vector shape2;

   mfem::DenseMatrix elmat11;
   mfem::DenseMatrix elmat22;
#endif
   friend class InternalConvectionInterfaceIntegratorMeshRevSens;
};

class ConvectionBCIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   friend void setInputs(ConvectionBCIntegrator &integ,
                         const MachInputs &inputs)
   {
      setValueFromInputs(inputs, "h", integ.h);
      setValueFromInputs(inputs, "fluid_temp", integ.theta_f);
   }

   void AssembleFaceVector(const mfem::FiniteElement &el1,
                           const mfem::FiniteElement &el2,
                           mfem::FaceElementTransformations &trans,
                           const mfem::Vector &elfun,
                           mfem::Vector &elvect) override;

   void AssembleFaceGrad(const mfem::FiniteElement &el1,
                         const mfem::FiniteElement &el2,
                         mfem::FaceElementTransformations &trans,
                         const mfem::Vector &elfun,
                         mfem::DenseMatrix &elmat) override;

   ConvectionBCIntegrator(double alpha = 1.0) : alpha(alpha) { }
   ConvectionBCIntegrator(double h, double theta_f, double alpha = 1.0)
    : alpha(alpha), h(h), theta_f(theta_f)
   { }

private:
   /// scales the terms; can be used to move to rhs/lhs
   double alpha;
   /// Convection heat transfer coefficient
   double h = 1.0;
   /// Fluid temperature
   double theta_f = 0.0;
#ifndef MFEM_THREAD_SAFE
   mfem::Vector shape;
#endif

   friend class ConvectionBCIntegratorMeshRevSens;
   friend class ConvectionBCIntegratorHRevSens;
   friend class ConvectionBCIntegratorFluidTempRevSens;
};

/// Integrator to assemble d(psi^T R)/dX for the ConvectionBCIntegrator
class ConvectionBCIntegratorMeshRevSens : public mfem::LinearFormIntegrator
{
public:
   /// \param[in] state - the state to use when evaluating d(psi^T R)/dX
   /// \param[in] adjoint - the adjoint to use when evaluating d(psi^T R)/dX
   /// \param[in] integ - reference to primal integrator
   ConvectionBCIntegratorMeshRevSens(mfem::GridFunction &state,
                                     mfem::GridFunction &adjoint,
                                     ConvectionBCIntegrator &integ)
    : state(state), adjoint(adjoint), integ(integ)
   { }

   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &mesh_coords_bar) override;

   // /// \brief - assemble an element's contribution to d(psi^T R)/dX
   // /// \param[in] el - the finite element that describes the mesh element
   // /// \param[in] trans - the transformation between reference and physical
   // /// space
   // /// \param[out] mesh_coords_bar - d(psi^T R)/dX for the element
   // /// \note the LinearForm that assembles this integrator's
   // FiniteElementSpace
   // /// MUST be the mesh's nodal finite element space
   // void AssembleRHSElementVect(const mfem::FiniteElement &el,
   //                             mfem::FaceElementTransformations &trans,
   //                             mfem::Vector &mesh_coords_bar) override;

private:
   /// the state to use when evaluating d(psi^T R)/dX
   mfem::GridFunction &state;
   /// the adjoint to use when evaluating d(psi^T R)/dX
   mfem::GridFunction &adjoint;
   /// reference to primal integrator
   ConvectionBCIntegrator &integ;
#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix PointMat_bar;
   mfem::Array<int> vdofs;
   mfem::Vector elfun, psi;
#endif
};

class ConvectionBCIntegratorHRevSens : public mfem::NonlinearFormIntegrator
{
public:
   double GetFaceEnergy(const mfem::FiniteElement &el1,
                        const mfem::FiniteElement &el2,
                        mfem::FaceElementTransformations &trans,
                        const mfem::Vector &elfun) override;

   /// \param[in] adjoint - the adjoint to use when evaluating d(psi^T R)/dX
   /// \param[in] integ - reference to primal integrator
   ConvectionBCIntegratorHRevSens(mfem::GridFunction &adjoint,
                                  ConvectionBCIntegrator &integ)
    : adjoint(adjoint), integ(integ)
   { }

private:
   /// the adjoint to use when evaluating d(psi^T R)/dX
   mfem::GridFunction &adjoint;
   /// reference to primal integrator
   ConvectionBCIntegrator &integ;

#ifndef MFEM_THREAD_SAFE
   mfem::Array<int> vdofs;
   mfem::Vector psi;
#endif
};

class ConvectionBCIntegratorFluidTempRevSens
 : public mfem::NonlinearFormIntegrator
{
public:
   double GetFaceEnergy(const mfem::FiniteElement &el1,
                        const mfem::FiniteElement &el2,
                        mfem::FaceElementTransformations &trans,
                        const mfem::Vector &elfun) override;

   /// \param[in] adjoint - the adjoint to use when evaluating d(psi^T R)/dX
   /// \param[in] integ - reference to primal integrator
   ConvectionBCIntegratorFluidTempRevSens(mfem::GridFunction &adjoint,
                                          ConvectionBCIntegrator &integ)
    : adjoint(adjoint), integ(integ)
   { }

private:
   /// the adjoint to use when evaluating d(psi^T R)/dX
   mfem::GridFunction &adjoint;
   /// reference to primal integrator
   ConvectionBCIntegrator &integ;

#ifndef MFEM_THREAD_SAFE
   mfem::Array<int> vdofs;
   mfem::Vector psi;
#endif
};

inline void addBdrSensitivityIntegrator(
    ConvectionBCIntegrator &primal_integ,
    std::map<std::string, FiniteElementState> &fields,
    std::map<std::string, mfem::ParLinearForm> &rev_sens,
    std::map<std::string, mfem::ParNonlinearForm> &rev_scalar_sens,
    std::map<std::string, mfem::ParLinearForm> &fwd_sens,
    std::map<std::string, mfem::ParNonlinearForm> &fwd_scalar_sens,
    mfem::Array<int> *attr_marker,
    std::string adjoint_name)
{
   auto &mesh_fes = fields.at("mesh_coords").space();
   rev_sens.emplace("mesh_coords", &mesh_fes);

   auto &state_fes = fields.at("state").space();
   rev_scalar_sens.emplace("h", &state_fes);
   rev_scalar_sens.emplace("fluid_temp", &state_fes);

   if (attr_marker == nullptr)
   {
      rev_sens.at("mesh_coords")
          .AddBoundaryIntegrator(new ConvectionBCIntegratorMeshRevSens(
              fields.at("state").gridFunc(),
              fields.at(adjoint_name).gridFunc(),
              primal_integ));

      rev_scalar_sens.at("h").AddBdrFaceIntegrator(
          new ConvectionBCIntegratorHRevSens(fields.at(adjoint_name).gridFunc(),
                                             primal_integ));

      rev_scalar_sens.at("fluid_temp")
          .AddBdrFaceIntegrator(new ConvectionBCIntegratorFluidTempRevSens(
              fields.at(adjoint_name).gridFunc(), primal_integ));
   }
   else
   {
      rev_sens.at("mesh_coords")
          .AddBoundaryIntegrator(new ConvectionBCIntegratorMeshRevSens(
                                     fields.at("state").gridFunc(),
                                     fields.at(adjoint_name).gridFunc(),
                                     primal_integ),
                                 *attr_marker);

      rev_scalar_sens.at("h").AddBdrFaceIntegrator(
          new ConvectionBCIntegratorHRevSens(fields.at(adjoint_name).gridFunc(),
                                             primal_integ),
          *attr_marker);

      rev_scalar_sens.at("fluid_temp")
          .AddBdrFaceIntegrator(
              new ConvectionBCIntegratorFluidTempRevSens(
                  fields.at(adjoint_name).gridFunc(), primal_integ),
              *attr_marker);
   }
}

class OutfluxBCIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   friend void setInputs(OutfluxBCIntegrator &integ, const MachInputs &inputs)
   {
      setValueFromInputs(inputs, "outflux", integ.flux);
   }

   void AssembleFaceVector(const mfem::FiniteElement &el1,
                           const mfem::FiniteElement &el2,
                           mfem::FaceElementTransformations &trans,
                           const mfem::Vector &elfun,
                           mfem::Vector &elvect) override;

   void AssembleFaceGrad(const mfem::FiniteElement &el1,
                         const mfem::FiniteElement &el2,
                         mfem::FaceElementTransformations &trans,
                         const mfem::Vector &elfun,
                         mfem::DenseMatrix &elmat) override;

   OutfluxBCIntegrator(double alpha = 1.0) : alpha(alpha) { }
   OutfluxBCIntegrator(double outflux, double alpha = 1.0)
    : alpha(alpha), flux(outflux)
   { }

private:
   /// scales the terms; can be used to move to rhs/lhs
   double alpha;
   /// Value of the flux leaving the surface
   double flux = 10.0;
#ifndef MFEM_THREAD_SAFE
   mfem::Vector shape;
#endif

   friend class OutfluxBCIntegratorMeshRevSens;
};

/// Integrator to assemble d(psi^T R)/dX for the OutfluxBCIntegrator
class OutfluxBCIntegratorMeshRevSens : public mfem::LinearFormIntegrator
{
public:
   /// \param[in] state - the state to use when evaluating d(psi^T R)/dX
   /// \param[in] adjoint - the adjoint to use when evaluating d(psi^T R)/dX
   /// \param[in] integ - reference to primal integrator
   OutfluxBCIntegratorMeshRevSens(mfem::GridFunction &state,
                                  mfem::GridFunction &adjoint,
                                  OutfluxBCIntegrator &integ)
    : state(state), adjoint(adjoint), integ(integ)
   { }

   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &mesh_coords_bar) override;

private:
   /// the state to use when evaluating d(psi^T R)/dX
   mfem::GridFunction &state;
   /// the adjoint to use when evaluating d(psi^T R)/dX
   mfem::GridFunction &adjoint;
   /// reference to primal integrator
   OutfluxBCIntegrator &integ;
#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix PointMat_bar;
   mfem::Array<int> vdofs;
   mfem::Vector elfun, psi;
#endif
};

inline void addBdrSensitivityIntegrator(
    OutfluxBCIntegrator &primal_integ,
    std::map<std::string, FiniteElementState> &fields,
    std::map<std::string, mfem::ParLinearForm> &rev_sens,
    std::map<std::string, mfem::ParNonlinearForm> &rev_scalar_sens,
    std::map<std::string, mfem::ParLinearForm> &fwd_sens,
    std::map<std::string, mfem::ParNonlinearForm> &fwd_scalar_sens,
    mfem::Array<int> *attr_marker,
    std::string adjoint_name)
{
   auto &mesh_fes = fields.at("mesh_coords").space();
   rev_sens.emplace("mesh_coords", &mesh_fes);

   if (attr_marker == nullptr)
   {
      rev_sens.at("mesh_coords")
          .AddBoundaryIntegrator(new OutfluxBCIntegratorMeshRevSens(
              fields.at("state").gridFunc(),
              fields.at(adjoint_name).gridFunc(),
              primal_integ));
   }
   else
   {
      rev_sens.at("mesh_coords")
          .AddBoundaryIntegrator(new OutfluxBCIntegratorMeshRevSens(
                                     fields.at("state").gridFunc(),
                                     fields.at(adjoint_name).gridFunc(),
                                     primal_integ),
                                 *attr_marker);
   }
}

// // MACH::Diffusion integrator copied over from mfem's declaration exactly
// // Only difference is that it has DiffusionIntegratorMeshRevSens as a friend
// /// TODO: Use inheritance or move this integrator to another file as needed
// /** Class for integrating the bilinear form a(u,v) := (Q grad u, grad v)
// where Q
//     can be a scalar or a matrix coefficient. **/
// class DiffusionIntegrator : public mfem::BilinearFormIntegrator
// {
// protected:
//    Coefficient *Q;
//    VectorCoefficient *VQ;
//    MatrixCoefficient *MQ;

// private:
//    Vector vec, vecdxt, pointflux, shape;
// #ifndef MFEM_THREAD_SAFE
//    DenseMatrix dshape, dshapedxt, invdfdx, M, dshapedxt_m;
//    DenseMatrix te_dshape, te_dshapedxt;
//    Vector D;
// #endif

//    // PA extension
//    const FiniteElementSpace *fespace;
//    const DofToQuad *maps;         ///< Not owned
//    const GeometricFactors *geom;  ///< Not owned
//    int dim, ne, dofs1D, quad1D;
//    Vector pa_data;
//    bool symmetric = true;  ///< False if using a nonsymmetric matrix
//    coefficient

// public:
//    /// Construct a diffusion integrator with coefficient Q = 1
//    DiffusionIntegrator(const IntegrationRule *ir = nullptr)
//     : BilinearFormIntegrator(ir),
//       Q(NULL),
//       VQ(NULL),
//       MQ(NULL),
//       maps(NULL),
//       geom(NULL)
//    { }

//    /// Construct a diffusion integrator with a scalar coefficient q
//    DiffusionIntegrator(Coefficient &q, const IntegrationRule *ir = nullptr)
//     : BilinearFormIntegrator(ir),
//       Q(&q),
//       VQ(NULL),
//       MQ(NULL),
//       maps(NULL),
//       geom(NULL)
//    { }

//    /// Construct a diffusion integrator with a vector coefficient q
//    DiffusionIntegrator(VectorCoefficient &q,
//                        const IntegrationRule *ir = nullptr)
//     : BilinearFormIntegrator(ir),
//       Q(NULL),
//       VQ(&q),
//       MQ(NULL),
//       maps(NULL),
//       geom(NULL)
//    { }

//    /// Construct a diffusion integrator with a matrix coefficient q
//    DiffusionIntegrator(MatrixCoefficient &q,
//                        const IntegrationRule *ir = nullptr)
//     : BilinearFormIntegrator(ir),
//       Q(NULL),
//       VQ(NULL),
//       MQ(&q),
//       maps(NULL),
//       geom(NULL)
//    { }

//    /** Given a particular Finite Element computes the element stiffness
//    matrix
//        elmat. */
//    virtual void AssembleElementMatrix(const FiniteElement &el,
//                                       ElementTransformation &Trans,
//                                       DenseMatrix &elmat);
//    /** Given a trial and test Finite Element computes the element stiffness
//        matrix elmat. */
//    virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
//                                        const FiniteElement &test_fe,
//                                        ElementTransformation &Trans,
//                                        DenseMatrix &elmat);

//    /// Perform the local action of the BilinearFormIntegrator
//    virtual void AssembleElementVector(const FiniteElement &el,
//                                       ElementTransformation &Tr,
//                                       const Vector &elfun,
//                                       Vector &elvect);

//    virtual void ComputeElementFlux(const FiniteElement &el,
//                                    ElementTransformation &Trans,
//                                    Vector &u,
//                                    const FiniteElement &fluxelem,
//                                    Vector &flux,
//                                    bool with_coef = true,
//                                    const IntegrationRule *ir = NULL);

//    virtual double ComputeFluxEnergy(const FiniteElement &fluxelem,
//                                     ElementTransformation &Trans,
//                                     Vector &flux,
//                                     Vector *d_energy = NULL);

//    using BilinearFormIntegrator::AssemblePA;

//    // virtual void AssembleMF(const FiniteElementSpace &fes);

//    // virtual void AssemblePA(const FiniteElementSpace &fes);

//    // virtual void AssembleEA(const FiniteElementSpace &fes, Vector &emat,
//    // const bool add);

//    // virtual void AssembleDiagonalPA(Vector &diag);

//    // virtual void AssembleDiagonalMF(Vector &diag);

//    // virtual void AddMultMF(const Vector&, Vector&) const;

//    // virtual void AddMultPA(const Vector&, Vector&) const;

//    // virtual void AddMultTransposePA(const Vector&, Vector&) const;

//    static const IntegrationRule &GetRule(const FiniteElement &trial_fe,
//                                          const FiniteElement &test_fe);

//    bool SupportsCeed() const { return DeviceCanUseCeed(); }

//    Coefficient *GetCoefficient() const { return Q; }

//    friend class DiffusionIntegratorMeshRevSens;
// };

// /// Integrator to assemble d(psi^T R)/dX for the DiffusionIntegrator
// class DiffusionIntegratorMeshRevSens : public mfem::LinearFormIntegrator
// {
// public:
//    /// \param[in] state - the state to use when evaluating d(psi^T R)/dX
//    /// \param[in] adjoint - the adjoint to use when evaluating d(psi^T R)/dX
//    /// \param[in] integ - reference to primal integrator
//    DiffusionIntegratorMeshRevSens(mfem::GridFunction &state,
//                                   mfem::GridFunction &adjoint,
//                                   DiffusionIntegrator &integ)
//     : state(state), adjoint(adjoint), integ(integ)
//    { }

//    /// \brief - assemble an element's contribution to d(psi^T R)/dX
//    /// \param[in] el - the finite element that describes the mesh element
//    /// \param[in] trans - the transformation between reference and physical
//    /// space
//    /// \param[out] mesh_coords_bar - d(psi^T R)/dX for the element
//    /// \note the LinearForm that assembles this integrator's
//    FiniteElementSpace
//    /// MUST be the mesh's nodal finite element space
//    void AssembleRHSElementVect(const mfem::FiniteElement &el,
//                                mfem::ElementTransformation &trans,
//                                mfem::Vector &mesh_coords_bar) override;

// private:
//    /// the state to use when evaluating d(psi^T R)/dX
//    mfem::GridFunction &state;
//    /// the adjoint to use when evaluating d(psi^T R)/dX
//    mfem::GridFunction &adjoint;
//    /// reference to primal integrator
//    DiffusionIntegrator &integ;
// #ifndef MFEM_THREAD_SAFE
//    mfem::DenseMatrix dshapedxt_bar, PointMat_bar;
//    mfem::Array<int> vdofs;
//    mfem::Vector elfun, psi;
// #endif
// };

// inline void addSensitivityIntegrator(
//     DiffusionIntegrator &primal_integ,
//     std::map<std::string, FiniteElementState> &fields,
//     std::map<std::string, mfem::ParLinearForm> &rev_sens,
//     std::map<std::string, mfem::ParNonlinearForm> &rev_scalar_sens,
//     std::map<std::string, mfem::ParLinearForm> &fwd_sens,
//     std::map<std::string, mfem::ParNonlinearForm> &fwd_scalar_sens)
// {
//    auto &mesh_fes = fields.at("mesh_coords").space();
//    rev_sens.emplace("mesh_coords", &mesh_fes);
//    rev_sens.at("mesh_coords")
//        .AddDomainIntegrator(
//            new DiffusionIntegratorMeshRevSens(fields.at("state").gridFunc(),
//                                               fields.at("adjoint").gridFunc(),
//                                               primal_integ));
// }

}  // namespace mach

#endif