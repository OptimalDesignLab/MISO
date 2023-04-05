#ifndef MACH_THERMAL_INTEG
#define MACH_THERMAL_INTEG

#include <fem/nonlininteg.hpp>
#include "mfem.hpp"

#include "coefficient.hpp"
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
   // /// \note the LinearForm that assembles this integrator's FiniteElementSpace
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

inline void addSensitivityIntegrator(
    ConvectionBCIntegrator &primal_integ,
    std::map<std::string, FiniteElementState> &fields,
    std::map<std::string, mfem::ParLinearForm> &rev_sens,
    std::map<std::string, mfem::ParNonlinearForm> &rev_scalar_sens,
    std::map<std::string, mfem::ParLinearForm> &fwd_sens,
    std::map<std::string, mfem::ParNonlinearForm> &fwd_scalar_sens)
{
   auto &mesh_fes = fields.at("mesh_coords").space();
   rev_sens.emplace("mesh_coords", &mesh_fes);
   rev_sens.at("mesh_coords")
       .AddBoundaryIntegrator(new ConvectionBCIntegratorMeshRevSens(
           fields.at("state").gridFunc(),
           fields.at("adjoint").gridFunc(),
           primal_integ));
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