#ifndef MACH_MFEM_COMMON_INTEG
#define MACH_MFEM_COMMON_INTEG

#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "mach_integrator.hpp"

namespace mach
{
class VolumeIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   double GetElementEnergy(const mfem::FiniteElement &el,
                           mfem::ElementTransformation &trans,
                           const mfem::Vector &elfun) override;
};

class StateIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   double GetElementEnergy(const mfem::FiniteElement &el,
                           mfem::ElementTransformation &trans,
                           const mfem::Vector &elfun) override;

private:
#ifndef MFEM_THREAD_SAFE
   mfem::Vector shape;
#endif
};

class IEAggregateIntegratorNumerator : public mfem::NonlinearFormIntegrator
{
public:
   friend void setOptions(IEAggregateIntegratorNumerator &integ,
                          const nlohmann::json &options);

   IEAggregateIntegratorNumerator(const double rho) : rho(rho) { }

   double GetElementEnergy(const mfem::FiniteElement &el,
                           mfem::ElementTransformation &trans,
                           const mfem::Vector &elfun) override;

private:
   /// aggregation parameter rho
   double rho;
#ifndef MFEM_THREAD_SAFE
   mfem::Vector shape;
#endif
};

class IEAggregateIntegratorDenominator : public mfem::NonlinearFormIntegrator
{
public:
   friend void setOptions(IEAggregateIntegratorDenominator &integ,
                          const nlohmann::json &options);

   IEAggregateIntegratorDenominator(const double rho) : rho(rho) { }

   double GetElementEnergy(const mfem::FiniteElement &el,
                           mfem::ElementTransformation &trans,
                           const mfem::Vector &elfun) override;

private:
   /// aggregation parameter rho
   double rho;
#ifndef MFEM_THREAD_SAFE
   mfem::Vector shape;
#endif
};

class IECurlMagnitudeAggregateIntegratorNumerator
 : public mfem::NonlinearFormIntegrator
{
public:
   friend void setOptions(IECurlMagnitudeAggregateIntegratorNumerator &integ,
                          const nlohmann::json &options);

   IECurlMagnitudeAggregateIntegratorNumerator(const double rho) : rho(rho) { }

   double GetElementEnergy(const mfem::FiniteElement &el,
                           mfem::ElementTransformation &trans,
                           const mfem::Vector &elfun) override;

private:
   /// aggregation parameter rho
   double rho;
#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix curlshape, curlshape_dFt;
#endif
};

class IECurlMagnitudeAggregateIntegratorDenominator
 : public mfem::NonlinearFormIntegrator
{
public:
   friend void setOptions(IECurlMagnitudeAggregateIntegratorDenominator &integ,
                          const nlohmann::json &options);

   IECurlMagnitudeAggregateIntegratorDenominator(const double rho)
    : rho(rho) { }

   double GetElementEnergy(const mfem::FiniteElement &el,
                           mfem::ElementTransformation &trans,
                           const mfem::Vector &elfun) override;

private:
   /// aggregation parameter rho
   double rho;
#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix curlshape, curlshape_dFt;
#endif
};

class DiffusionIntegratorMeshSens final : public mfem::LinearFormIntegrator
{
public:
   DiffusionIntegratorMeshSens() = default;

   /// \brief - assemble an element's contribution to d(psi^T D u)/dX
   /// \param[in] el - the finite element that describes the mesh element
   /// \param[in] trans - the transformation between reference and physical
   /// space \param[out] mesh_coords_bar - d(psi^T D u)/dX for the element \note
   /// the LinearForm that assembles this integrator's FiniteElementSpace
   ///       MUST be the mesh's nodal finite element space
   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &mesh_coords_bar) override;

   void setState(const mfem::GridFunction &u) { state = &u; }

   void setAdjoint(const mfem::GridFunction &psi) { adjoint = &psi; }

private:
   /// the state to use when evaluating d(psi^T D u)/dX
   const mfem::GridFunction *state{nullptr};
   /// the adjoint to use when evaluating d(psi^T D u)/dX
   const mfem::GridFunction *adjoint{nullptr};
#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix dshape, dshapedxt;
   mfem::DenseMatrix dshapedxt_bar, PointMat_bar;
   mfem::Array<int> vdofs;
   mfem::Vector elfun, psi;
#endif
};

template <>
inline void addSensitivityIntegrator<mfem::DiffusionIntegrator>(
    mfem::DiffusionIntegrator &primal_integ,
    std::unordered_map<std::string, mfem::ParGridFunction> &fields,
    std::map<std::string, mfem::ParLinearForm> &sens,
    std::map<std::string, mfem::ParNonlinearForm> &scalar_sens)
{
   auto *mesh_fes = fields.at("mesh_coords").ParFESpace();
   sens.emplace("mesh_coords", mesh_fes);
   sens.at("mesh_coords").AddDomainIntegrator(new DiffusionIntegratorMeshSens);
}

class VectorFEWeakDivergenceIntegratorMeshSens final
 : public mfem::LinearFormIntegrator
{
public:
   VectorFEWeakDivergenceIntegratorMeshSens() = default;

   /// \brief - assemble an element's contribution to d(psi^T W u)/dX
   /// \param[in] el - the finite element that describes the mesh element
   /// \param[in] trans - the transformation between reference and physical
   /// space \param[out] mesh_coords_bar - d(psi^T W u)/dX for the element \note
   /// the LinearForm that assembles this integrator's FiniteElementSpace
   ///       MUST be the mesh's nodal finite element space
   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &mesh_coords_bar) override;

   void setState(const mfem::GridFunction &u) { state = &u; }

   void setAdjoint(const mfem::GridFunction &psi) { adjoint = &psi; }

private:
   /// the state to use when evaluating d(psi^T W u)/dX
   const mfem::GridFunction *state{nullptr};
   /// the adjoint to use when evaluating d(psi^T W u)/dX
   const mfem::GridFunction *adjoint{nullptr};
#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix dshape, dshapedxt, vshape, vshapedxt;
   mfem::DenseMatrix dshapedxt_bar, vshapedxt_bar, PointMat_bar;
   mfem::Array<int> vdofs;
   mfem::Vector elfun, psi;
#endif
};

template <>
inline void addSensitivityIntegrator<mfem::VectorFEWeakDivergenceIntegrator>(
    mfem::VectorFEWeakDivergenceIntegrator &primal_integ,
    std::unordered_map<std::string, mfem::ParGridFunction> &fields,
    std::map<std::string, mfem::ParLinearForm> &sens,
    std::map<std::string, mfem::ParNonlinearForm> &scalar_sens)
{
   auto *mesh_fes = fields.at("mesh_coords").ParFESpace();
   sens.emplace("mesh_coords", mesh_fes);
   sens.at("mesh_coords")
       .AddDomainIntegrator(new VectorFEWeakDivergenceIntegratorMeshSens);
}

/** Not differentiated, not needed since we use linear form version of
MagneticLoad class VectorFECurlIntegratorMeshSens final : public
mfem::LinearFormIntegrator
{
public:
   VectorFECurlIntegratorMeshSens(double alpha = 1.0)
      : Q(nullptr), state(nullptr), adjoint(nullptr), alpha(alpha)
   { }

   VectorFECurlIntegratorMeshSens(mfem::Coefficient &Q, double alpha = 1.0)
      : Q(&Q), state(nullptr), adjoint(nullptr), alpha(alpha)
   { }

   /// \brief - assemble an element's contribution to d(psi^T C u)/dX
   /// \param[in] el - the finite element that describes the mesh element
   /// \param[in] trans - the transformation between reference and physical
space
   /// \param[out] mesh_coords_bar - d(psi^T C u)/dX for the element
   /// \note the LinearForm that assembles this integrator's FiniteElementSpace
   ///       MUST be the mesh's nodal finite element space
   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &mesh_coords_bar) override;

   void setState(const mfem::GridFunction &u)
   { state = &u; }

   void setAdjoint(const mfem::GridFunction &psi)
   { adjoint = &psi; }

private:
   /// coefficient used in primal integrator
   mfem::Coefficient *Q;
   /// the state to use when evaluating d(psi^T C u)/dX
   const mfem::GridFunction *state;
   /// the adjoint to use when evaluating d(psi^T C u)/dX
   const mfem::GridFunction *adjoint;
   /// scaling term if the bilinear form has a negative sign in the residual
   const double alpha;
#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix curlshape, curlshape_dFt, vshape, vshapedxt;
   mfem::DenseMatrix curlshape_dFt_bar, vshapedxt_bar, PointMat_bar;
   mfem::Array<int> vdofs;
   mfem::Vector elfun, psi;
#endif
};
*/

class VectorFEMassIntegratorMeshSens final : public mfem::LinearFormIntegrator
{
public:
   VectorFEMassIntegratorMeshSens(double alpha = 1.0) : alpha(alpha) { }

   /// \brief - assemble an element's contribution to d(psi^T M u)/dX
   /// \param[in] el - the finite element that describes the mesh element
   /// \param[in] trans - the transformation between reference and physical
   /// space \param[out] mesh_coords_bar - d(psi^T M u)/dX for the element \note
   /// the LinearForm that assembles this integrator's FiniteElementSpace
   ///       MUST be the mesh's nodal finite element space
   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &mesh_coords_bar) override;

   void setState(const mfem::GridFunction &u) { state = &u; }

   void setAdjoint(const mfem::GridFunction &psi) { adjoint = &psi; }

private:
   /// the state to use when evaluating d(psi^T M u)/dX
   const mfem::GridFunction *state{nullptr};
   /// the adjoint to use when evaluating d(psi^T M u)/dX
   const mfem::GridFunction *adjoint{nullptr};
   /// scaling term if the bilinear form has a negative sign in the residual
   const double alpha;
#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix vshape, vshapedxt;
   mfem::DenseMatrix vshapedxt_bar, PointMat_bar;
   mfem::Array<int> vdofs;
   mfem::Vector elfun, psi;
#endif
};

template <>
inline void addSensitivityIntegrator<mfem::VectorFEMassIntegrator>(
    mfem::VectorFEMassIntegrator &primal_integ,
    std::unordered_map<std::string, mfem::ParGridFunction> &fields,
    std::map<std::string, mfem::ParLinearForm> &sens,
    std::map<std::string, mfem::ParNonlinearForm> &scalar_sens)
{
   auto *mesh_fes = fields.at("mesh_coords").ParFESpace();
   sens.emplace("mesh_coords", mesh_fes);

   auto *integ = new VectorFEMassIntegratorMeshSens;
   integ->setState(fields.at("in"));
   sens.at("mesh_coords").AddDomainIntegrator(integ);
}

class VectorFEDomainLFIntegratorMeshSens : public mfem::LinearFormIntegrator
{
public:
   VectorFEDomainLFIntegratorMeshSens(mfem::VectorCoefficient &F,
                                      double alpha = 1.0)
    : F(F), adjoint(nullptr), alpha(alpha)
   { }

   /// \brief - assemble an element's contribution to d(psi^T f)/dX
   /// \param[in] el - the finite element that describes the mesh element
   /// \param[in] trans - the transformation between reference and physical
   /// space \param[out] mesh_coords_bar - d(psi^T f)/dX for the element \note
   /// the LinearForm that assembles this integrator's FiniteElementSpace
   ///       MUST be the mesh's nodal finite element space
   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &mesh_coords_bar) override;

   void setAdjoint(const mfem::GridFunction &psi) { adjoint = &psi; }

private:
   /// vector coefficient from linear form
   mfem::VectorCoefficient &F;
   /// the adjoint to use when evaluating d(psi^T f)/dX
   const mfem::GridFunction *adjoint;
   /// scaling term if the linear form has a negative sign in the residual
   const double alpha;
#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix vshape, vshapedxt;
   mfem::DenseMatrix vshapedxt_bar, PointMat_bar;
   mfem::Array<int> vdofs;
   mfem::Vector psi;
#endif
};

// template <>
// inline void addSensitivityIntegrator<VectorFEDomainLFIntegrator>(
//    VectorFEDomainLFIntegrator &primal_integ,
//    std::unordered_map<std::string, mfem::ParGridFunction> &fields,
//    std::map<std::string, mfem::ParLinearForm> &sens,
//    std::map<std::string, mfem::ParNonlinearForm> &scalar_sens)
// {
//    auto mesh_fes = fields.at("mesh_coords").ParFESpace();
//    sens.emplace("mesh_coords", mesh_fes);
//    sens.at("mesh_coords").AddDomainIntegrator(
//       new VectorFEDomainLFIntegratorMeshSens);
// }

class VectorFEDomainLFCurlIntegrator final
 : public mfem::VectorFEDomainLFCurlIntegrator
{
public:
   VectorFEDomainLFCurlIntegrator(mfem::VectorCoefficient &V,
                                  double alpha = 1.0)
    : mfem::VectorFEDomainLFCurlIntegrator(V), F(V), alpha(alpha)
   { }

   inline void AssembleRHSElementVect(const mfem::FiniteElement &el,
                                      mfem::ElementTransformation &trans,
                                      mfem::Vector &elvect) override
   {
      mfem::VectorFEDomainLFCurlIntegrator::AssembleRHSElementVect(
          el, trans, elvect);
      if (alpha != 1.0)
      {
         elvect *= alpha;
      }
   }

private:
   /// vector coefficient from linear form
   mfem::VectorCoefficient &F;
   /// scaling term if the linear form has a negative sign in the residual
   const double alpha;
   /// class that implements mesh sensitivities for
   /// VectorFEDomainLFCurlIntegrator
   friend class VectorFEDomainLFCurlIntegratorMeshSens;
};

class VectorFEDomainLFCurlIntegratorMeshSens final
 : public mfem::LinearFormIntegrator
{
public:
   VectorFEDomainLFCurlIntegratorMeshSens(
       mach::VectorFEDomainLFCurlIntegrator &integ)
    : integ(integ), adjoint(nullptr)
   { }

   /// \brief - assemble an element's contribution to d(psi^T f)/dX
   /// \param[in] el - the finite element that describes the mesh element
   /// \param[in] trans - the transformation between reference and physical
   /// space \param[out] mesh_coords_bar - d(psi^T f)/dX for the element \note
   /// the LinearForm that assembles this integrator's FiniteElementSpace
   ///       MUST be the mesh's nodal finite element space
   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &mesh_coords_bar) override;

   inline void setAdjoint(const mfem::GridFunction &psi) { adjoint = &psi; }

private:
   /// reference to primal integrator
   mach::VectorFEDomainLFCurlIntegrator &integ;
   /// the adjoint to use when evaluating d(psi^T f)/dX
   const mfem::GridFunction *adjoint;
#ifndef MFEM_THREAD_SAFE
   mfem::Array<int> vdofs;
   mfem::Vector psi;
   mfem::DenseMatrix curlshape;
   mfem::DenseMatrix curlshape_bar, PointMat_bar;
#endif
};

template <>
inline void addSensitivityIntegrator<VectorFEDomainLFCurlIntegrator>(
    VectorFEDomainLFCurlIntegrator &primal_integ,
    std::unordered_map<std::string, mfem::ParGridFunction> &fields,
    std::map<std::string, mfem::ParLinearForm> &sens,
    std::map<std::string, mfem::ParNonlinearForm> &scalar_sens)
{
   auto *mesh_fes = fields.at("mesh_coords").ParFESpace();
   sens.emplace("mesh_coords", mesh_fes);
   auto *sens_integ = new VectorFEDomainLFCurlIntegratorMeshSens(primal_integ);
   sens_integ->setAdjoint(fields.at("adjoint"));
   sens.at("mesh_coords").AddDomainIntegrator(sens_integ);
}

/** Not yet differentiated, class only needed if magnets are on the boundary
    and not normal to boundary
class VectorFEBoundaryTangentLFIntegrator final
   : public mfem::VectorFEBoundaryTangentLFIntegrator
{
public:
   VectorFEBoundaryTangentLFIntegrator(mfem::VectorCoefficient &V,
                                       double alpha = 1.0)
   : mfem::VectorFEBoundaryTangentLFIntegrator(V), F(V), alpha(alpha)
   { }

   inline void AssembleRHSElementVect(const FiniteElement &el,
                                      ElementTransformation &trans,
                                      Vector &elvect) override
   {
      mfem::VectorFEBoundaryTangentLFIntegrator::
         AssembleRHSElementVect(el, trans, elvect);
      if (alpha != 1.0)
         elvect *= alpha;
   }

private:
   /// vector coefficient from linear form
   mfem::VectorCoefficient &F;
   /// scaling term if the linear form has a negative sign in the residual
   const double alpha;
   /// class that implements mesh sensitivities for
VectorFEBoundaryTangentLFIntegrator friend class
VectorFEBoundaryTangentLFIntegratorMeshSens;
};

/// Not yet differentiated
class VectorFEBoundaryTangentLFIntegratorMeshSens final
   : public mfem::LinearFormIntegrator
{
public:
   VectorFEBoundaryTangentLFIntegratorMeshSens(
      mach::VectorFEBoundaryTangentLFIntegrator &integ)
   : integ(integ), adjoint(nullptr)
   { }

   /// \brief - assemble an element's contribution to d(psi^T f)/dX
   /// \param[in] el - the finite element that describes the mesh element
   /// \param[in] trans - the transformation between reference and physical
space
   /// \param[out] mesh_coords_bar - d(psi^T f)/dX for the element
   /// \note the LinearForm that assembles this integrator's FiniteElementSpace
   ///       MUST be the mesh's nodal finite element space
   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &mesh_coords_bar) override;

   inline void setAdjoint(const mfem::GridFunction &psi)
   { adjoint = &psi; }

private:
   /// reference to primal integrator
   mach::VectorFEBoundaryTangentLFIntegrator &integ;
   /// the adjoint to use when evaluating d(psi^T f)/dX
   const mfem::GridFunction *adjoint;
#ifndef MFEM_THREAD_SAFE
   mfem::Array<int> vdofs;
   mfem::Vector psi;
   mfem::DenseMatrix vshape;
   mfem::DenseMatrix vshape_bar, PointMat_bar;
#endif
};
*/

class TestLFIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   TestLFIntegrator(mfem::Coefficient &Q) : Q(Q) { }

   double GetElementEnergy(const mfem::FiniteElement &el,
                           mfem::ElementTransformation &trans,
                           const mfem::Vector &elfun) override;

private:
   mfem::Coefficient &Q;
};

class TestLFMeshSensIntegrator : public mfem::LinearFormIntegrator
{
public:
   TestLFMeshSensIntegrator(mfem::Coefficient &Q) : Q(Q) { }

   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &elvect) override;

private:
   mfem::Coefficient &Q;
};

/// Class that evaluates the residual part and derivatives
/// for a DomainLFIntegrator (lininteg)
class DomainResIntegrator : public mfem::NonlinearFormIntegrator
{
   mfem::Vector shape;
   mfem::Coefficient &Q;
   int oa, ob;
   mfem::GridFunction *adjoint;

public:
   /// Constructs a domain integrator with a given Coefficient
   DomainResIntegrator(mfem::Coefficient &QF,
                       mfem::GridFunction *adj,
                       int a = 2,
                       int b = 0)
    : Q(QF), oa(a), ob(b), adjoint(adj)
   { }

   /// Constructs a domain integrator with a given Coefficient
   DomainResIntegrator(mfem::Coefficient &QF,
                       mfem::GridFunction *adj,
                       const mfem::IntegrationRule *ir)
    : Q(QF), oa(1), ob(1), adjoint(adj)
   { }

   /// Computes the residual contribution
   // virtual double GetElementEnergy(const FiniteElement &elx,
   //                                    ElementTransformation &Trx,
   //                                    const Vector &elfunx);

   /// Computes dR/dX, X being mesh node locations
   void AssembleElementVector(const mfem::FiniteElement &elx,
                              mfem::ElementTransformation &Trx,
                              const mfem::Vector &elfunx,
                              mfem::Vector &elvect) override;

   /// Computes R at an integration point
   double calcFunctional(int elno,
                         const mfem::IntegrationPoint &ip,
                         mfem::Vector &x_q,
                         mfem::ElementTransformation &Tr,
                         mfem::DenseMatrix &Jac_q);

   /// Computes dR/dX at an integration point using reverse mode
   static double calcFunctionalRevDiff(int elno,
                                       mfem::IntegrationPoint &ip,
                                       mfem::Vector &x_q,
                                       mfem::ElementTransformation &Tr,
                                       mfem::DenseMatrix &Jac_q,
                                       mfem::Vector &x_bar,
                                       mfem::DenseMatrix &Jac_bar)
   {
      return 0.0;
   }
};

/// Class that evaluates the residual part and derivatives
/// for a MassIntegrator (bilininteg)
class MassResIntegrator : public mfem::NonlinearFormIntegrator
{
protected:
   mfem::Vector shape;
   mfem::Coefficient *Q;
   mfem::GridFunction *state;
   mfem::GridFunction *adjoint;

public:
   /// Constructs a domain integrator with a given Coefficient
   MassResIntegrator(mfem::GridFunction *u,
                     mfem::GridFunction *adj,
                     const mfem::IntegrationRule *ir = nullptr)
    : Q(nullptr), state(u), adjoint(adj)
   { }

   /// Constructs a domain integrator with a given Coefficient
   MassResIntegrator(mfem::Coefficient &QF,
                     mfem::GridFunction *u,
                     mfem::GridFunction *adj,
                     const mfem::IntegrationRule *ir = nullptr)
    : Q(&QF), state(u), adjoint(adj)
   { }

   /// Computes the residual contribution
   // virtual double GetElementEnergy(const FiniteElement &elx,
   //                                    ElementTransformation &Trx,
   //                                    const Vector &elfun);

   /// Computes dR/dX, X being mesh node locations
   void AssembleElementVector(const mfem::FiniteElement &elx,
                              mfem::ElementTransformation &Trx,
                              const mfem::Vector &elfunx,
                              mfem::Vector &elvect) override;

   /// Computes R at an integration point
   double calcFunctional(int elno,
                         mfem::IntegrationPoint &ip,
                         mfem::Vector &x_q,
                         mfem::ElementTransformation &Tr,
                         mfem::DenseMatrix &Jac_q);

   /// Computes dR/dX at an integration point using reverse mode
   static double calcFunctionalRevDiff(int elno,
                                       mfem::IntegrationPoint &ip,
                                       mfem::Vector &x_q,
                                       mfem::ElementTransformation &Tr,
                                       mfem::DenseMatrix &Jac_q,
                                       mfem::Vector &x_bar,
                                       mfem::DenseMatrix &Jac_bar)
   {
      return 0.0;
   }
};

/// Class that evaluates the residual part and derivatives
/// for a DiffusionIntegrator (bilininteg)
/// NOTE: MatrixCoefficient not implemented
class DiffusionResIntegrator : public mfem::NonlinearFormIntegrator,
                               public mfem::LinearFormIntegrator
{
protected:
   mfem::Vector shape;
   mfem::DenseMatrix dshape;
   mfem::Coefficient *Q;
   mfem::GridFunction *state;
   mfem::GridFunction *adjoint;

public:
   /// Constructs a domain integrator with a given Coefficient
   DiffusionResIntegrator(mfem::Coefficient &QF,
                          mfem::GridFunction *u,
                          mfem::GridFunction *adj,
                          const mfem::IntegrationRule *ir = nullptr)
    : Q(&QF), state(u), adjoint(adj)
   { }

   /// Computes the residual contribution
   // virtual double GetElementEnergy(const FiniteElement &elx,
   //                                    ElementTransformation &Trx,
   //                                    const Vector &elfun);

   /// Computes dR/dX, X being mesh node locations
   void AssembleElementVector(const mfem::FiniteElement &elx,
                              mfem::ElementTransformation &Trx,
                              const mfem::Vector &elfunx,
                              mfem::Vector &elvect) override;

   /// Computes dR/dX, X being mesh node locations
   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &Trx,
                               mfem::Vector &elvect) override;
};

/// Class that evaluates the residual part and derivatives
/// for a BoundaryNormalLFIntegrator (lininteg)
/// NOTE: Add using AddBdrFaceIntegrator
class BoundaryNormalResIntegrator : public mfem::LinearFormIntegrator
{
   mfem::Vector shape;
   mfem::VectorCoefficient &Q;
   int oa, ob;
   mfem::GridFunction *state;
   mfem::GridFunction *adjoint;

public:
   /// Constructs a boundary integrator with a given Coefficient
   BoundaryNormalResIntegrator(mfem::VectorCoefficient &QF,
                               mfem::GridFunction *u,
                               mfem::GridFunction *adj,
                               int a = 2,
                               int b = 0)
    : Q(QF), oa(a), ob(b), state(u), adjoint(adj)
   { }

   /// Computes dR/dX, X being mesh node locations (DO NOT USE)
   void AssembleRHSElementVect(const mfem::FiniteElement &elx,
                               mfem::ElementTransformation &Trx,
                               mfem::Vector &elvect) override
   { }

   /// Computes dR/dX, X being mesh node locations
   void AssembleRHSElementVect(const mfem::FiniteElement &elx,
                               mfem::FaceElementTransformations &Trx,
                               mfem::Vector &elvect) override;
};

}  // namespace mach

#endif