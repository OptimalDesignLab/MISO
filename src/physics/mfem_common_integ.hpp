#ifndef MACH_MFEM_COMMON_INTEG
#define MACH_MFEM_COMMON_INTEG

#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "finite_element_state.hpp"
#include "mach_input.hpp"
#include "mach_integrator.hpp"

namespace mach
{
class StateCoefficient;
class VectorStateCoefficient;

class BoundaryNormalIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   BoundaryNormalIntegrator(mfem::Coefficient &kappa) : kappa(kappa) { }

   double GetFaceEnergy(const mfem::FiniteElement &el1,
                        const mfem::FiniteElement &el2,
                        mfem::FaceElementTransformations &trans,
                        const mfem::Vector &elfun) override;

   void AssembleFaceVector(const mfem::FiniteElement &el1,
                           const mfem::FiniteElement &el2,
                           mfem::FaceElementTransformations &trans,
                           const mfem::Vector &elfun,
                           mfem::Vector &elfun_bar) override;

private:
   mfem::Coefficient &kappa;

#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix dshape1;
   mfem::Vector dshape1dn;
#endif

   friend class BoundaryNormalIntegratorMeshSens;
};

class BoundaryNormalIntegratorMeshSens : public mfem::LinearFormIntegrator
{
public:
   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &mesh_coords_bar) override;

   /// \param[in] state - the state to use when evaluating df/dX
   /// \param[in] integ - reference to primal integrator
   BoundaryNormalIntegratorMeshSens(mfem::GridFunction &state,
                                    BoundaryNormalIntegrator &integ)
    : state(state), integ(integ)
   { }

private:
   /// the state to use when evaluating df/dX
   mfem::GridFunction &state;
   /// reference to primal integrator
   BoundaryNormalIntegrator &integ;
#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix PointMat_bar;
   mfem::Array<int> vdofs;
   mfem::Vector elfun;
#endif
};

inline void addBdrSensitivityIntegrator(
    BoundaryNormalIntegrator &primal_integ,
    std::map<std::string, FiniteElementState> &fields,
    std::map<std::string, mfem::ParLinearForm> &output_sens,
    std::map<std::string, mfem::ParNonlinearForm> &output_scalar_sens,
    mfem::Array<int> *attr_marker,
    std::string state_name)
{
   auto &mesh_fes = fields.at("mesh_coords").space();
   output_sens.emplace("mesh_coords", &mesh_fes);

   if (attr_marker == nullptr)
   {
      output_sens.at("mesh_coords")
          .AddDomainIntegrator(new BoundaryNormalIntegratorMeshSens(
              fields.at(state_name).gridFunc(), primal_integ));
   }
   else
   {
      output_sens.at("mesh_coords")
          .AddDomainIntegrator(new BoundaryNormalIntegratorMeshSens(
                                   fields.at(state_name).gridFunc(), primal_integ),
                               *attr_marker);
   }
}

class VolumeIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   double GetElementEnergy(const mfem::FiniteElement &el,
                           mfem::ElementTransformation &trans,
                           const mfem::Vector &elfun) override;

   void AssembleElementVector(const mfem::FiniteElement &el,
                              mfem::ElementTransformation &trans,
                              const mfem::Vector &elfun,
                              mfem::Vector &elfun_bar) override
   {
      elfun_bar.SetSize(elfun.Size());
      elfun_bar = 0.0;
   }

   VolumeIntegrator(mfem::Coefficient *rho = nullptr) : rho(rho) { }

private:
   /// Optional density coefficient to get mass
   mfem::Coefficient *rho;

   friend class VolumeIntegratorMeshSens;
};

class VolumeIntegratorMeshSens : public mfem::LinearFormIntegrator
{
public:
   /// \param[in] state - the state grid function
   /// \param[in] integ - reference to primal integrator that holds inputs for
   /// integrator
   VolumeIntegratorMeshSens(mfem::GridFunction &state, VolumeIntegrator &integ)
    : state(state), integ(integ)
   { }

   /// \brief - assemble an element's contribution to dJdX
   /// \param[in] mesh_el - the finite element that describes the mesh element
   /// \param[in] mesh_trans - the transformation between reference and physical
   /// space
   /// \param[out] mesh_coords_bar - dJdX for the element
   void AssembleRHSElementVect(const mfem::FiniteElement &mesh_el,
                               mfem::ElementTransformation &mesh_trans,
                               mfem::Vector &mesh_coords_bar) override;

private:
   /// State GridFunction, needed to get integration order for each element
   mfem::GridFunction &state;
   /// reference to primal integrator
   VolumeIntegrator &integ;

#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix PointMat_bar;
#endif
};

inline void addDomainSensitivityIntegrator(
    VolumeIntegrator &primal_integ,
    std::map<std::string, FiniteElementState> &fields,
    std::map<std::string, mfem::ParLinearForm> &output_sens,
    std::map<std::string, mfem::ParNonlinearForm> &output_scalar_sens,
    mfem::Array<int> *attr_marker,
    std::string state_name)
{
   auto &mesh_fes = fields.at("mesh_coords").space();
   output_sens.emplace("mesh_coords", &mesh_fes);

   if (attr_marker == nullptr)
   {
      output_sens.at("mesh_coords")
          .AddDomainIntegrator(new VolumeIntegratorMeshSens(
              fields.at(state_name).gridFunc(), primal_integ));
   }
   else
   {
      output_sens.at("mesh_coords")
          .AddDomainIntegrator(new VolumeIntegratorMeshSens(
                                   fields.at(state_name).gridFunc(), primal_integ),
                               *attr_marker);
   }
}

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

class MagnitudeCurlStateIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   double GetElementEnergy(const mfem::FiniteElement &el,
                           mfem::ElementTransformation &trans,
                           const mfem::Vector &elfun) override;

   void AssembleElementVector(const mfem::FiniteElement &el,
                              mfem::ElementTransformation &trans,
                              const mfem::Vector &elfun,
                              mfem::Vector &elfun_bar) override;

private:
#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix curlshape;
   mfem::DenseMatrix curlshape_dFt;
#endif

   friend class MagnitudeCurlStateIntegratorMeshSens;
};

class MagnitudeCurlStateIntegratorMeshSens : public mfem::LinearFormIntegrator
{
public:
   /// \param[in] integ - reference to primal integrator that holds inputs for
   /// integrator
   MagnitudeCurlStateIntegratorMeshSens(mfem::GridFunction &state,
                                        MagnitudeCurlStateIntegrator &integ)
    : state(state), integ(integ)
   { }

   /// \brief - assemble an element's contribution to dJdX
   /// \param[in] el - the finite element that describes the mesh element
   /// \param[in] trans - the transformation between reference and physical
   /// space
   /// \param[out] mesh_coords_bar - dJdX for the element
   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &mesh_coords_bar) override;

private:
   /// state vector for evaluating integrator
   mfem::GridFunction &state;
   /// reference to primal integrator
   MagnitudeCurlStateIntegrator &integ;

#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix curlshape_dFt_bar;
   mfem::DenseMatrix PointMat_bar;
   mfem::Array<int> vdofs;
   mfem::Vector elfun;
#endif
};

inline void addDomainSensitivityIntegrator(
    MagnitudeCurlStateIntegrator &primal_integ,
    std::map<std::string, FiniteElementState> &fields,
    std::map<std::string, mfem::ParLinearForm> &output_sens,
    std::map<std::string, mfem::ParNonlinearForm> &output_scalar_sens,
    mfem::Array<int> *attr_marker,
    std::string state_name)
{
   auto &mesh_fes = fields.at("mesh_coords").space();
   output_sens.emplace("mesh_coords", &mesh_fes);

   if (attr_marker == nullptr)
   {
      output_sens.at("mesh_coords")
          .AddDomainIntegrator(new MagnitudeCurlStateIntegratorMeshSens(
              fields.at(state_name).gridFunc(), primal_integ));
   }
   else
   {
      output_sens.at("mesh_coords")
          .AddDomainIntegrator(new MagnitudeCurlStateIntegratorMeshSens(
                                   fields.at(state_name).gridFunc(), primal_integ),
                               *attr_marker);
   }
}

class IEAggregateIntegratorNumerator : public mfem::NonlinearFormIntegrator
{
public:
   friend void setOptions(IEAggregateIntegratorNumerator &integ,
                          const nlohmann::json &options);

   friend void setInputs(IEAggregateIntegratorNumerator &integ,
                         const MachInputs &inputs);

   IEAggregateIntegratorNumerator(const double rho)
    : rho(rho)
   { }

   double GetElementEnergy(const mfem::FiniteElement &el,
                           mfem::ElementTransformation &trans,
                           const mfem::Vector &elfun) override;

   void AssembleElementVector(const mfem::FiniteElement &el,
                              mfem::ElementTransformation &trans,
                              const mfem::Vector &elfun,
                              mfem::Vector &elfun_bar) override;

private:
   /// aggregation parameter rho
   double rho;
   /// true max value - used to improve numerical conditioning
   double true_max = 1.0;
#ifndef MFEM_THREAD_SAFE
   mfem::Vector shape;
#endif
   friend class IEAggregateIntegratorNumeratorMeshSens;
};

class IEAggregateIntegratorNumeratorMeshSens : public mfem::LinearFormIntegrator
{
public:
   /// \brief - Compute forces/torques based on the virtual work method
   /// \param[in] state - the state vector to evaluate force at
   /// \param[in] integ - reference to primal integrator that holds inputs for
   /// integrators
   IEAggregateIntegratorNumeratorMeshSens(mfem::GridFunction &state,
                                          IEAggregateIntegratorNumerator &integ)
    : state(state), integ(integ)
   { }

   /// \brief - assemble an element's contribution to dJdX
   /// \param[in] el - the finite element that describes the mesh element
   /// \param[in] trans - the transformation between reference and physical
   /// space
   /// \param[out] mesh_coords_bar - dJdX for the element
   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &mesh_coords_bar) override;

private:
   /// state vector for evaluating integrator
   mfem::GridFunction &state;
   /// reference to primal integrator
   IEAggregateIntegratorNumerator &integ;

#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix PointMat_bar;
   mfem::Array<int> vdofs;
   mfem::Vector elfun;
#endif
};

inline void addDomainSensitivityIntegrator(
    IEAggregateIntegratorNumerator &primal_integ,
    std::map<std::string, FiniteElementState> &fields,
    std::map<std::string, mfem::ParLinearForm> &output_sens,
    std::map<std::string, mfem::ParNonlinearForm> &output_scalar_sens,
    mfem::Array<int> *attr_marker,
    std::string state_name)
{
   auto &mesh_fes = fields.at("mesh_coords").space();
   output_sens.emplace("mesh_coords", &mesh_fes);

   if (attr_marker == nullptr)
   {
      output_sens.at("mesh_coords")
          .AddDomainIntegrator(new IEAggregateIntegratorNumeratorMeshSens(
              fields.at(state_name).gridFunc(), primal_integ));
   }
   else
   {
      output_sens.at("mesh_coords")
          .AddDomainIntegrator(
              new IEAggregateIntegratorNumeratorMeshSens(
                  fields.at(state_name).gridFunc(), primal_integ),
              *attr_marker);
   }
}

class IEAggregateIntegratorDenominator : public mfem::NonlinearFormIntegrator
{
public:
   friend void setOptions(IEAggregateIntegratorDenominator &integ,
                          const nlohmann::json &options);

   friend void setInputs(IEAggregateIntegratorDenominator &integ,
                         const MachInputs &inputs);

   IEAggregateIntegratorDenominator(const double rho)
    : rho(rho)
   { }

   double GetElementEnergy(const mfem::FiniteElement &el,
                           mfem::ElementTransformation &trans,
                           const mfem::Vector &elfun) override;

   void AssembleElementVector(const mfem::FiniteElement &el,
                              mfem::ElementTransformation &trans,
                              const mfem::Vector &elfun,
                              mfem::Vector &elfun_bar) override;

private:
   /// aggregation parameter rho
   double rho;
   /// true max value - used to improve numerical conditioning
   double true_max = 1.0;
#ifndef MFEM_THREAD_SAFE
   mfem::Vector shape;
#endif
   friend class IEAggregateIntegratorDenominatorMeshSens;
};

class IEAggregateIntegratorDenominatorMeshSens
 : public mfem::LinearFormIntegrator
{
public:
   /// \brief - Compute forces/torques based on the virtual work method
   /// \param[in] state - the state vector to evaluate force at
   /// \param[in] integ - reference to primal integrator that holds inputs for
   /// integrators
   IEAggregateIntegratorDenominatorMeshSens(
       mfem::GridFunction &state,
       IEAggregateIntegratorDenominator &integ)
    : state(state), integ(integ)
   { }

   /// \brief - assemble an element's contribution to dJdX
   /// \param[in] el - the finite element that describes the mesh element
   /// \param[in] trans - the transformation between reference and physical
   /// space
   /// \param[out] mesh_coords_bar - dJdX for the element
   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &mesh_coords_bar) override;

private:
   /// state vector for evaluating integrator
   mfem::GridFunction &state;
   /// reference to primal integrator
   IEAggregateIntegratorDenominator &integ;

#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix PointMat_bar;
   mfem::Array<int> vdofs;
   mfem::Vector elfun;
#endif
};

inline void addDomainSensitivityIntegrator(
    IEAggregateIntegratorDenominator &primal_integ,
    std::map<std::string, FiniteElementState> &fields,
    std::map<std::string, mfem::ParLinearForm> &output_sens,
    std::map<std::string, mfem::ParNonlinearForm> &output_scalar_sens,
    mfem::Array<int> *attr_marker,
    std::string state_name)
{
   auto &mesh_fes = fields.at("mesh_coords").space();
   output_sens.emplace("mesh_coords", &mesh_fes);

   if (attr_marker == nullptr)
   {
      output_sens.at("mesh_coords")
          .AddDomainIntegrator(new IEAggregateIntegratorDenominatorMeshSens(
              fields.at(state_name).gridFunc(), primal_integ));
   }
   else
   {
      output_sens.at("mesh_coords")
          .AddDomainIntegrator(
              new IEAggregateIntegratorDenominatorMeshSens(
                  fields.at(state_name).gridFunc(), primal_integ),
              *attr_marker);
   }
}

class IECurlMagnitudeAggregateIntegratorNumerator
 : public mfem::NonlinearFormIntegrator
{
public:
   friend void setOptions(IECurlMagnitudeAggregateIntegratorNumerator &integ,
                          const nlohmann::json &options);

   IECurlMagnitudeAggregateIntegratorNumerator(double rho,
                                               double actual_max = 1.0)
    : rho(rho), actual_max(actual_max)
   { }

   double GetElementEnergy(const mfem::FiniteElement &el,
                           mfem::ElementTransformation &trans,
                           const mfem::Vector &elfun) override;

   void AssembleElementVector(const mfem::FiniteElement &el,
                              mfem::ElementTransformation &trans,
                              const mfem::Vector &elfun,
                              mfem::Vector &elfun_bar) override;

private:
   /// aggregation parameter rho
   double rho;
   /// actual maximum from the data, makes the calculation more stable
   double actual_max;
#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix curlshape, curlshape_dFt;
#endif
   friend class IECurlMagnitudeAggregateIntegratorNumeratorMeshSens;
};

class IECurlMagnitudeAggregateIntegratorNumeratorMeshSens
 : public mfem::LinearFormIntegrator
{
public:
   /// \brief - Compute forces/torques based on the virtual work method
   /// \param[in] state - the state vector to evaluate force at
   /// \param[in] integ - reference to primal integrator that holds inputs for
   /// integrators
   IECurlMagnitudeAggregateIntegratorNumeratorMeshSens(
       mfem::GridFunction &state,
       IECurlMagnitudeAggregateIntegratorNumerator &integ)
    : state(state), integ(integ)
   { }

   /// \brief - assemble an element's contribution to dJdX
   /// \param[in] el - the finite element that describes the mesh element
   /// \param[in] trans - the transformation between reference and physical
   /// space
   /// \param[out] mesh_coords_bar - dJdX for the element
   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &mesh_coords_bar) override;

private:
   /// state vector for evaluating integrator
   mfem::GridFunction &state;
   /// reference to primal integrator
   IECurlMagnitudeAggregateIntegratorNumerator &integ;

#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix curlshape_dFt_bar;
   mfem::DenseMatrix PointMat_bar;
   mfem::Array<int> vdofs;
   mfem::Vector elfun;
#endif
};

inline void addDomainSensitivityIntegrator(
    IECurlMagnitudeAggregateIntegratorNumerator &primal_integ,
    std::map<std::string, FiniteElementState> &fields,
    std::map<std::string, mfem::ParLinearForm> &output_sens,
    std::map<std::string, mfem::ParNonlinearForm> &output_scalar_sens,
    mfem::Array<int> *attr_marker,
    std::string state_name)
{
   auto &mesh_fes = fields.at("mesh_coords").space();
   output_sens.emplace("mesh_coords", &mesh_fes);

   if (attr_marker == nullptr)
   {
      output_sens.at("mesh_coords")
          .AddDomainIntegrator(
              new IECurlMagnitudeAggregateIntegratorNumeratorMeshSens(
                  fields.at(state_name).gridFunc(), primal_integ));
   }
   else
   {
      output_sens.at("mesh_coords")
          .AddDomainIntegrator(
              new IECurlMagnitudeAggregateIntegratorNumeratorMeshSens(
                  fields.at(state_name).gridFunc(), primal_integ),
              *attr_marker);
   }
}

class IECurlMagnitudeAggregateIntegratorDenominator
 : public mfem::NonlinearFormIntegrator
{
public:
   friend void setOptions(IECurlMagnitudeAggregateIntegratorDenominator &integ,
                          const nlohmann::json &options);

   IECurlMagnitudeAggregateIntegratorDenominator(const double rho,
                                                 double actual_max = 1.0)
    : rho(rho), actual_max(actual_max)
   { }

   double GetElementEnergy(const mfem::FiniteElement &el,
                           mfem::ElementTransformation &trans,
                           const mfem::Vector &elfun) override;

   void AssembleElementVector(const mfem::FiniteElement &el,
                              mfem::ElementTransformation &trans,
                              const mfem::Vector &elfun,
                              mfem::Vector &elfun_bar) override;

private:
   /// aggregation parameter rho
   double rho;
   /// actual maximum from the data, makes the calculation more stable
   double actual_max;
#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix curlshape, curlshape_dFt;
#endif
   friend class IECurlMagnitudeAggregateIntegratorDenominatorMeshSens;
};

class IECurlMagnitudeAggregateIntegratorDenominatorMeshSens
 : public mfem::LinearFormIntegrator
{
public:
   /// \brief - Compute forces/torques based on the virtual work method
   /// \param[in] state - the state vector to evaluate force at
   /// \param[in] integ - reference to primal integrator that holds inputs for
   /// integrators
   IECurlMagnitudeAggregateIntegratorDenominatorMeshSens(
       mfem::GridFunction &state,
       IECurlMagnitudeAggregateIntegratorDenominator &integ)
    : state(state), integ(integ)
   { }

   /// \brief - assemble an element's contribution to dJdX
   /// \param[in] el - the finite element that describes the mesh element
   /// \param[in] trans - the transformation between reference and physical
   /// space
   /// \param[out] mesh_coords_bar - dJdX for the element
   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &mesh_coords_bar) override;

private:
   /// state vector for evaluating force
   mfem::GridFunction &state;
   /// reference to primal integrator
   IECurlMagnitudeAggregateIntegratorDenominator &integ;

#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix curlshape_dFt_bar;
   mfem::DenseMatrix PointMat_bar;
   mfem::Array<int> vdofs;
   mfem::Vector elfun;
#endif
};

inline void addDomainSensitivityIntegrator(
    IECurlMagnitudeAggregateIntegratorDenominator &primal_integ,
    std::map<std::string, FiniteElementState> &fields,
    std::map<std::string, mfem::ParLinearForm> &output_sens,
    std::map<std::string, mfem::ParNonlinearForm> &output_scalar_sens,
    mfem::Array<int> *attr_marker,
    std::string state_name)
{
   auto &mesh_fes = fields.at("mesh_coords").space();
   output_sens.emplace("mesh_coords", &mesh_fes);

   if (attr_marker == nullptr)
   {
      output_sens.at("mesh_coords")
          .AddDomainIntegrator(
              new IECurlMagnitudeAggregateIntegratorDenominatorMeshSens(
                  fields.at(state_name).gridFunc(), primal_integ));
   }
   else
   {
      output_sens.at("mesh_coords")
          .AddDomainIntegrator(
              new IECurlMagnitudeAggregateIntegratorDenominatorMeshSens(
                  fields.at(state_name).gridFunc(), primal_integ),
              *attr_marker);
   }
}

// New induced exponential numerator integrator for demagnetization proximity
// constraint
class IEAggregateDemagIntegratorNumerator : public mfem::NonlinearFormIntegrator
{
public:
   friend void setOptions(IEAggregateDemagIntegratorNumerator &integ,
                          const nlohmann::json &options);

   friend void setInputs(IEAggregateDemagIntegratorNumerator &integ,
                         const MachInputs &inputs);

   IEAggregateDemagIntegratorNumerator(
       const double rho,
       StateCoefficient &B_knee,
       VectorStateCoefficient &mag_coeff,
       mfem::GridFunction *temperature_field = nullptr,
       mfem::GridFunction *flux_density_field = nullptr,
       std::string state_name = "state")
    : rho(rho),
      B_knee(B_knee),
      mag_coeff(mag_coeff),
      temperature_field(temperature_field),
      flux_density_field(flux_density_field),
      _state_name(std::move(state_name))
   { }

   double GetElementEnergy(const mfem::FiniteElement &el,
                           mfem::ElementTransformation &trans,
                           const mfem::Vector &elfun) override;

   void AssembleElementVector(const mfem::FiniteElement &el,
                              mfem::ElementTransformation &trans,
                              const mfem::Vector &elfun,
                              mfem::Vector &elfun_bar) override;

   const std::string &state_name() { return _state_name; }

private:
   /// aggregation parameter rho
   double rho;
   /// flux density at the knee point
   StateCoefficient &B_knee;
   /// magnetization in the permanent magnets
   VectorStateCoefficient &mag_coeff;
   // Fields being used
   mfem::GridFunction *temperature_field;   // temperature field
   mfem::GridFunction *flux_density_field;  // flux density field
   /// name of state integrating over - needed for mesh sens
   /// TODO: Determine how to handle state name, seeing as demagnetization is
   /// not a state or field per se
   std::string _state_name;
   /// true max value - used to improve numerical conditioning
   /// NOTE: This value is not necessarily known for demagnetization, but it
   /// should be on the order of 1
   double true_max = 1.0;
#ifndef MFEM_THREAD_SAFE
   mfem::Array<int> vdofs;
   mfem::Vector temp_shape;
   mfem::Vector temp_elfun;
   mfem::Vector B_shape;
   mfem::Vector B_elfun;
#endif
   /// TODO: If needed, 1) create and implement mesh sens class and 2)
   /// addDomainSensitivityIntegrator
   // friend class IEAggregateIntegratorDemagNumeratorMeshSens;
};

// New induced exponential denominator integrator for demagnetization proximity
// constraint
class IEAggregateDemagIntegratorDenominator
 : public mfem::NonlinearFormIntegrator
{
public:
   friend void setOptions(IEAggregateDemagIntegratorDenominator &integ,
                          const nlohmann::json &options);

   friend void setInputs(IEAggregateDemagIntegratorDenominator &integ,
                         const MachInputs &inputs);

   IEAggregateDemagIntegratorDenominator(
       const double rho,
       StateCoefficient &B_knee,
       VectorStateCoefficient &mag_coeff,
       mfem::GridFunction *temperature_field = nullptr,
       mfem::GridFunction *flux_density_field = nullptr,
       std::string state_name = "state")
    : rho(rho),
      B_knee(B_knee),
      mag_coeff(mag_coeff),
      temperature_field(temperature_field),
      flux_density_field(flux_density_field),
      _state_name(std::move(state_name))
   { }

   double GetElementEnergy(const mfem::FiniteElement &el,
                           mfem::ElementTransformation &trans,
                           const mfem::Vector &elfun) override;

   void AssembleElementVector(const mfem::FiniteElement &el,
                              mfem::ElementTransformation &trans,
                              const mfem::Vector &elfun,
                              mfem::Vector &elfun_bar) override;

   const std::string &state_name() { return _state_name; }

private:
   /// aggregation parameter rho
   double rho;
   // flux density at the knee point
   StateCoefficient &B_knee;
   /// magnetization in the permanent magnets
   VectorStateCoefficient &mag_coeff;
   // Fields being used
   mfem::GridFunction *temperature_field;   // temperature field
   mfem::GridFunction *flux_density_field;  // flux density field
   /// name of state integrating over - needed for mesh sens
   /// TODO: Determine how to handle state name, seeing as demagnetization is
   /// not a state or field per se
   std::string _state_name;
   /// true max value - used to improve numerical conditioning
   /// NOTE: This value is not necessarily known for demagnetization, but it
   /// should be on the order of 1
   double true_max = 1.0;
#ifndef MFEM_THREAD_SAFE
   mfem::Array<int> vdofs;
   mfem::Vector temp_shape;
   mfem::Vector temp_elfun;
   mfem::Vector B_shape;
   mfem::Vector B_elfun;
#endif
   /// TODO: If needed, 1) create and implement mesh sens class and 2)
   /// addDomainSensitivityIntegrator
   // friend class IEAggregateIntegratorDemagDenominatorMeshSens;
};

class DiffusionIntegratorMeshSens final : public mfem::LinearFormIntegrator
{
public:
   DiffusionIntegratorMeshSens() = default;

   /// \brief - assemble an element's contribution to d(psi^T D u)/dX
   /// \param[in] el - the finite element that describes the mesh element
   /// \param[in] trans - the transformation between reference and physical
   /// space
   /// \param[out] mesh_coords_bar - d(psi^T D u)/dX for the element
   /// \note the LinearForm that assembles this integrator's FiniteElementSpace
   /// MUST be the mesh's nodal finite element space
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

inline void addSensitivityIntegrator(
    mfem::DiffusionIntegrator &primal_integ,
    std::map<std::string, FiniteElementState> &fields,
    std::map<std::string, mfem::ParLinearForm> &rev_sens,
    std::map<std::string, mfem::ParNonlinearForm> &rev_scalar_sens,
    std::map<std::string, mfem::ParLinearForm> &fwd_sens,
    std::map<std::string, mfem::ParNonlinearForm> &fwd_scalar_sens)
{
   auto &mesh_fes = fields.at("mesh_coords").space();
   rev_sens.emplace("mesh_coords", &mesh_fes);
   rev_sens.at("mesh_coords")
       .AddDomainIntegrator(new DiffusionIntegratorMeshSens);
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

inline void addSensitivityIntegrator(
    mfem::VectorFEWeakDivergenceIntegrator &primal_integ,
    std::map<std::string, FiniteElementState> &fields,
    std::map<std::string, mfem::ParLinearForm> &rev_sens,
    std::map<std::string, mfem::ParNonlinearForm> &rev_scalar_sens,
    std::map<std::string, mfem::ParLinearForm> &fwd_sens,
    std::map<std::string, mfem::ParNonlinearForm> &fwd_scalar_sens)
{
   auto &mesh_fes = fields.at("mesh_coords").space();
   rev_sens.emplace("mesh_coords", &mesh_fes);
   rev_sens.at("mesh_coords")
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

inline void addSensitivityIntegrator(
    mfem::VectorFEMassIntegrator &primal_integ,
    std::map<std::string, FiniteElementState> &fields,
    std::map<std::string, mfem::ParLinearForm> &rev_sens,
    std::map<std::string, mfem::ParNonlinearForm> &rev_scalar_sens,
    std::map<std::string, mfem::ParLinearForm> &fwd_sens,
    std::map<std::string, mfem::ParNonlinearForm> &fwd_scalar_sens)
{
   auto &mesh_fes = fields.at("mesh_coords").space();
   rev_sens.emplace("mesh_coords", &mesh_fes);

   auto *integ = new VectorFEMassIntegratorMeshSens;
   integ->setState(fields.at("in").gridFunc());
   rev_sens.at("mesh_coords").AddDomainIntegrator(integ);
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
//    std::map<std::string, FiniteElementState> &fields,
//    std::map<std::string, mfem::ParLinearForm> &rev_sens,
//    std::map<std::string, mfem::ParNonlinearForm> &scalar_sens)
// {
//    auto mesh_fes = fields.at("mesh_coords").ParFESpace();
//    rev_sens.emplace("mesh_coords", mesh_fes);
//    rev_sens.at("mesh_coords").AddDomainIntegrator(
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

inline void addSensitivityIntegrator(
    VectorFEDomainLFCurlIntegrator &primal_integ,
    std::map<std::string, FiniteElementState> &fields,
    std::map<std::string, mfem::ParLinearForm> &rev_sens,
    std::map<std::string, mfem::ParNonlinearForm> &rev_scalar_sens,
    std::map<std::string, mfem::ParLinearForm> &fwd_sens,
    std::map<std::string, mfem::ParNonlinearForm> &fwd_scalar_sens)
{
   auto &mesh_fes = fields.at("mesh_coords").space();
   rev_sens.emplace("mesh_coords", &mesh_fes);
   auto *sens_integ = new VectorFEDomainLFCurlIntegratorMeshSens(primal_integ);
   sens_integ->setAdjoint(fields.at("adjoint").gridFunc());
   rev_sens.at("mesh_coords").AddDomainIntegrator(sens_integ);
}

class DomainLFIntegrator final : public mfem::DomainLFIntegrator
{
public:
   DomainLFIntegrator(mfem::Coefficient &Q, double alpha = 1.0)
    : mfem::DomainLFIntegrator(Q, 2, -2), F(Q), alpha(alpha)
   { }

   inline void AssembleRHSElementVect(const mfem::FiniteElement &el,
                                      mfem::ElementTransformation &trans,
                                      mfem::Vector &elvect) override
   {
      mfem::DomainLFIntegrator::AssembleRHSElementVect(el, trans, elvect);
      if (alpha != 1.0)
      {
         elvect *= alpha;
      }
   }

private:
   /// coefficient for linear form
   mfem::Coefficient &F;
   /// scaling term if the linear form has a negative sign in the residual
   const double alpha;
   /// class that implements mesh sensitivities for
   /// DomainLFIntegrator
   friend class DomainLFIntegratorMeshRevSens;
};

class DomainLFIntegratorMeshRevSens final : public mfem::LinearFormIntegrator
{
public:
   /// \param[in] adjoint - the adjoint to use when evaluating d(psi^T R)/dX
   /// \param[in] integ - reference to primal integrator
   DomainLFIntegratorMeshRevSens(mfem::GridFunction &adjoint,
                                 mach::DomainLFIntegrator &integ)
    : adjoint(adjoint), integ(integ)
   { }

   /// \brief - assemble an element's contribution to d(psi^T f)/dX
   /// \param[in] el - the finite element that describes the mesh element
   /// \param[in] trans - the transformation between reference and physical
   /// space
   /// \param[out] mesh_coords_bar - d(psi^T f)/dX for the element
   /// \note the LinearForm that assembles this integrator's FiniteElementSpace
   /// MUST be the mesh's nodal finite element space
   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &mesh_coords_bar) override;

private:
   /// the adjoint to use when evaluating d(psi^T R)/dX
   mfem::GridFunction &adjoint;
   /// reference to primal integrator
   mach::DomainLFIntegrator &integ;

#ifndef MFEM_THREAD_SAFE
   mfem::Vector shape, shape_bar;
   mfem::DenseMatrix PointMat_bar;
   mfem::Array<int> vdofs;
   mfem::Vector psi;
#endif
};

inline void addDomainSensitivityIntegrator(
    mach::DomainLFIntegrator &primal_integ,
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
          .AddDomainIntegrator(new DomainLFIntegratorMeshRevSens(
              fields.at(adjoint_name).gridFunc(), primal_integ));
   }
   else
   {
      rev_sens.at("mesh_coords")
          .AddDomainIntegrator(
              new DomainLFIntegratorMeshRevSens(
                  fields.at(adjoint_name).gridFunc(), primal_integ),
              *attr_marker);
   }
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
   /// \note the LinearForm that assembles this integrator's
FiniteElementSpace
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