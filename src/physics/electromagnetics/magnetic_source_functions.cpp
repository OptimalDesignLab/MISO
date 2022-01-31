#include <memory>
#include <vector>

#include "adept.h"
#include "mfem.hpp"

#include "magnetic_source_functions.hpp"

namespace
{
using adept::adouble;

template <typename xdouble = double>
void north_magnetization(const xdouble &remnant_flux,
                         const xdouble *x,
                         xdouble *M)
{
   xdouble r[] = {0.0, 0.0, 0.0};
   r[0] = x[0];
   r[1] = x[1];
   xdouble norm_r = sqrt(r[0] * r[0] + r[1] * r[1]);
   M[0] = r[0] * remnant_flux / norm_r;
   M[1] = r[1] * remnant_flux / norm_r;
   M[2] = 0.0;
}

template <typename xdouble = double>
void south_magnetization(const xdouble &remnant_flux,
                         const xdouble *x,
                         xdouble *M)
{
   xdouble r[] = {0.0, 0.0, 0.0};
   r[0] = x[0];
   r[1] = x[1];
   xdouble norm_r = sqrt(r[0] * r[0] + r[1] * r[1]);
   M[0] = -r[0] * remnant_flux / norm_r;
   M[1] = -r[1] * remnant_flux / norm_r;
   M[2] = 0.0;
}

template <typename xdouble = double>
void cw_magnetization(const xdouble &remnant_flux, const xdouble *x, xdouble *M)
{
   xdouble r[] = {0.0, 0.0, 0.0};
   r[0] = x[0];
   r[1] = x[1];
   xdouble norm_r = sqrt(r[0] * r[0] + r[1] * r[1]);
   M[0] = -r[1] * remnant_flux / norm_r;
   M[1] = r[0] * remnant_flux / norm_r;
   M[2] = 0.0;
}

template <typename xdouble = double>
void ccw_magnetization(const xdouble &remnant_flux,
                       const xdouble *x,
                       xdouble *M)
{
   xdouble r[] = {0.0, 0.0, 0.0};
   r[0] = x[0];
   r[1] = x[1];
   xdouble norm_r = sqrt(r[0] * r[0] + r[1] * r[1]);
   M[0] = r[1] * remnant_flux / norm_r;
   M[1] = -r[0] * remnant_flux / norm_r;
   M[2] = 0.0;
}

template <typename xdouble = double>
void x_axis_magnetization(const xdouble &remnant_flux,
                          const xdouble *x,
                          xdouble *M)
{
   M[0] = remnant_flux;
   M[1] = 0.0;
   M[2] = 0.0;
}

template <typename xdouble = double>
void y_axis_magnetization(const xdouble &remnant_flux,
                          const xdouble *x,
                          xdouble *M)
{
   M[0] = 0.0;
   M[1] = remnant_flux;
   M[2] = 0.0;
}

template <typename xdouble = double>
void z_axis_magnetization(const xdouble &remnant_flux,
                          const xdouble *x,
                          xdouble *M)
{
   M[0] = 0.0;
   M[1] = 0.0;
   M[2] = remnant_flux;
}

/// function describing permanent magnet magnetization pointing outwards
/// \param[in] x - position x in space
/// \param[out] M - magetic flux density at position x cause by permanent
///                 magnets
void northMagnetizationSource(double remnant_flux,
                              const mfem::Vector &x,
                              mfem::Vector &M)
{
   north_magnetization(remnant_flux, x.GetData(), M.GetData());
}

/// \param[in] x - position x in space of evaluation
/// \param[in] V_bar -
/// \param[out] x_bar - V_bar^T Jacobian
void northMagnetizationSourceRevDiff(adept::Stack &diff_stack,
                                     double remnant_flux,
                                     const mfem::Vector &x,
                                     const mfem::Vector &V_bar,
                                     mfem::Vector &x_bar)
{
   mfem::DenseMatrix source_jac(3);
   // declare vectors of active input variables
   std::vector<adouble> x_a(x.Size());
   // copy data from mfem::Vector
   adept::set_values(x_a.data(), x.Size(), x.GetData());
   // start recording
   diff_stack.new_recording();
   // the depedent variable must be declared after the recording
   std::vector<adouble> M_a(x.Size());
   north_magnetization<adouble>(remnant_flux, x_a.data(), M_a.data());
   // set the independent and dependent variable
   diff_stack.independent(x_a.data(), x.Size());
   diff_stack.dependent(M_a.data(), x.Size());
   // calculate the jacobian w.r.t position
   diff_stack.jacobian(source_jac.GetData());
   source_jac.MultTranspose(V_bar, x_bar);
}

/// function describing permanent magnet magnetization pointing inwards
/// \param[in] x - position x in space
/// \param[out] M - magetic flux density at position x cause by permanent
///                 magnets
void southMagnetizationSource(double remnant_flux,
                              const mfem::Vector &x,
                              mfem::Vector &M)
{
   south_magnetization(remnant_flux, x.GetData(), M.GetData());
}

/// \param[in] x - position x in space of evaluation
/// \param[in] V_bar -
/// \param[out] x_bar - V_bar^T Jacobian
void southMagnetizationSourceRevDiff(adept::Stack &diff_stack,
                                     double remnant_flux,
                                     const mfem::Vector &x,
                                     const mfem::Vector &V_bar,
                                     mfem::Vector &x_bar)
{
   mfem::DenseMatrix source_jac(3);
   // declare vectors of active input variables
   std::vector<adouble> x_a(x.Size());
   // copy data from mfem::Vector
   adept::set_values(x_a.data(), x.Size(), x.GetData());
   // start recording
   diff_stack.new_recording();
   // the depedent variable must be declared after the recording
   std::vector<adouble> M_a(x.Size());
   south_magnetization<adouble>(remnant_flux, x_a.data(), M_a.data());
   // set the independent and dependent variable
   diff_stack.independent(x_a.data(), x.Size());
   diff_stack.dependent(M_a.data(), x.Size());
   // calculate the jacobian w.r.t position
   diff_stack.jacobian(source_jac.GetData());
   source_jac.MultTranspose(V_bar, x_bar);
}

/// function describing permanent magnet magnetization pointing inwards
/// \param[in] x - position x in space
/// \param[out] M - magetic flux density at position x cause by permanent
///                 magnets
void cwMagnetizationSource(double remnant_flux,
                           const mfem::Vector &x,
                           mfem::Vector &M)
{
   cw_magnetization(remnant_flux, x.GetData(), M.GetData());
}

/// \param[in] x - position x in space of evaluation
/// \param[in] V_bar -
/// \param[out] x_bar - V_bar^T Jacobian
void cwMagnetizationSourceRevDiff(adept::Stack &diff_stack,
                                  double remnant_flux,
                                  const mfem::Vector &x,
                                  const mfem::Vector &V_bar,
                                  mfem::Vector &x_bar)
{
   mfem::DenseMatrix source_jac(3);
   // declare vectors of active input variables
   std::vector<adouble> x_a(x.Size());
   // copy data from mfem::Vector
   adept::set_values(x_a.data(), x.Size(), x.GetData());
   // start recording
   diff_stack.new_recording();
   // the depedent variable must be declared after the recording
   std::vector<adouble> M_a(x.Size());
   cw_magnetization<adouble>(remnant_flux, x_a.data(), M_a.data());
   // set the independent and dependent variable
   diff_stack.independent(x_a.data(), x.Size());
   diff_stack.dependent(M_a.data(), x.Size());
   // calculate the jacobian w.r.t position
   diff_stack.jacobian(source_jac.GetData());
   source_jac.MultTranspose(V_bar, x_bar);
}

/// function describing permanent magnet magnetization pointing inwards
/// \param[in] x - position x in space
/// \param[out] M - magetic flux density at position x cause by permanent
///                 magnets
void ccwMagnetizationSource(double remnant_flux,
                            const mfem::Vector &x,
                            mfem::Vector &M)
{
   ccw_magnetization(remnant_flux, x.GetData(), M.GetData());
}

/// \param[in] x - position x in space of evaluation
/// \param[in] V_bar -
/// \param[out] x_bar - V_bar^T Jacobian
void ccwMagnetizationSourceRevDiff(adept::Stack &diff_stack,
                                   double remnant_flux,
                                   const mfem::Vector &x,
                                   const mfem::Vector &V_bar,
                                   mfem::Vector &x_bar)
{
   mfem::DenseMatrix source_jac(3);
   // declare vectors of active input variables
   std::vector<adouble> x_a(x.Size());
   // copy data from mfem::Vector
   adept::set_values(x_a.data(), x.Size(), x.GetData());
   // start recording
   diff_stack.new_recording();
   // the depedent variable must be declared after the recording
   std::vector<adouble> M_a(x.Size());
   ccw_magnetization<adouble>(remnant_flux, x_a.data(), M_a.data());
   // set the independent and dependent variable
   diff_stack.independent(x_a.data(), x.Size());
   diff_stack.dependent(M_a.data(), x.Size());
   // calculate the jacobian w.r.t position
   diff_stack.jacobian(source_jac.GetData());
   source_jac.MultTranspose(V_bar, x_bar);
}

/// function defining magnetization aligned with the x axis
/// \param[in] x - position x in space of evaluation
/// \param[out] J - current density at position x
void xAxisMagnetizationSource(double remnant_flux,
                              const mfem::Vector &x,
                              mfem::Vector &M)
{
   x_axis_magnetization(remnant_flux, x.GetData(), M.GetData());
}

/// function defining magnetization aligned with the x axis
/// \param[in] x - position x in space of evaluation
/// \param[in] V_bar -
/// \param[out] x_bar - V_bar^T Jacobian
void xAxisMagnetizationSourceRevDiff(adept::Stack &diff_stack,
                                     double remnant_flux,
                                     const mfem::Vector &x,
                                     const mfem::Vector &V_bar,
                                     mfem::Vector &x_bar)
{
   mfem::DenseMatrix source_jac(3);
   // declare vectors of active input variables
   std::vector<adouble> x_a(x.Size());
   // copy data from mfem::Vector
   adept::set_values(x_a.data(), x.Size(), x.GetData());
   // start recording
   diff_stack.new_recording();
   // the depedent variable must be declared after the recording
   std::vector<adouble> M_a(x.Size());
   x_axis_magnetization<adouble>(remnant_flux, x_a.data(), M_a.data());
   // set the independent and dependent variable
   diff_stack.independent(x_a.data(), x.Size());
   diff_stack.dependent(M_a.data(), x.Size());
   // calculate the jacobian w.r.t state vaiables
   diff_stack.jacobian(source_jac.GetData());
   source_jac.MultTranspose(V_bar, x_bar);
}

/// function defining magnetization aligned with the y axis
/// \param[in] x - position x in space of evaluation
/// \param[out] J - current density at position x
void yAxisMagnetizationSource(double remnant_flux,
                              const mfem::Vector &x,
                              mfem::Vector &M)
{
   y_axis_magnetization(remnant_flux, x.GetData(), M.GetData());
}

/// function defining magnetization aligned with the x axis
/// \param[in] x - position x in space of evaluation
/// \param[in] V_bar -
/// \param[out] x_bar - V_bar^T Jacobian
void yAxisMagnetizationSourceRevDiff(adept::Stack &diff_stack,
                                     double remnant_flux,
                                     const mfem::Vector &x,
                                     const mfem::Vector &V_bar,
                                     mfem::Vector &x_bar)
{
   mfem::DenseMatrix source_jac(3);
   // declare vectors of active input variables
   std::vector<adouble> x_a(x.Size());
   // copy data from mfem::Vector
   adept::set_values(x_a.data(), x.Size(), x.GetData());
   // start recording
   diff_stack.new_recording();
   // the depedent variable must be declared after the recording
   std::vector<adouble> M_a(x.Size());
   y_axis_magnetization<adouble>(remnant_flux, x_a.data(), M_a.data());
   // set the independent and dependent variable
   diff_stack.independent(x_a.data(), x.Size());
   diff_stack.dependent(M_a.data(), x.Size());
   // calculate the jacobian w.r.t state vaiables
   diff_stack.jacobian(source_jac.GetData());
   source_jac.MultTranspose(V_bar, x_bar);
}

/// function defining magnetization aligned with the z axis
/// \param[in] x - position x in space of evaluation
/// \param[out] J - current density at position x
void zAxisMagnetizationSource(double remnant_flux,
                              const mfem::Vector &x,
                              mfem::Vector &M)
{
   z_axis_magnetization(remnant_flux, x.GetData(), M.GetData());
}

/// function defining magnetization aligned with the x axis
/// \param[in] x - position x in space of evaluation
/// \param[in] V_bar -
/// \param[out] x_bar - V_bar^T Jacobian
void zAxisMagnetizationSourceRevDiff(adept::Stack &diff_stack,
                                     double remnant_flux,
                                     const mfem::Vector &x,
                                     const mfem::Vector &V_bar,
                                     mfem::Vector &x_bar)
{
   mfem::DenseMatrix source_jac(3);
   // declare vectors of active input variables
   std::vector<adouble> x_a(x.Size());
   // copy data from mfem::Vector
   adept::set_values(x_a.data(), x.Size(), x.GetData());
   // start recording
   diff_stack.new_recording();
   // the depedent variable must be declared after the recording
   std::vector<adouble> M_a(x.Size());
   z_axis_magnetization<adouble>(remnant_flux, x_a.data(), M_a.data());
   // set the independent and dependent variable
   diff_stack.independent(x_a.data(), x.Size());
   diff_stack.dependent(M_a.data(), x.Size());
   // calculate the jacobian w.r.t state vaiables
   diff_stack.jacobian(source_jac.GetData());
   source_jac.MultTranspose(V_bar, x_bar);
}

}  // anonymous namespace

namespace mach
{
void MagnetizationCoefficient::Eval(mfem::Vector &V,
                                    mfem::ElementTransformation &trans,
                                    const mfem::IntegrationPoint &ip)
{
   mag_coeff.Eval(V, trans, ip);
}

void MagnetizationCoefficient::EvalRevDiff(const mfem::Vector &V_bar,
                                           mfem::ElementTransformation &trans,
                                           const mfem::IntegrationPoint &ip,
                                           mfem::DenseMatrix &PointMat_bar)
{
   mag_coeff.EvalRevDiff(V_bar, trans, ip, PointMat_bar);
}

MagnetizationCoefficient::MagnetizationCoefficient(
    adept::Stack &diff_stack,
    const nlohmann::json &magnet_options,
    const nlohmann::json &materials,
    int vdim)
 : mfem::VectorCoefficient(vdim), mag_coeff(vdim)
{
   for (auto &[material, group_details] : magnet_options.items())
   {
      remnant_flux_map.emplace(material,
                               materials[material]["B_r"].get<double>());
      auto &remnant_flux = remnant_flux_map.at(material);

      for (auto &[source, attrs] : group_details.items())
      {
         if (source == "north")
         {
            for (auto &attr : attrs)
            {
               mag_coeff.addCoefficient(
                   attr,
                   std::make_unique<mfem::VectorFunctionCoefficient>(
                       vdim,
                       [&remnant_flux](const mfem::Vector &x, mfem::Vector &M)
                       { northMagnetizationSource(remnant_flux, x, M); },
                       [&diff_stack, &remnant_flux](const mfem::Vector &x,
                                                    const mfem::Vector &M_bar,
                                                    mfem::Vector &x_bar)
                       {
                          northMagnetizationSourceRevDiff(
                              diff_stack, remnant_flux, x, M_bar, x_bar);
                       }));
            }
         }
         else if (source == "south")
         {
            for (auto &attr : attrs)
            {
               mag_coeff.addCoefficient(
                   attr,
                   std::make_unique<mfem::VectorFunctionCoefficient>(
                       vdim,
                       [&remnant_flux](const mfem::Vector &x, mfem::Vector &M)
                       { southMagnetizationSource(remnant_flux, x, M); },
                       [&diff_stack, &remnant_flux](const mfem::Vector &x,
                                                    const mfem::Vector &M_bar,
                                                    mfem::Vector &x_bar)
                       {
                          southMagnetizationSourceRevDiff(
                              diff_stack, remnant_flux, x, M_bar, x_bar);
                       }));
            }
         }
         else if (source == "cw")
         {
            for (auto &attr : attrs)
            {
               mag_coeff.addCoefficient(
                   attr,
                   std::make_unique<mfem::VectorFunctionCoefficient>(
                       vdim,
                       [&remnant_flux](const mfem::Vector &x, mfem::Vector &M)
                       { cwMagnetizationSource(remnant_flux, x, M); },
                       [&diff_stack, &remnant_flux](const mfem::Vector &x,
                                                    const mfem::Vector &M_bar,
                                                    mfem::Vector &x_bar) {
                          cwMagnetizationSourceRevDiff(
                              diff_stack, remnant_flux, x, M_bar, x_bar);
                       }));
            }
         }
         else if (source == "ccw")
         {
            for (auto &attr : attrs)
            {
               mag_coeff.addCoefficient(
                   attr,
                   std::make_unique<mfem::VectorFunctionCoefficient>(
                       vdim,
                       [&remnant_flux](const mfem::Vector &x, mfem::Vector &M)
                       { ccwMagnetizationSource(remnant_flux, x, M); },
                       [&diff_stack, &remnant_flux](const mfem::Vector &x,
                                                    const mfem::Vector &M_bar,
                                                    mfem::Vector &x_bar) {
                          ccwMagnetizationSourceRevDiff(
                              diff_stack, remnant_flux, x, M_bar, x_bar);
                       }));
            }
         }
         else if (source == "x")
         {
            for (auto &attr : attrs)
            {
               mag_coeff.addCoefficient(
                   attr,
                   std::make_unique<mfem::VectorFunctionCoefficient>(
                       vdim,
                       [&remnant_flux](const mfem::Vector &x, mfem::Vector &M)
                       { xAxisMagnetizationSource(remnant_flux, x, M); },
                       [&diff_stack, &remnant_flux](const mfem::Vector &x,
                                                    const mfem::Vector &M_bar,
                                                    mfem::Vector &x_bar)
                       {
                          xAxisMagnetizationSourceRevDiff(
                              diff_stack, remnant_flux, x, M_bar, x_bar);
                       }));
            }
         }
         else if (source == "y")
         {
            for (auto &attr : attrs)
            {
               mag_coeff.addCoefficient(
                   attr,
                   std::make_unique<mfem::VectorFunctionCoefficient>(
                       vdim,
                       [&remnant_flux](const mfem::Vector &x, mfem::Vector &M)
                       { yAxisMagnetizationSource(remnant_flux, x, M); },
                       [&diff_stack, &remnant_flux](const mfem::Vector &x,
                                                    const mfem::Vector &M_bar,
                                                    mfem::Vector &x_bar)
                       {
                          yAxisMagnetizationSourceRevDiff(
                              diff_stack, remnant_flux, x, M_bar, x_bar);
                       }));
            }
         }
         else if (source == "z")
         {
            for (auto &attr : attrs)
            {
               mag_coeff.addCoefficient(
                   attr,
                   std::make_unique<mfem::VectorFunctionCoefficient>(
                       vdim,
                       [&remnant_flux](const mfem::Vector &x, mfem::Vector &M)
                       { zAxisMagnetizationSource(remnant_flux, x, M); },
                       [&diff_stack, &remnant_flux](const mfem::Vector &x,
                                                    const mfem::Vector &M_bar,
                                                    mfem::Vector &x_bar)
                       {
                          zAxisMagnetizationSourceRevDiff(
                              diff_stack, remnant_flux, x, M_bar, x_bar);
                       }));
            }
         }
      }
   }
}

}  // namespace mach
