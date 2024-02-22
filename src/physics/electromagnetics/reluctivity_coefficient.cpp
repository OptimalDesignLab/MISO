#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "mfem.hpp"
#include "nlohmann/json.hpp"
#include "tinysplinecxx.h"

#include "reluctivity_coefficient.hpp"
#include "utils.hpp"

namespace
{
/// permeability of free space
constexpr double mu_0 = 4e-7 * M_PI;
constexpr double nu0 = 1 / mu_0;

class logNuBBSplineReluctivityCoefficient : public miso::StateCoefficient
{
public:
   /// \brief Define a reluctivity model from a B-Spline fit of
   /// log(nu) as a function of B
   /// \param[in] cps - spline control points -> nu ~ exp(cps)
   /// \param[in] knots - spline knot vector -> B ~ knots
   /// \param[in] degree - degree of B-Spline curve
   logNuBBSplineReluctivityCoefficient(const std::vector<double> &cps,
                                       const std::vector<double> &knots,
                                       int degree = 3);

   /// \brief Evaluate the reluctivity in the element described by trans at the
   /// point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state) override;

   /// \brief Evaluate the derivative of reluctivity with respsect to B in the
   /// element described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double EvalStateDeriv(mfem::ElementTransformation &trans,
                         const mfem::IntegrationPoint &ip,
                         double state) override;

   double EvalState2ndDeriv(mfem::ElementTransformation &trans,
                            const mfem::IntegrationPoint &ip,
                            const double state) override;

   void EvalRevDiff(const double Q_bar,
                    mfem::ElementTransformation &trans,
                    const mfem::IntegrationPoint &ip,
                    double state,
                    mfem::DenseMatrix &PointMat_bar) override
   { }

protected:
   /// max nu value in the data
   double lognu_max;
   /// max B value in the data
   double b_max;
   /// spline representing log(nu)
   std::unique_ptr<tinyspline::BSpline> lognu;
   /// spline representing dlog(nu)/dB
   std::unique_ptr<tinyspline::BSpline> dlognudb;
   /// spline representing d2log(nu)/dB2
   std::unique_ptr<tinyspline::BSpline> d2lognudb2;
};

class BHBSplineReluctivityCoefficient : public miso::StateCoefficient
{
public:
   /// \brief Define a reluctivity model from a B-Spline fit with linear
   /// extrapolation at the far end
   /// \param[in] B - magnetic flux density values from B-H curve
   /// \param[in] H - magnetic field intensity valyes from B-H curve
   BHBSplineReluctivityCoefficient(const std::vector<double> &cps,
                                   const std::vector<double> &knots,
                                   int degree = 3);

   /// \brief Evaluate the reluctivity in the element described by trans at the
   /// point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state) override;

   /// \brief Evaluate the derivative of reluctivity with respsect to magnetic
   /// flux in the element described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double EvalStateDeriv(mfem::ElementTransformation &trans,
                         const mfem::IntegrationPoint &ip,
                         double state) override;

   void EvalRevDiff(const double Q_bar,
                    mfem::ElementTransformation &trans,
                    const mfem::IntegrationPoint &ip,
                    double state,
                    mfem::DenseMatrix &PointMat_bar) override
   { }

   // ~BHBSplineReluctivityCoefficient() override;

protected:
   /// max H value in the data
   double h_max;
   /// max B value in the data
   double b_max;
   /// spline representing H(B)
   std::unique_ptr<tinyspline::BSpline> bh;
   /// spline representing dH(B)/dB
   std::unique_ptr<tinyspline::BSpline> dbdh;
};

class team13ReluctivityCoefficient : public miso::StateCoefficient
{
public:
   /// \brief Define a reluctivity model for the team13 steel
   team13ReluctivityCoefficient() { std::cout << "using team13 coeff!\n"; }

   /// \brief Evaluate the reluctivity in the element described by trans at the
   /// point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state) override;

   /// \brief Evaluate the derivative of reluctivity with respsect to magnetic
   /// flux in the element described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double EvalStateDeriv(mfem::ElementTransformation &trans,
                         const mfem::IntegrationPoint &ip,
                         double state) override;
};

std::unique_ptr<mfem::Coefficient> constructLinearReluctivityCoeff(
    const std::string &material_name,
    const nlohmann::json &materials)
{
   auto mu_r = materials[material_name].value("mu_r", 1.0);
   return std::make_unique<mfem::ConstantCoefficient>(1.0 / (mu_r * mu_0));
}

void getCpsKnotsAndDegree(const nlohmann::json &material,
                          const nlohmann::json &materials,
                          const std::string &model,
                          std::vector<double> &cps,
                          std::vector<double> &knots,
                          int &degree)
{
   const auto &material_name = material["name"].get<std::string>();

   if (material["reluctivity"].contains("cps"))
   {
      cps = material["reluctivity"]["cps"].get<std::vector<double>>();
   }
   else
   {
      cps = materials[material_name]["reluctivity"][model]["cps"]
                .get<std::vector<double>>();
   }
   if (material["reluctivity"].contains("knots"))
   {
      knots = material["reluctivity"]["knots"].get<std::vector<double>>();
   }
   else
   {
      knots = materials[material_name]["reluctivity"][model]["knots"]
                  .get<std::vector<double>>();
   }
   if (material["reluctivity"].contains("degree"))
   {
      degree = material["reluctivity"]["degree"].get<int>();
   }
   else
   {
      degree =
          materials[material_name]["reluctivity"][model].value("degree", 3);
   }
}

std::unique_ptr<mfem::Coefficient> constructReluctivityCoeff(
    const nlohmann::json &component,
    const nlohmann::json &materials)
{
   std::unique_ptr<mfem::Coefficient> temp_coeff;
   const auto &material = component["material"];

   /// If "material" is a string, it is interpreted to be the name of a
   /// material. We default to a linear reluctivity with mu_r = 1.0 unless
   /// there is a different value in the material library
   if (material.is_string())
   {
      const auto &material_name = material.get<std::string>();
      temp_coeff = constructLinearReluctivityCoeff(material_name, materials);
   }
   else
   {
      const auto &material_name = material["name"].get<std::string>();

      if (material.contains("reluctivity"))
      {
         const auto &nu_model =
             material["reluctivity"]["model"].get<std::string>();
         if (nu_model == "linear")
         {
            temp_coeff =
                constructLinearReluctivityCoeff(material_name, materials);
         }
         else if (nu_model == "lognu")
         {
            std::vector<double> cps;
            std::vector<double> knots;
            int degree = 0;
            getCpsKnotsAndDegree(
                material, materials, nu_model, cps, knots, degree);
            temp_coeff = std::make_unique<logNuBBSplineReluctivityCoefficient>(
                cps, knots, degree);
         }
         else if (nu_model == "bh")
         {
            std::vector<double> cps;
            std::vector<double> knots;
            int degree = 0;
            getCpsKnotsAndDegree(
                material, materials, nu_model, cps, knots, degree);
            temp_coeff = std::make_unique<BHBSplineReluctivityCoefficient>(
                cps, knots, degree);
         }
         else if (nu_model == "team13")
         {
            throw miso::MISOException("");
         }
         else
         {
            std::string error_msg =
                "Unrecognized reluctivity model for material \"";
            error_msg += material_name;
            error_msg += "\"!\n";
            throw miso::MISOException(error_msg);
         }
      }
      else
      {
         temp_coeff = constructLinearReluctivityCoeff(material_name, materials);
      }
   }

   // auto has_nonlinear =
   //     materials[material].contains("B") &&
   //     materials[material].contains("H");
   // if (component.contains("linear"))
   // {
   //    auto linear = component["linear"].get<bool>();
   //    if (linear)
   //    {
   //       auto mu_r = materials[material]["mu_r"].get<double>();
   //       temp_coeff =
   //           std::make_unique<mfem::ConstantCoefficient>(1.0 / (mu_r *
   //           mu_0));
   //    }
   //    else
   //    {
   //       auto b = materials[material]["B"].get<std::vector<double>>();
   //       auto h = materials[material]["H"].get<std::vector<double>>();
   //       temp_coeff = std::make_unique<BHBSplineReluctivityCoefficient>(b,
   //       h);
   //    }
   // }
   // else
   // {
   // if (has_nonlinear)
   // {
   //    auto b = materials[material]["B"].get<std::vector<double>>();
   //    auto h = materials[material]["H"].get<std::vector<double>>();
   //    temp_coeff = std::make_unique<BHBSplineReluctivityCoefficient>(b, h);
   // }
   // else
   // {
   //    auto mu_r = materials[material]["mu_r"].get<double>();
   //    temp_coeff =
   //        std::make_unique<mfem::ConstantCoefficient>(1.0 / (mu_r * mu_0));
   // }
   // }
   return temp_coeff;
}

}  // anonymous namespace

namespace miso
{
double ReluctivityCoefficient::Eval(mfem::ElementTransformation &trans,
                                    const mfem::IntegrationPoint &ip)
{
   return nu.Eval(trans, ip);
}

double ReluctivityCoefficient::Eval(mfem::ElementTransformation &trans,
                                    const mfem::IntegrationPoint &ip,
                                    double state)
{
   return nu.Eval(trans, ip, state);
}

double ReluctivityCoefficient::EvalStateDeriv(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state)
{
   return nu.EvalStateDeriv(trans, ip, state);
}

double ReluctivityCoefficient::EvalState2ndDeriv(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    const double state)
{
   return nu.EvalState2ndDeriv(trans, ip, state);
}

void ReluctivityCoefficient::EvalRevDiff(const double Q_bar,
                                         mfem::ElementTransformation &trans,
                                         const mfem::IntegrationPoint &ip,
                                         double state,
                                         mfem::DenseMatrix &PointMat_bar)
{
   nu.EvalRevDiff(Q_bar, trans, ip, state, PointMat_bar);
}

ReluctivityCoefficient::ReluctivityCoefficient(const nlohmann::json &nu_options,
                                               const nlohmann::json &materials)
 : nu(std::make_unique<mfem::ConstantCoefficient>(1.0 / mu_0))
{
   /// loop over all components, construct a reluctivity coefficient for each
   for (const auto &component : nu_options["components"])
   {
      int attr = component.value("attr", -1);
      if (-1 != attr)
      {
         nu.addCoefficient(attr,
                           constructReluctivityCoeff(component, materials));
      }
      else
      {
         for (const auto &attribute : component["attrs"])
         {
            nu.addCoefficient(attribute,
                              constructReluctivityCoeff(component, materials));
         }
      }
   }
}

}  // namespace miso

/// Move these coefficients to their own header file so I can test them
/// Make sure they predict similar nu and dnudb values

namespace
{
logNuBBSplineReluctivityCoefficient::logNuBBSplineReluctivityCoefficient(
    const std::vector<double> &cps,
    const std::vector<double> &knots,
    int degree)
 : lognu_max(cps[cps.size() - 1]),
   b_max(knots[knots.size() - 1]),
   lognu(std::make_unique<tinyspline::BSpline>(cps.size(), 1, degree))
{
   lognu->setControlPoints(cps);
   lognu->setKnots(knots);

   dlognudb = std::make_unique<tinyspline::BSpline>(lognu->derive());
}

double logNuBBSplineReluctivityCoefficient::Eval(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    const double state)
{
   if (state <= b_max)
   {
      double nu = exp(lognu->eval(state).result()[0]);
      return nu;
   }
   else
   {
      return nu0;
   }
}

double logNuBBSplineReluctivityCoefficient::EvalStateDeriv(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    const double state)
{
   if (state > b_max)
   {
      std::cout << "lognu state: " << state;
      std::cout << " !!!LARGE STATE!!!";
      std::cout << "\n";
   }

   if (state <= b_max)
   {
      double nu = exp(lognu->eval(state).result()[0]);
      double dnudb = nu * dlognudb->eval(state).result()[0];
      return dnudb;
   }
   else
   {
      return 0.0;
   }
}

double logNuBBSplineReluctivityCoefficient::EvalState2ndDeriv(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    const double state)
{
   if (d2lognudb2 == nullptr)
   {
      d2lognudb2 = std::make_unique<tinyspline::BSpline>(dlognudb->derive());
   }
   if (state > b_max)
   {
      std::cout << "lognu state: " << state;
      std::cout << " !!!LARGE STATE!!!";
      std::cout << "\n";
   }

   if (state <= b_max)
   {
      double lognu_val = lognu->eval(state).result()[0];
      double nu = exp(lognu_val);

      double dlognudb_val = dlognudb->eval(state).result()[0];
      // double dnudb = nu * dlognudb_val;

      double d2lognudb2_val = d2lognudb2->eval(state).result()[0];
      double d2nudb2 = nu * (d2lognudb2_val + pow(dlognudb_val, 2));

      return d2nudb2;
   }
   else
   {
      return 0.0;
   }
}

// double logNuBBSplineReluctivityCoefficient::EvalState2ndDeriv(
//     mfem::ElementTransformation &trans,
//     const mfem::IntegrationPoint &ip,
//     const double state)
// {
//    if (state <= b_max)
//    {
//       double t = state / b_max;
//       double nu = exp(lognu->eval(t).result()[0]);
//       double first_deriv = dlognudb->eval(t).result()[0];
//       double second_deriv = d2lognudb2->eval(t).result()[0];
//       double d2nudb2 = nu * (second_deriv + pow(first_deriv, 2));
//       return d2nudb2;
//    }
//    else
//    {
//       return 0.0;
//    }
// }

BHBSplineReluctivityCoefficient::BHBSplineReluctivityCoefficient(
    const std::vector<double> &cps,
    const std::vector<double> &knots,
    int degree)
 // : b_max(B[B.size()-1]), nu(H.size(), 1, 3)
 : h_max(cps[cps.size() - 1]),
   b_max(knots[knots.size() - 1]),
   bh(std::make_unique<tinyspline::BSpline>(cps.size(), 1, degree))
{
   std::vector<double> scaled_knots(knots);
   for (int i = 0; i < knots.size(); ++i)
   {
      scaled_knots[i] = scaled_knots[i] / b_max;
   }
   bh->setControlPoints(cps);
   bh->setKnots(scaled_knots);

   dbdh = std::make_unique<tinyspline::BSpline>(bh->derive());
   // dnudb = nu.derive();
}

double BHBSplineReluctivityCoefficient::Eval(mfem::ElementTransformation &trans,
                                             const mfem::IntegrationPoint &ip,
                                             const double state)
{
   constexpr double nu0 = 1 / (4e-7 * M_PI);
   // std::cout << "eval state state: " << state << "\n";
   if (state <= 1e-14)
   {
      double t = state / b_max;
      double nu = dbdh->eval(t).result()[0] / b_max;
      return nu;
   }
   else if (state <= b_max)
   {
      double t = state / b_max;
      double nu = bh->eval(t).result()[0] / state;
      // std::cout << "eval state nu: " << nu << "\n";
      return nu;
   }
   else
   {
      return (h_max - nu0 * b_max) / state + nu0;
   }
}

double BHBSplineReluctivityCoefficient::EvalStateDeriv(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    const double state)
{
   constexpr double nu0 = 1 / (4e-7 * M_PI);

   /// TODO: handle state == 0
   if (state <= b_max)
   {
      double t = state / b_max;
      double h = bh->eval(t).result()[0];
      return dbdh->eval(t).result()[0] / (state * b_max) - h / pow(state, 2);
   }
   else
   {
      return -(h_max - nu0 * b_max) / pow(state, 2);
   }
}

// BHBSplineReluctivityCoefficient::~BHBSplineReluctivityCoefficient() =
// default;

/** unused
double team13h(double b_hat)
{
   const double h =
       exp((0.0011872363994136887 * pow(b_hat, 2) * (15 * pow(b_hat, 2) - 9.0) -
            0.19379133411847338 * pow(b_hat, 2) -
            0.012675319795245974 * b_hat * (3 * pow(b_hat, 2) - 1.0) +
            0.52650810858405916 * b_hat + 0.77170389255937188) /
           (-0.037860246476916264 * pow(b_hat, 2) +
            0.085040155318288846 * b_hat + 0.1475250808150366)) -
       31;
   return h;
}
*/

double team13dhdb_hat(double b_hat)
{
   const double dhdb_hat =
       (-0.0013484718812450662 * pow(b_hat, 5) +
        0.0059829967461202211 * pow(b_hat, 4) +
        0.0040413617616232578 * pow(b_hat, 3) -
        0.013804440762666015 * pow(b_hat, 2) - 0.0018970139190370716 * b_hat +
        0.013917259962808418) *
       exp((0.017808545991205332 * pow(b_hat, 4) -
            0.038025959385737926 * pow(b_hat, 3) -
            0.20447646171319658 * pow(b_hat, 2) + 0.53918342837930511 * b_hat +
            0.77170389255937188) /
           (-0.037860246476916264 * pow(b_hat, 2) +
            0.085040155318288846 * b_hat + 0.1475250808150366)) /
       (0.0014333982632928504 * pow(b_hat, 4) -
        0.0064392824815713142 * pow(b_hat, 3) -
        0.0039388438258098624 * pow(b_hat, 2) + 0.025091111571707653 * b_hat +
        0.02176364946948308);
   return dhdb_hat;
}

double team13d2hdb_hat2(double b_hat)
{
   const double d2hdb_hat2 =
       (1.8183764145086082e-6 * pow(b_hat, 10) -
        1.6135805755447689e-5 * pow(b_hat, 9) +
        2.2964027416433258e-5 * pow(b_hat, 8) +
        0.00010295509167249583 * pow(b_hat, 7) -
        0.0001721199302193437 * pow(b_hat, 6) -
        0.00031470749218644612 * pow(b_hat, 5) +
        0.00054873370082066282 * pow(b_hat, 4) +
        0.00078428896855240252 * pow(b_hat, 3) -
        0.00020176627749697931 * pow(b_hat, 2) -
        0.00054403666453702558 * b_hat - 0.00019679534359955033) *
       exp((0.017808545991205332 * pow(b_hat, 4) -
            0.038025959385737926 * pow(b_hat, 3) -
            0.20447646171319658 * pow(b_hat, 2) + 0.53918342837930511 * b_hat +
            0.77170389255937188) /
           (-0.037860246476916264 * pow(b_hat, 2) +
            0.085040155318288846 * b_hat + 0.1475250808150366)) /
       (2.0546305812109595e-6 * pow(b_hat, 8) -
        1.8460112651872795e-5 * pow(b_hat, 7) +
        3.0172495078875982e-5 * pow(b_hat, 6) +
        0.00012265776759231136 * pow(b_hat, 5) -
        0.0002452310649846335 * pow(b_hat, 4) -
        0.00047794451332165656 * pow(b_hat, 3) +
        0.00045811664722395466 * pow(b_hat, 2) + 0.001092148314092672 * b_hat +
        0.00047365643823053111);
   return d2hdb_hat2;
}

double team13b_hat(double b)
{
   const double b_hat = 1.10803324099723 * b + 1.10803324099723 * atan(20 * b) -
                        0.9944598337950139;
   return b_hat;
}

double team13db_hatdb(double b)
{
   const double db_hatdb = (443.213296398892 * pow(b, 2) + 23.26869806094183) /
                           (400 * pow(b, 2) + 1);
   return db_hatdb;
}

double team13d2b_hatdb2(double b)
{
   const double d2b_hatdb2 =
       -17728.53185595568 * b / pow(400 * pow(b, 2) + 1, 2);
   return d2b_hatdb2;
}

double team13ReluctivityCoefficient::Eval(mfem::ElementTransformation &trans,
                                          const mfem::IntegrationPoint &ip,
                                          const double state)
{
   if (state > 2.2)
   {
      return 1 / (4 * M_PI * 1e-7);
   }
   const double b_hat = team13b_hat(state);
   const double db_hatdb = team13db_hatdb(state);

   const double dhdb_hat = team13dhdb_hat(b_hat);

   const double nu = dhdb_hat * db_hatdb;
   // std::cout << "state: " << state << " nu: " << nu << "\n";

   // try
   // {
   //    if (!isfinite(nu))
   //    {
   //       throw MISOException("nan!");
   //    }
   // }
   // catch(const std::exception& e)
   // {
   //    std::cerr << e.what() << '\n';
   // }

   return nu;
}

double team13ReluctivityCoefficient::EvalStateDeriv(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    const double state)
{
   if (state > 2.2)
   {
      return 0;
   }
   const double b_hat = team13b_hat(state);
   const double db_hatdb = team13db_hatdb(state);
   const double d2b_hatdb2 = team13d2b_hatdb2(state);

   const double dhdb_hat = team13dhdb_hat(b_hat);
   const double d2hdb_hat2 = team13d2hdb_hat2(b_hat);

   // const double dnudb = d2hdb_hat2 * pow(db_hatdb, 2) + dhdb_hat *
   // d2b_hatdb2; std::cout << "state: " << state << " dnudb: " << dnudb <<
   // "\n";

   // try
   // {
   //    if (!isfinite(dnudb))
   //    {
   //       throw MISOException("nan!");
   //    }
   // }
   // catch(const std::exception& e)
   // {
   //    std::cerr << e.what() << '\n';
   // }

   return d2hdb_hat2 * pow(db_hatdb, 2) + dhdb_hat * d2b_hatdb2;
}

}  // namespace
