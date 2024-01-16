#ifndef MACH_OPTIMIZATION
#define MACH_OPTIMIZATION

#include <fstream>
#include <iostream>
#include <iomanip>

#include "adept.h"
#include "mfem.hpp"
#include "galer_diff.hpp"
#include "centgridfunc.hpp"

#include "utils.hpp"
#include "json.hpp"
#include "mach_types.hpp"
namespace mach
{

   class OptProblem : public mfem::Operator
   {
   public:
      /// class constructor
      /// \param[in] opt_file_name - option file
      /// \param[in] initail - initial condition
      /// \param[in] smesh - mesh file
      OptProblem(mfem::Vector init,
                 const std::string &opt_file_name = std::string("mach_options.json"),
                 std::unique_ptr<mfem::Mesh> smesh = nullptr);

      ~OptProblem() = default;

      /// @brief Initialize the solver
      virtual void InitializeSolver()
      {
      }
      virtual void InitializeSolver(mfem::VectorFunctionCoefficient &velocity,
                                    mfem::FunctionCoefficient &inflow)
      {
      }

      // compute the full residual norm
      virtual double GetEnergy(const mfem::Vector &x) = 0;

      /// @brief  Compute the jacobian of the functional w.r.t to the design variable
      /// @param x - State of design variable of the jacobian
      /// @param y - Jacobian of the design variable
      virtual void Mult(const mfem::Vector &x, mfem::Vector &y) = 0;

   protected:
      nlohmann::json options;
      int dim;
      int num_state;
      int ROMSize;
      int numBasis;
      int FullSize;
      int numDesignVar;
      mfem::Vector designVar;

      std::unique_ptr<mfem::Mesh> mesh;
      std::unique_ptr<mfem::FiniteElementCollection> fec;
      std::unique_ptr<mfem::DGDSpace> fes_dgd;
      std::unique_ptr<mfem::FiniteElementSpace> fes_full;

      // some working variables
      std::unique_ptr<mfem::CentGridFunction> u_dgd;
      std::unique_ptr<mfem::GridFunction> u_full;

      // nonlinear and linear solver
      std::unique_ptr<mfem::NewtonSolver> newton_solver;
      std::unique_ptr<mfem::UMFPackSolver> solver;
   };

   class LinearProblem : public OptProblem
   {
   public:
      /// class constructor
      /// \param[in] opt_file_name - option file
      /// \param[in] initail - initial condition
      /// \param[in] smesh - mesh file
      LinearProblem(mfem::Vector init,
                    const std::string &opt_file_name = std::string("mach_options.json"),
                    std::unique_ptr<mfem::Mesh> smesh = nullptr);

      void InitializeSolver(mfem::VectorFunctionCoefficient &velocity,
                            mfem::FunctionCoefficient &inflow) override;

      void SetInitialCondition(void (*u_init)(const mfem::Vector &,
                                              mfem::Vector &));

      void SetInitialCondition(const mfem::Vector uic);

      double GetEnergy(const mfem::Vector &x);

      /// compute the jacobian of the functional w.r.t the design variable
      void Mult(const mfem::Vector &x, mfem::Vector &y) const override;

      void printSolution(const mfem::Vector &c, const std::string &file_name);
      /// class destructor
      ~LinearOptimizer()
      {
      }

   protected:
      // aux variables
      mfem::Array<int> outflux_bdr;
      mfem::Array<int> influx_bdr;

      // the constraints
      std::unique_ptr<mfem::BilinearForm> res_dgd = nullptr;
      std::unique_ptr<mfem::BilinearForm> res_full = nullptr;

      // rhs operator
      mfem::Vector b_dgd;
      std::unique_ptr<mfem::LinearForm> b_full = nullptr;

      // save operators for convenient
      mfem::SparseMatrix *k_full = nullptr;
      mfem::SparseMatrix *k_dgd = nullptr;
   };

   class EulerProblem : public OptProblem
   {
   public:
      /// class constructor
      /// \param[in] opt_file_name - option file
      /// \param[in] initail - initial condition
      /// \param[in] smesh - mesh file
      EulerProblem(mfem::Vector init,
                   const std::string &opt_file_name = std::string("mach_options.json"),
                   std::unique_ptr<mfem::Mesh> smesh = nullptr);

      void InitializeSolver() override;

      void SetInitialCondition(void (*u_init)(const mfem::Vector &,
                                              mfem::Vector &));

      void SetInitialCondition(const mfem::Vector uic);

      double GetEnergy(const mfem::Vector &x) override;

      /// compute the jacobian of the functional w.r.t the design variable
      void Mult(const mfem::Vector &x, mfem::Vector &y) override;

      void addVolumeIntegrators(double alpha);
      void addBoundaryIntegrators(double alpha);
      void addInterfaceIntegrators(double alpha);

      void checkJacobian(mfem::Vector &x);
      void getFreeStreamState(mfem::Vector &q_ref);

      void printSolution(const mfem::Vector &c, const std::string &file_name);
      /// class destructor
      ~EulerProblem();

   protected:
      static adept::Stack diff_stack;
      bool entvar;

      // airfoil variables
      double mach_fs;
      double aoa_fs;
      int iroll;
      int ipitch;

      // aux variables
      std::vector<mfem::Array<int>> bndry_marker;
      std::vector<mfem::Array<int>> output_bndry_marker;

      // the constraints
      std::unique_ptr<NonlinearFormType> res_dgd;
      std::unique_ptr<NonlinearFormType> res_full;
   };

} // end of namesapce
#endif