import unittest
import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_check_totals

from miso import PDESolver, MISOMesh, MISOFunctional

class TestEMFunctionals(unittest.TestCase):
    coulomb_options = {
        "mesh": {
            "file": "data/coulomb1984.mesh",
            "refine": 0
        },
        "space-dis": {
            "basis-type": "nedelec",
            "degree": 1
        },
        "lin-solver": {
            "type": "minres",
            "printlevel": 1,
            "maxiter": 100,
            "abstol": 1e-14,
            "reltol": 1e-14
        },
        "nonlin-solver": {
            "type": "newton",
            "printlevel": 1,
            "maxiter": 5,
            "reltol": 1e-6,
            "abstol": 1e-6
        },
        "components": {
            "ring": {
                "material": "copperwire",
                "attrs": [1, 2],
            }
        },
        "bcs": {
            "essential": [1, 3]
        },
        "current": {
            "test": {
                "ring": [1, 2]
            }
        }
    }

    box_options = {
        "mesh": {
            "file": "data/simple_box.mesh",
            "refine": 0
        },
        "space-dis": {
            "basis-type": "nedelec",
            "degree": 1
        },
        "lin-solver": {
            "type": "minres",
            "printlevel": 1,
            "maxiter": 100,
            "abstol": 1e-14,
            "reltol": 1e-14
        },
        "nonlin-solver": {
            "type": "newton",
            "printlevel": 1,
            "maxiter": 5,
            "reltol": 1e-6,
            "abstol": 1e-6
        },
        "components": {
            "ring": {
                "attrs": [1],
                "material": {
                    "name": "box1",
                    "mu_r": 795774.7154594767
                },
            }
        },
        "bcs": {
            "essential": "all"
        },
        "current": {
            "test": {
                "z": [1]
            }
        }
    }

    square_options = {
        "mesh": {
            "file": "data/simple_square.mesh",
            "refine": 0
        },
        "space-dis": {
            "basis-type": "h1",
            "degree": 1
        },
        "lin-solver": {
            "type": "pcg",
            "printlevel": 1,
            "maxiter": 100,
            "abstol": 1e-14,
            "reltol": 1e-14
        },
        "nonlin-solver": {
            "type": "newton",
            "printlevel": 1,
            "maxiter": 5,
            "reltol": 1e-6,
            "abstol": 1e-6
        },
        "components": {
            "ring": {
                "attrs": [1],
                "material": {
                    "name": "box1",
                    "mu_r": 795774.7154594767
                },
            }
        },
        "bcs": {
            "essential": "all"
        },
        "current": {
            "test": {
                "z": [1]
            }
        }
    }

    heat_source_options = {
        "mesh": {
            "file": "data/simple_square.mesh",
            "refine": 0
        },
        "space-dis": {
            "basis-type": "h1",
            "degree": 1
        },
        "lin-solver": {
            "type": "pcg",
            "printlevel": 1,
            "maxiter": 100,
            "abstol": 1e-14,
            "reltol": 1e-14
        },
        "nonlin-solver": {
            "type": "newton",
            "printlevel": 1,
            "maxiter": 5,
            "reltol": 1e-6,
            "abstol": 1e-6
        },
        "components": {
            "test": {
               "attrs": [1],
               "material": {
                    "name": "test",
                    "core_loss": {
                        "model": "CAL2",
                        "T0": 293.15,
                        "kh_T0": [5.97783049251564E-02, -6.58569751792524E-02, 3.52052785575931E-02, -6.54762513683037E-03],
                        "ke_T0": [3.83147202762929E-05, -4.19965038193089E-05, 2.09788988466414E-05, -3.88567697029196E-06],
                        "T1": 473.15,
                        "kh_T1": [5.78728253280150E-02, -7.94684973286488E-02, 5.09165213772802E-02, -1.11117379956941E-02],
                        "ke_T1": [3.20525407302126E-05, -1.43502199723297E-05, -3.74786590271071E-06, 2.68517704958978E-06]
                    },
                    "rho": 1.0
                }
            }
        },
        "bcs": {
            "essential": "all"
        },
        "current": {
            "test": {
                "z": [1]
            }
        }
    }

    def test_coulomb_forward(self):
        prob = om.Problem()

        emSolver = PDESolver(type="magnetostatic",
                             solver_options=self.coulomb_options,
                             comm=prob.comm)

        state_size = emSolver.getFieldSize("state")
        state = np.zeros(state_size)
        inputs = {
            "current_density:test": 3e6,
            "state": state
        }
        emSolver.solveForState(inputs, state)

        ivc = prob.model.add_subsystem("ivc",
                                       om.IndepVarComp(),
                                       promotes_outputs=["state"])
        ivc.add_output("state", val=state)

        prob.model.add_subsystem("mesh",
                                 MISOMesh(solver=emSolver),
                                 promotes_outputs=["*"])


        prob.model.add_subsystem("energy",
                                 MISOFunctional(solver=emSolver,
                                                func="energy",
                                                depends=["state", "mesh_coords"]),
                                 promotes_inputs=[("mesh_coords", "x_em0"), "state"],
                                 promotes_outputs=["energy"])

        force_options = {
            "attributes": [1],
            "axis": [0, 0, 1]
        }
        prob.model.add_subsystem("force",
                                 MISOFunctional(solver=emSolver,
                                                func="force",
                                                func_options=force_options,
                                                depends=["state", "mesh_coords"]),
                                 promotes_inputs=[("mesh_coords", "x_em0"), "state"],
                                 promotes_outputs=["force"])

        torque_options = {
            "attributes": [1],
            "axis": [0, 0, 1],
            "about": [0.0, 0.0, 0.0]
        }
        prob.model.add_subsystem("torque",
                                 MISOFunctional(solver=emSolver,
                                                func="torque",
                                                func_options=torque_options,
                                                depends=["state", "mesh_coords"]),
                                 promotes_inputs=[("mesh_coords", "x_em0"), "state"],
                                 promotes_outputs=["torque"])

        prob.setup()
        prob.run_model()

        energy = prob.get_val("energy")[0]
        self.assertAlmostEqual(energy, 0.0142746123)
        force = prob.get_val("force")[0]
        self.assertAlmostEqual(force, -0.0791988853)
        torque = prob.get_val("torque")[0]
        self.assertAlmostEqual(torque, 0.0000104977)

    def test_energy_partials_coulomb(self):
        prob = om.Problem()

        emSolver = PDESolver(type="magnetostatic",
                             solver_options=self.coulomb_options,
                             comm=prob.comm)

        state_size = emSolver.getFieldSize("state")
        state = np.zeros(state_size)
        inputs = {
            "current_density:test": 3e6,
            "state": state
        }
        emSolver.solveForState(inputs, state)

        ivc = prob.model.add_subsystem("ivc",
                                       om.IndepVarComp(),
                                       promotes_outputs=["state"])
        ivc.add_output("state", val=state)

        prob.model.add_subsystem("mesh",
                                 MISOMesh(solver=emSolver),
                                 promotes_outputs=["*"])


        energy = prob.model.add_subsystem("energy",
                                          MISOFunctional(solver=emSolver,
                                                         func="energy",
                                                         depends=["state", "mesh_coords"]),
                                          promotes_inputs=[("mesh_coords", "x_em0"), "state"],
                                          promotes_outputs=["energy"])
        energy.set_check_partial_options(wrt="*", directional=True)

        force_options = {
            "attributes": [1],
            "axis": [0, 0, 1]
        }
        force = prob.model.add_subsystem("force",
                                         MISOFunctional(solver=emSolver,
                                                        func="force",
                                                        func_options=force_options,
                                                        depends=["state", "mesh_coords"]),
                                         promotes_inputs=[("mesh_coords", "x_em0"), "state"],
                                         promotes_outputs=["force"])
        force.set_check_partial_options(wrt="*", directional=True)

        torque_options = {
            "attributes": [1],
            "axis": [0, 0, 1],
            "about": [0.0, 0.0, 0.0]
        }
        torque = prob.model.add_subsystem("torque",
                                          MISOFunctional(solver=emSolver,
                                                         func="torque",
                                                         func_options=torque_options,
                                                         depends=["state", "mesh_coords"]),
                                          promotes_inputs=[("mesh_coords", "x_em0"), "state"],
                                          promotes_outputs=["torque"])
        torque.set_check_partial_options(wrt="*", directional=True)

        prob.setup()
        prob.run_model()

        data = prob.check_partials(form="central")
        assert_check_partials(data)

    def test_flux_magnitude(self):
        prob = om.Problem()

        emSolver = PDESolver(type="magnetostatic",
                             solver_options=self.box_options,
                             comm=prob.comm)

        state_size = emSolver.getFieldSize("state")
        state = np.random.randn(state_size)

        ivc = prob.model.add_subsystem("ivc",
                                       om.IndepVarComp(),
                                       promotes_outputs=["state"])
        ivc.add_output("state", val=state)

        prob.model.add_subsystem("mesh",
                                 MISOMesh(solver=emSolver),
                                 promotes_outputs=["*"])

        flux_mag = prob.model.add_subsystem("flux_magnitude",
                                            MISOFunctional(solver=emSolver,
                                                           func="flux_magnitude",
                                                           check_partials=True,
                                                           depends=["state", "mesh_coords"]),
                                            promotes_inputs=[("mesh_coords", "x_em0"), "state"],
                                            promotes_outputs=["flux_magnitude"])
        flux_mag.set_check_partial_options(wrt="*", directional=True)

        prob.setup()
        prob.run_model()

        data = prob.check_partials(form="central")
        assert_check_partials(data)

    def test_flux_density(self):
        prob = om.Problem()

        emSolver = PDESolver(type="magnetostatic",
                             solver_options=self.square_options,
                             comm=prob.comm)

        state_size = emSolver.getFieldSize("state")
        state = np.random.randn(state_size)

        def field_func(x):
            return x[0]**2 + x[1]**2
        emSolver.setState(field_func, state)

        print(f"state: {state}")

        ivc = prob.model.add_subsystem("ivc",
                                       om.IndepVarComp(),
                                       promotes_outputs=["state"])
        ivc.add_output("state", val=state)

        prob.model.add_subsystem("mesh",
                                 MISOMesh(solver=emSolver),
                                 promotes_outputs=["*"])

        flux = prob.model.add_subsystem("flux_density",
                                        MISOFunctional(solver=emSolver,
                                                       func="flux_density",
                                                       check_partials=True,
                                                       depends=["state", "mesh_coords"]),
                                        promotes_inputs=[("mesh_coords", "x_em0"), "state"],
                                        promotes_outputs=["flux_density"])
        flux.set_check_partial_options(wrt="*", directional=True)

        flux_mag = prob.model.add_subsystem("flux_magnitude",
                                            MISOFunctional(solver=emSolver,
                                                           func="flux_magnitude",
                                                           check_partials=True,
                                                           depends=["state", "mesh_coords"]),
                                            promotes_inputs=[("mesh_coords", "x_em0"), "state"],
                                            promotes_outputs=["flux_magnitude"])
        flux_mag.set_check_partial_options(wrt="*", directional=True)

        avg_flux = prob.model.add_subsystem("average_flux_magnitude",
                                            MISOFunctional(solver=emSolver,
                                                           func="average_flux_magnitude",
                                                           check_partials=True,
                                                           depends=["state", "mesh_coords"]),
                                            promotes_inputs=[("mesh_coords", "x_em0"), "state"],
                                            promotes_outputs=["average_flux_magnitude"])
        avg_flux.set_check_partial_options(wrt="*", directional=True)

        max_flux = prob.model.add_subsystem("max_flux_magnitude",
                                            MISOFunctional(solver=emSolver,
                                                           func="max_flux_magnitude",
                                                           func_options={"rho": 1},
                                                           check_partials=True,
                                                           depends=["state", "mesh_coords"]),
                                            promotes_inputs=[("mesh_coords", "x_em0"), "state"],
                                            promotes_outputs=["max_flux_magnitude"])
        max_flux.set_check_partial_options(wrt="*", directional=True)

        prob.setup()
        prob.run_model()

        data = prob.check_partials(form="central")
        assert_check_partials(data)
        # print(data)

        # flux_mag_data = data.pop("flux_magnitude")
        # assert_check_partials({"flux_magnitude": flux_mag_data})
        # avg_flux_data = data.pop("average_flux_magnitude")
        # assert_check_partials({"average_flux_magnitude": avg_flux_data})
        # max_flux_data = data.pop("max_flux_magnitude")
        # assert_check_partials({"max_flux_magnitude": max_flux_data})

    def test_heat_source(self):
        prob = om.Problem()

        emSolver = PDESolver(type="magnetostatic",
                             solver_options=self.heat_source_options,
                             comm=prob.comm)

        state_size = emSolver.getFieldSize("state")
        state = np.random.randn(state_size)

        def field_func(x):
            return x[0]**2 + x[1]**2
        emSolver.setState(field_func, state)

        print(f"state: {state}")

        ivc = prob.model.add_subsystem("ivc",
                                       om.IndepVarComp(),
                                       promotes_outputs=["state"])
        ivc.add_output("state", val=state)

        prob.model.add_subsystem("mesh",
                                 MISOMesh(solver=emSolver),
                                 promotes_outputs=["*"])

        heat_source_inputs = [("mesh_coords", "x_em0"),
                              "temperature",
                              "frequency",
                              "wire_length",
                              "rms_current",
                              "strand_radius",
                              "strands_in_hand",
                              "stack_length",
                              "peak_flux",
                              "model_depth",
                              "num_turns",
                              "num_slots"]

        prob.model.add_subsystem("heat_source",
                                 MISOFunctional(solver=emSolver,
                                              func="heat_source",
                                              func_options={
                                                  "dc_loss": {
                                                      "attributes": [1]
                                                  },
                                                  "ac_loss": {
                                                      "attributes": [1]
                                                  },
                                                  "core_loss": {
                                                      "attributes": [1]
                                                  },
                                              },
                                              depends=heat_source_inputs,
                                              check_partials=True),
                               promotes_inputs=heat_source_inputs,
                               promotes_outputs=[("heat_source", "thermal_load")])

        prob.setup()

        temp_size = emSolver.getFieldSize("temperature")
        prob["temperature"][:] = 100 + np.random.randn(temp_size)

        flux_size = emSolver.getFieldSize("peak_flux")
        prob["peak_flux"][:] = np.random.randn(flux_size)**2

        prob["rms_current"] = 10

        prob.run_model()

        print(f"heat_source: {prob['thermal_load']}")

        data = prob.check_partials(form="central")
        assert_check_partials(data)

class TestThermalFunctionals(unittest.TestCase):
    square_options = {
        "mesh": {
            "file": "data/simple_square.mesh",
            "refine": 0
        },
        "space-dis": {
            "basis-type": "h1",
            "degree": 1
        },
        "lin-solver": {
            "type": "pcg",
            "printlevel": 1,
            "maxiter": 100,
            "abstol": 1e-14,
            "reltol": 1e-14
        },
        "nonlin-solver": {
            "type": "newton",
            "printlevel": 1,
            "maxiter": 5,
            "reltol": 1e-12,
            "abstol": 1e-12
        },
        "components": {
            "test": {
                "attrs": [1],
                "material": {
                    "name": "test",
                    "kappa": 1.0,
                    "conductivity": {
                        "model": "linear",
                        "sigma_T_ref": 58.14e6,
                        "T_ref": 293.15,
                        "alpha_resistivity": 3.8e-3
                    }
                },
            }
        },
        "bcs": {
            "convection": [1, 2],
            "outflux": [3, 4]
        }
    }

    def test_max_temp(self):
        prob = om.Problem()

        thermal_solver = PDESolver(type="thermal",
                                   solver_options=self.square_options,
                                   comm=prob.comm)

        state_size = thermal_solver.getFieldSize("state")
        state = np.random.randn(state_size)

        def field_func(x):
            return x[0]**2 + x[1]**2
        thermal_solver.setState(field_func, state)

        ivc = prob.model.add_subsystem("ivc",
                                       om.IndepVarComp(),
                                       promotes_outputs=["state"])
        ivc.add_output("state", val=state)

        prob.model.add_subsystem("mesh",
                                 MISOMesh(solver=thermal_solver),
                                 promotes_outputs=["*"])

        max_temp = prob.model.add_subsystem("max_state",
                                            MISOFunctional(solver=thermal_solver,
                                                           func="max_state",
                                                           func_options={"rho": 10.0},
                                                           check_partials=True,
                                                           depends=["state", "mesh_coords"]),
                                            promotes_inputs=[("mesh_coords", "x_conduct0"), "state"],
                                            promotes_outputs=["max_state"])
        max_temp.set_check_partial_options(wrt="*", directional=True)

        prob.setup()
        prob.run_model()

        data = prob.check_partials(form="central")
        assert_check_partials(data)

if __name__ == "__main__":
    unittest.main()