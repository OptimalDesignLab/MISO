import unittest

from mach.pyMach import MachSolver, Mesh, Vector

class SolverRegressionTests(unittest.TestCase):
    def test_steady_vortex(self):

        # Provide the options explicitly for regression tests
        options = {
            "mesh": {
                "file": "airfoil_steady.mesh",
                "refine": 0
            },
            "print-options": False,
            "flow-param": {
                "mach": 1.0,
                "aoa": 0.0
            },
            "space-dis": {
                "degree": 1,
                "lps-coeff": 1.0,
                "basis-type": "csbp"
            },
            "time-dis": {
                "steady": True,
                "steady-abstol": 1e-12,
                "steady-restol": 1e-10,
                "ode-solver": "PTC",
                "t-final": 100,
                "dt": 1e12,
                "cfl": 1.0,
                "res-exp": 2.0
            },
            "bcs": {
                "vortex": [1, 1, 1, 0],
                "slip-wall": [0, 0, 0, 1]
            },
            "newton": {
                "printlevel": 0,
                "maxiter": 50,
                "reltol": 1e-1,
                "abstol": 1e-12
            },
            "petscsolver": {
                "ksptype": "gmres",
                "pctype": "lu",
                "abstol": 1e-15,
                "reltol": 1e-15,
                "maxiter": 100,
                "printlevel": 0
            },
            "lin-solver": {
                "printlevel": 0,
                "filllevel": 3,
                "maxiter": 100,
                "reltol": 1e-2,
                "abstol": 1e-12
            },
            "saveresults": False,
            "outputs":
            { 
                "drag": [0, 0, 0, 1]
            }
        }
        solver = MachSolver("Euler", options, entvar = False)

if __name__ == '__main__':
    unittest.main()
