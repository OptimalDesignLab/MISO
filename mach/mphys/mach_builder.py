import copy
import numpy as np
import openmdao.api as om
from mphys import Builder

from mach.pyMach import PDESolver, MeshWarper
from mach.mach_state import MachState
from mach.mach_output import MachFunctional
from mach.mesh_warper import MachMeshWarper

def _getPhysicsAbbreviation(solver_options):
    type = solver_options["solver-type"]["type"]

    aero_physics = set(["euler", "navierstokes"])
    if type in aero_physics:
        return "aero"
    elif type == "magnetostatic":
        return "em"
    elif type == "thermal":
        return "conduct"
    else:
        raise RuntimeError("Bad physics given to MachSolver!")

class MachCouplingGroup(om.Group):
    def initialize(self):
        self.options.declare("solver", type=PDESolver, recordable=False)
        self.options.declare("depends", types=list)
        self.options.declare("check_partials", default=False)
        self.options.declare("scenario_name", default=None)

    def setup(self):
        self.solver = self.options["solver"]
        self.depends = self.options["depends"]
        self.check_partials = self.options["check_partials"]

        # Promote variables with physics-specific tag that MPhys expects
        solver_options = self.solver.getOptions()
        mesh_input = "x_" + _getPhysicsAbbreviation(solver_options) + "_vol"
        state_output = _getPhysicsAbbreviation(solver_options) + "_state"
        self.add_subsystem("solver",
                           MachState(solver=self.solver,
                                     depends=self.depends,
                                     check_partials=self.check_partials),
                           promotes_inputs=[*self.depends, ("mesh_coords", mesh_input)],
                           promotes_outputs=[("state", state_output)])

class MachPrecouplingGroup:
    """
    Group that handles surface -> volume mesh movement

    To properly support parallel analysis, I'll need to have this component
    partition the input surface coords
    """
    def initialize(self):
        self.options.declare("solver", type=PDESolver, recordable=False)
        self.options.declare("warper", type=MeshWarper, recordable=False)
        self.options.declare("scenario_name", default=None)

    def setup(self):
        self.solver = self.options["solver"]
        self.warper = self.options["warper"]

        # Promote variables with physics-specific tag that MPhys expects
        solver_options = self.solver.getOptions()
        mesh_input = "x_" + _getPhysicsAbbreviation(solver_options)
        mesh_output = "x_" + _getPhysicsAbbreviation(solver_options) + "_vol"
        self.add_subsystem("solver",
                           MachMeshWarper(warper=self.warper),
                           promotes_inputs=[("surf_mesh_coords", mesh_input)],
                           promotes_outputs=[("vol_mesh_coords", mesh_output)])

class MachOutputsGroup:
    """
    Group that handles calculating outputs after the state solve
    """
    def initialize(self):
        self.options.declare("solver", type=PDESolver, recordable=False)
        self.options.declare("outputs", type=dict, default=None)
        self.options.declare("scenario_name", default=None)

    def setup(self):
        self.solver = self.options["solver"]
        self.outputs = self.options["outputs"]
        self.check_partials = self.options["check_partials"]

        # Promote variables with physics-specific tag that MPhys expects
        solver_options = self.solver.getOptions()
        mesh_input = "x_" + _getPhysicsAbbreviation(solver_options) + "_vol"
        state_input = _getPhysicsAbbreviation(solver_options) + "_state"
        promoted_inputs = [("mesh_coords", mesh_input), ("state", state_input)]

        for output in self.outputs:
            if "options" in self.outputs[output]:
                output_opts = self.outputs[output]["options"]
            else:
                output_opts = None

            if "depends" in self.outputs[output]:
                depends = self.outputs[output]["depends"]
            else:
                depends = []

            self.add_subsystem(output,
                               MachFunctional(solver=self.solver,
                                             func=output,
                                             func_options=output_opts,
                                             depends=depends),
                               promotes_inputs=[*depends, *promoted_inputs],
                               promotes_outputs=[output])


class MachMesh(om.IndepVarComp):
    """
    Component to read the initial surface mesh coordinates
    """

    def initialize(self):
        self.options.declare('solver', default=None, desc='the mach solver object itself', recordable=False)
        self.options.declare('warper', default=None, desc='the mesh warper object itself', recordable=False)

    def setup(self):
        solver = self.options['solver']
        warper = self.options['warper']

        if isinstance(warper, MachMeshWarper):
            local_surf_mesh_size = warper.getSurfaceCoordsSize()
            surf_coords = np.empty(local_surf_mesh_size)
            warper.getInitialSurfaceCoords(surf_coords)
        else:
            raise NotImplementedError("MachMesh class not implemented for mesh warpers besides MachMeshWarper!\n")

        solver_options = solver.getOptions()
        mesh_output = "x_" + _getPhysicsAbbreviation(solver_options) + "0"

        self.add_output(mesh_output,
                        distributed=True,
                        val=surf_coords,
                        desc='surface mesh node coordinates',
                        tags=['mphys_coordinates'])

class MachMeshGroup(om.Group):
    def initialize(self):
        self.options.declare('solver', default=None, desc='the mach solver object itself', recordable=False)
        self.options.declare('warper', default=None, desc='the mesh warper object itself', recordable=False)

    def setup(self):
        solver = self.options["solver"]
        warper = self.options['warper']

        solver_options = solver.getOptions()
        mesh_output = "x_" + _getPhysicsAbbreviation(solver_options) + "0"

        self.add_subsystem('mesh',
                           MachMesh(solver=solver, warper=warper),
                           promotes_outputs=[mesh_output])

class MachBuilder(Builder):
    def __init__(self, solver_type, solver_options, warper_type, warper_options, outputs, check_partials=False):
        self.solver_type = copy.deepcopy(solver_type)
        self.solver_options = copy.deepcopy(solver_options)
        self.warper_type = copy.deepcopy(warper_type)
        self.warper_options = copy.deepcopy(warper_options)
        self.outputs = copy.deepcopy(outputs)
        self.check_partials = check_partials
    
    def initialize(self, comm):
        # Create PDE solver instance
        self.comm = comm
        self.solver = PDESolver(type=self.solver_type,
                                solver_options=self.solver_options,
                                comm=comm)
        if (self.warper_type != "idwarp"):
            self.warper = MeshWarper(warper_options=self.warper_options,
                                     comm=comm)

    def get_coupling_group_subsystem(self, scenario_name=None):
        return MachCouplingGroup(solver=self.solver,
                                 check_partials=self.check_partials,
                                 scenario_name=scenario_name)

    def get_mesh_coordinate_subsystem(self, scenario_name=None):
        return MachMeshGroup(solver=self.solver,
                             warper=self.warper,
                             scenario_name=scenario_name)

    def get_pre_coupling_subsystem(self, scenario_name=None):
        return MachPrecouplingGroup(solver=self.solver,
                                    warper=self.warper,
                                    scenario_name=scenario_name)

    def get_post_coupling_subsystem(self, scenario_name=None):
        return MachOutputsGroup(solver=self.solver,
                                outputs=self.outputs,
                                check_partials=self.check_partials,
                                scenario_name=scenario_name)

    def get_number_of_nodes(self):
        """
        Get the number of state nodes on this processor
        """
        num_states = self.get_ndof()
        state_size = self.solver.getStateSize()
        return state_size // num_states

    def get_ndof(self):
        """
        Get the number of states per node
        """
        return self.solver.getNumStates()
