import copy
import numpy as np
import openmdao.api as om
from mphys import Builder

from miso.pyMISO import PDESolver, MeshWarper
from miso.miso_state import MISOState
from miso.miso_output import MISOFunctional
from miso.mesh_warper import MISOMeshWarper

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
        raise RuntimeError("Bad physics given to MISOSolver!")

class MISOCouplingGroup(om.Group):
    def initialize(self):
        self.options.declare("solver", types=PDESolver, recordable=False)
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
        self.add_subsystem("state",
                           MISOState(solver=self.solver,
                                     depends=["mesh_coords", *self.depends],
                                     check_partials=self.check_partials),
                           promotes_inputs=[*self.depends, ("mesh_coords", mesh_input)],
                           promotes_outputs=[("state", state_output)])

class MISOPrecouplingGroup(om.Group):
    """
    Group that handles surface -> volume mesh movement

    To properly support parallel analysis, I"ll need to have this component
    partition the input surface coords
    """
    def initialize(self):
        self.options.declare("solver", types=PDESolver, recordable=False)
        self.options.declare("warper", recordable=False) # formerly types=MeshWarper
        self.options.declare("scenario_name", default=None)

    def setup(self):
        self.solver = self.options["solver"]
        self.warper = self.options["warper"]

        # Promote variables with physics-specific tag that MPhys expects
        solver_options = self.solver.getOptions()
        mesh_input = "x_" + _getPhysicsAbbreviation(solver_options)
        mesh_output = "x_" + _getPhysicsAbbreviation(solver_options) + "_vol"
        # NOTE: Added conditional logic to allow for turning off mesh warper
        # TODO: Determine if this OK
        if isinstance(self.warper, MeshWarper):
            self.add_subsystem("mesh_warper",
                            MISOMeshWarper(warper=self.warper),
                            promotes_inputs=[("surf_mesh_coords", mesh_input)],
                            promotes_outputs=[("vol_mesh_coords", mesh_output)])

class MISOOutputsGroup(om.Group):
    """
    Group that handles calculating outputs after the state solve
    """
    def initialize(self):
        self.options.declare("solver", types=PDESolver, recordable=False)
        self.options.declare("outputs", types=dict, default=None)
        self.options.declare("check_partials", default=False)
        self.options.declare("scenario_name", default=None)

    def setup(self):
        self.solver = self.options["solver"]
        self.outputs = self.options["outputs"]
        self.check_partials = self.options["check_partials"]

        # Promote variables with physics-specific tag that MPhys expects
        solver_options = self.solver.getOptions()
        mesh_input = "x_" + _getPhysicsAbbreviation(solver_options) + "_vol"
        state_input = _getPhysicsAbbreviation(solver_options) + "_state"
        promoted_inputs = {
            "mesh_coords": ("mesh_coords", mesh_input),
            "state": ("state", state_input)
        }

        for output in self.outputs:
            if "options" in self.outputs[output]:
                output_opts = self.outputs[output]["options"]
            else:
                output_opts = None

            if "depends" in self.outputs[output]:
                depends = self.outputs[output]["depends"]
            else:
                depends = []

            if isinstance(output, str):
                output_prom_name = output
                func_name = output
            elif isinstance(output, tuple):
                output_prom_name = output[0]
                func_name = output[1]
            else:
                raise RuntimeError(f"Unrecognized type for output: {output}!")

            depends = [promoted_inputs[input] if input in promoted_inputs else input for input in depends]
            self.add_subsystem(output_prom_name,
                               MISOFunctional(solver=self.solver,
                                             func=func_name,
                                             func_options=output_opts,
                                             depends=depends),
                               promotes_inputs=[*depends],
                               promotes_outputs=[(func_name, output_prom_name)])


class MISOMesh(om.IndepVarComp):
    """
    Component to read the initial surface mesh coordinates
    """

    def initialize(self):
        self.options.declare("solver", default=None, desc="the miso solver object itself", recordable=False)
        self.options.declare("warper", default=None, desc="the mesh warper object itself", recordable=False)
        self.options.declare("scenario_name", default=None)

    def setup(self):
        solver = self.options["solver"]
        warper = self.options["warper"]

        if isinstance(warper, MeshWarper):
            local_surf_mesh_size = warper.getSurfaceCoordsSize()
            surf_coords = np.empty(local_surf_mesh_size)
            warper.getInitialSurfaceCoords(surf_coords)
        else:
            raise NotImplementedError("MISOMesh class only implemented for MISOMeshWarper!\n")

        solver_options = solver.getOptions()
        mesh_output = "x_" + _getPhysicsAbbreviation(solver_options) + "0"

        self.add_output(mesh_output,
                        distributed=True,
                        val=surf_coords,
                        desc="surface mesh node coordinates",
                        tags=["mphys_coordinates"])

class MISOMeshGroup(om.Group):
    def initialize(self):
        self.options.declare("solver", default=None, desc="the miso solver object itself", recordable=False)
        self.options.declare("warper", default=None, desc="the mesh warper object itself", recordable=False)
        self.options.declare("scenario_name", default=None)

    def setup(self):
        solver = self.options["solver"]
        warper = self.options["warper"]

        solver_options = solver.getOptions()
        mesh_output = "x_" + _getPhysicsAbbreviation(solver_options) + "0"

        self.add_subsystem("mesh",
                           MISOMesh(solver=solver, warper=warper),
                           promotes_outputs=[mesh_output])

class MISOBuilder(Builder):
    def __init__(self, solver_type, solver_options, solver_inputs, warper_type, warper_options, outputs, check_partials=False):
        self.solver_type = copy.deepcopy(solver_type)
        self.solver_options = copy.deepcopy(solver_options)
        self.solver_inputs = copy.deepcopy(solver_inputs)
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
        # NOTE: Adding conditional logic to allow for turning off mesh warper
        # TODO: Determine if this is OK
        if (self.warper_type == None):
            self.warper = None
        elif (self.warper_type != "idwarp"):
            self.warper = MeshWarper(warper_options=self.warper_options,
                                     comm=comm)

    def get_coupling_group_subsystem(self, scenario_name=None):
        return MISOCouplingGroup(solver=self.solver,
                                 depends=self.solver_inputs,
                                 check_partials=self.check_partials,
                                 scenario_name=scenario_name)

    def get_mesh_coordinate_subsystem(self, scenario_name=None):
        return MISOMeshGroup(solver=self.solver,
                             warper=self.warper,
                             scenario_name=scenario_name)

    def get_pre_coupling_subsystem(self, scenario_name=None):
        return MISOPrecouplingGroup(solver=self.solver,
                                    warper=self.warper,
                                    scenario_name=scenario_name)

    def get_post_coupling_subsystem(self, scenario_name=None):
        return MISOOutputsGroup(solver=self.solver,
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
