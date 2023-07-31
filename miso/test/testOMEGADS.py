from miso import omEGADS
import openmdao.api as om


if __name__ == "__main__":
    model = om.Group()
    ivc = om.IndepVarComp()

    ivc.add_output('thickness', 2.9) # thickness of annulus

    model.add_subsystem('des_vars', ivc)
    model.add_subsystem('mesh', omEGADS(csm_file='data/testOMEGADS/quarter_annulus',
                                        model_file='data/testOMEGADS/quarter_annulus.egads',
                                        mesh_file='data/testOMEGADS/quarter_annulus.smb',
                                        tess_file='data/testOMEGADS/quarter_annulus.eto'))
    
    model.connect('des_vars.thickness', 'mesh.thickness')

    prob = om.Problem(model)
    prob.setup()
    prob.run_model()