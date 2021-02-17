from pygeoiga import myPath
import matplotlib.pyplot as plt
datapath = myPath+"/FE_solvers/data/"

def test_create_script_3_layer():
    from pygeoiga.nurb.cad import make_3_layer_patches
    from pygeoiga.nurb.nrb_to_gmsh import convert_geometry_mp_to_gmsh
    geometry = make_3_layer_patches(refine=False)
    mesh, script, physical_tag_id = convert_geometry_mp_to_gmsh(geometry, size=10,
                                                                save_geo=datapath + "3_layer_anticline.geo",
                                                                save_msh=datapath + "3_layer_anticline.msh")
    from pygeoiga.FE_solvers.run_moose import create_script
    input = datapath + "3_layer_anticline.msh"
    create_script(input, topology_info=physical_tag_id, bot_bc=10, top_bc=40, geometry=geometry)

def test_create_script_square():
    from pygeoiga.FE_solvers.run_moose import create_script
    from pygeoiga.nurb.nrb_to_gmsh import convert_single_NURB_to_gmsh
    from pygeoiga.nurb.cad import make_surface_square
    U,V, B = make_surface_square()
    mesh, script, physical_tag_id = convert_single_NURB_to_gmsh(B, size=0.5,
                                                                save_geo=datapath + "square_surf.geo",
                                                                save_msh=datapath + "square_surf.msh")
    input = datapath + "square_surf.msh"
    create_script(input, topology_info=physical_tag_id,
                   right_bc=10, left_bc=20, geometry=None, kappa = 4)

def test_create_script_quarter_disk():
    from pygeoiga.FE_solvers.run_moose import create_script
    from pygeoiga.nurb.nrb_to_gmsh import convert_single_NURB_to_gmsh
    from pygeoiga.nurb.cad import quarter_disk
    k, B = quarter_disk()
    mesh, script, physical_tag_id = convert_single_NURB_to_gmsh(B, size=0.5,
                                                                save_geo=datapath + "quarter_disk.geo",
                                                                save_msh=datapath + "quarter_disk.msh")
    input = datapath + "quarter_disk.msh"
    create_script(input, topology_info=physical_tag_id,
                    right_bc=10, left_bc=40, geometry=None, kappa = 4)

def test_create_script_biquadratic():
    from pygeoiga.FE_solvers.run_moose import create_script
    from pygeoiga.nurb.nrb_to_gmsh import convert_single_NURB_to_gmsh
    from pygeoiga.nurb.cad import make_surface_biquadratic
    k, B = make_surface_biquadratic()
    mesh, script, physical_tag_id = convert_single_NURB_to_gmsh(B, size=0.5,
                                               save_geo=datapath + "biquadratic.geo",
                                               save_msh=datapath + "biquadratic.msh")
    input = datapath + "biquadratic.msh"
    create_script(input, topology_info=physical_tag_id,
                   bot_bc=40, top_bc=10, geometry=None, kappa=4)

def test_create_script_fault():
    from pygeoiga.nurb.cad import make_fault_model
    from pygeoiga.nurb.nrb_to_gmsh import convert_geometry_mp_to_gmsh
    geometry = make_fault_model(refine=False)
    mesh, script, physical_tag_id = convert_geometry_mp_to_gmsh(geometry, size=50,
                                               save_geo=datapath + "fault_model.geo",
                                               save_msh=datapath + "fault_model.msh")
    from pygeoiga.FE_solvers.run_moose import create_script
    input = datapath + "fault_model.msh"
    create_script(input, topology_info=physical_tag_id, top_bc=10, bot_bc=40, geometry=geometry)

def test_create_script_unconformity():
    from pygeoiga.nurb.cad import make_unconformity_model
    from pygeoiga.nurb.nrb_to_gmsh import convert_geometry_mp_to_gmsh
    geometry = make_unconformity_model(refine=False)
    mesh, script, physical_tag_id = convert_geometry_mp_to_gmsh(geometry, size=50,
                                               save_geo=datapath + "unconformity_model.geo",
                                               save_msh=datapath + "unconformity_model.msh")
    from pygeoiga.FE_solvers.run_moose import create_script
    input = datapath + "unconformity_model.msh"
    create_script(input, topology_info=physical_tag_id, top_bc=10, bot_bc=40, geometry=geometry)

def test_create_script_salt_dome():
    from pygeoiga.nurb.cad import make_salt_dome
    from pygeoiga.nurb.nrb_to_gmsh import convert_geometry_mp_to_gmsh
    geometry = make_salt_dome(refine=False)
    mesh, script, physical_tag_id = convert_geometry_mp_to_gmsh(geometry, size=100,
                                               save_geo=datapath + "salt_dome_model.geo",
                                               save_msh=datapath + "salt_dome_model.msh")
    from pygeoiga.FE_solvers.run_moose import create_script
    input = datapath + "salt_dome_model.msh"
    create_script(input, topology_info=physical_tag_id, top_bc=10, bot_bc=40, geometry=geometry)



