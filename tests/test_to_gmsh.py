from pygeoiga.nurb.nrb_to_gmsh import convert_geometry_mp_to_gmsh, convert_single_NURB_to_gmsh, plot_mesh, write_mesh
from pygeoiga import myPath
data_path = myPath+"/FE_solvers/data/"

def test_mesh_square():
    from pygeoiga.nurb.cad import make_surface_square
    U, V, B = make_surface_square()
    mesh, script, physical_tag_id = convert_single_NURB_to_gmsh(B, size=10,
                                               save_geo = data_path + "square_surf.geo",
                                               save_msh = data_path + "square_surf.msh")
    print(script)
    plot_mesh(mesh)


def test_mesh_quarter_disk():
    from pygeoiga.nurb.cad import quarter_disk
    k, B = quarter_disk()
    mesh, script, physical_tag_id = convert_single_NURB_to_gmsh(B, size=0.5,
                                               save_geo=data_path + "quarter_disk.geo",
                                               save_msh=data_path + "quarter_disk.msh")
    print(script)
    plot_mesh(mesh)

def test_mesh_biquadratic():
    from pygeoiga.nurb.cad import make_surface_biquadratic
    k, B = make_surface_biquadratic()
    mesh, script, physical_tag_id = convert_single_NURB_to_gmsh(B, size=0.5,
                                               save_geo=data_path + "biquadratic.geo",
                                               save_msh=data_path + "biquadratic.msh")
    print(script)
    plot_mesh(mesh)

def test_mesh_mp_3_layer():
    from pygeoiga.nurb.cad import make_3_layer_patches
    geometry = make_3_layer_patches()
    mesh, script, physical_tag_id = convert_geometry_mp_to_gmsh(geometry, size=50,
                                               save_geo=data_path + "3_layer_anticline.geo",
                                               save_msh=data_path + "3_layer_anticline.msh",
                                               mesh_file_type="msh2")
    print(script, physical_tag_id)
    plot_mesh(mesh)

def test_mesh_mp_fault():
    from pygeoiga.nurb.cad import make_fault_model
    geometry = make_fault_model(refine=False)
    mesh, script, physical_tag_id = convert_geometry_mp_to_gmsh(geometry, size=50,
                                               save_geo=data_path + "fault_model.geo",
                                               save_msh=data_path + "fault_model.msh")
    print(script, physical_tag_id)
    plot_mesh(mesh)

def test_mesh_mp_unconformity():
    from pygeoiga.nurb.cad import make_unconformity_model
    geometry = make_unconformity_model()
    mesh, script, physical_tag_id = convert_geometry_mp_to_gmsh(geometry, size=10,
                                               save_geo=data_path + "unconformity_model.geo",
                                               save_msh=data_path + "unconformity_model.msh")
    print(script, physical_tag_id)
    plot_mesh(mesh)

def test_mesh_mp_salt_dome():
    from pygeoiga.nurb.cad import make_salt_dome
    geometry = make_salt_dome(refine=False)
    mesh, script, physical_tag_id = convert_geometry_mp_to_gmsh(geometry, size=100,
                                               save_geo=data_path + "salt_dome_model.geo",
                                               save_msh=data_path + "salt_dome_model.msh",
                                               mesh_file_type="msh2")
    print(script,physical_tag_id)
    plot_mesh(mesh)

