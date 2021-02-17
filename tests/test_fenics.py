from pygeoiga import myPath
import matplotlib.pyplot as plt
datapath = myPath+"/FE_solvers/data/"

def test_convert_msh_to_xml_DEPRECAED():
    from pygeoiga.FE_solvers.run_fenics import _convert_msh_to_xml
    input = datapath + "3_layer_anticline.msh"
    output = datapath + "3_layer_anticline.xml"
    _convert_msh_to_xml(input, output)

def test_convert_msh_to_xdmf():
    from pygeoiga.FE_solvers.run_fenics import convert_msh_to_xdmf
    input = datapath + "3_layer_anticline.msh"
    convert_msh_to_xdmf(input)

def test_run_workflow_square():
    from pygeoiga.FE_solvers.run_fenics import convert_msh_to_xdmf, run_simulation
    from pygeoiga.nurb.nrb_to_gmsh import convert_single_NURB_to_gmsh
    from pygeoiga.nurb.cad import make_surface_square
    U,V, B = make_surface_square()
    mesh, script, physical_tag_id = convert_single_NURB_to_gmsh(B, size=0.5,
                                                                save_geo=datapath + "square_surf.geo",
                                                                save_msh=datapath + "square_surf.msh")
    input = datapath + "square_surf.msh"
    plt.pause(3)
    convert_msh_to_xdmf(input)
    run_simulation(input, topology_info=physical_tag_id,
                   right_bc=10, left_bc=20, geometry=None, kappa = 4)

def test_run_workflow_quarter_disk():
    from pygeoiga.FE_solvers.run_fenics import convert_msh_to_xdmf, run_simulation
    from pygeoiga.nurb.nrb_to_gmsh import convert_single_NURB_to_gmsh
    from pygeoiga.nurb.cad import quarter_disk
    k, B = quarter_disk()
    mesh, script, physical_tag_id = convert_single_NURB_to_gmsh(B, size=0.5,
                                                                save_geo=datapath + "quarter_disk.geo",
                                                                save_msh=datapath + "quarter_disk.msh")
    input = datapath + "quarter_disk.msh"
    plt.pause(3) #needs a bit of time to create the files
    convert_msh_to_xdmf(input)
    run_simulation(input, topology_info=physical_tag_id,
                   right_bc=10, left_bc=40, geometry=None, kappa = 4)

def test_run_workflow_biquadratic():
    from pygeoiga.FE_solvers.run_fenics import convert_msh_to_xdmf, run_simulation
    from pygeoiga.nurb.nrb_to_gmsh import convert_single_NURB_to_gmsh
    from pygeoiga.nurb.cad import make_surface_biquadratic
    k, B = make_surface_biquadratic()
    mesh, script, physical_tag_id = convert_single_NURB_to_gmsh(B, size=0.5,
                                               save_geo=datapath + "biquadratic.geo",
                                               save_msh=datapath + "biquadratic.msh")
    input = datapath + "biquadratic.msh"
    plt.pause(3)  # needs a bit of time to create the files
    convert_msh_to_xdmf(input)
    run_simulation(input, topology_info=physical_tag_id,
                   bot_bc=40, top_bc=10, geometry=None, kappa=4)

def test_run_workflow_3_layer_patch():
    from pygeoiga.nurb.cad import make_3_layer_patches
    from pygeoiga.nurb.nrb_to_gmsh import convert_geometry_mp_to_gmsh
    geometry = make_3_layer_patches(refine=False)
    mesh, script, physical_tag_id = convert_geometry_mp_to_gmsh(geometry, size=10,
                                               save_geo=datapath + "3_layer_anticline.geo",
                                               save_msh=datapath + "3_layer_anticline.msh")
    from pygeoiga.FE_solvers.run_fenics import convert_msh_to_xdmf, run_simulation
    input = datapath + "3_layer_anticline.msh"
    convert_msh_to_xdmf(input)
    nodal_coordinates, temperature_nodes = run_simulation(input, topology_info=physical_tag_id, top_bc=10, bot_bc=40, geometry=geometry)

def test_run_workflow_fault():
    from pygeoiga.nurb.cad import make_fault_model
    from pygeoiga.nurb.nrb_to_gmsh import convert_geometry_mp_to_gmsh
    geometry = make_fault_model(refine=False)
    mesh, script, physical_tag_id = convert_geometry_mp_to_gmsh(geometry, size=50,
                                               save_geo=datapath + "fault_model.geo",
                                               save_msh=datapath + "fault_model.msh")
    from pygeoiga.FE_solvers.run_fenics import convert_msh_to_xdmf, run_simulation
    input = datapath + "fault_model.msh"
    convert_msh_to_xdmf(input)
    nodal_coordinates, temperature_nodes = run_simulation(input, topology_info=physical_tag_id, top_bc=10, bot_bc=40, geometry=geometry)

def test_run_workflow_unconformity():
    #carefull: Is too short the corner so it cannot be plotted
    from pygeoiga.nurb.cad import make_unconformity_model
    from pygeoiga.nurb.nrb_to_gmsh import convert_geometry_mp_to_gmsh
    geometry = make_unconformity_model(refine=False)
    mesh, script, physical_tag_id = convert_geometry_mp_to_gmsh(geometry, size=50,
                                               save_geo=datapath + "unconformity_model.geo",
                                               save_msh=datapath + "unconformity_model.msh")
    from pygeoiga.FE_solvers.run_fenics import convert_msh_to_xdmf, run_simulation
    input = datapath + "unconformity_model.msh"
    convert_msh_to_xdmf(input)
    nodal_coordinates, temperature_nodes = run_simulation(input, topology_info=physical_tag_id, top_bc=10, bot_bc=40, geometry=geometry)

def test_run_workflow_salt_dome():
    from pygeoiga.nurb.cad import make_salt_dome
    from pygeoiga.nurb.nrb_to_gmsh import convert_geometry_mp_to_gmsh
    geometry = make_salt_dome(refine=False)
    mesh, script, physical_tag_id = convert_geometry_mp_to_gmsh(geometry, size=200,
                                               save_geo=datapath + "salt_dome_model.geo",
                                               save_msh=datapath + "salt_dome_model.msh")
    from pygeoiga.FE_solvers.run_fenics import convert_msh_to_xdmf, run_simulation
    input = datapath + "salt_dome_model.msh"
    convert_msh_to_xdmf(input)
    nodal_coordinates, temperature_nodes = run_simulation(input, topology_info=physical_tag_id, top_bc=10, bot_bc=90, geometry=geometry)

def test_plot_local_solution_salt():
    from pygeoiga.nurb.cad import make_salt_dome
    from pygeoiga.nurb.nrb_to_gmsh import convert_geometry_mp_to_gmsh
    geometry = make_salt_dome(refine=False)
    mesh, script, physical_tag_id = convert_geometry_mp_to_gmsh(geometry, size=100,
                                                                save_geo=datapath + "salt_dome_model.geo",
                                                                save_msh=datapath + "salt_dome_model.msh")
    from pygeoiga.FE_solvers.run_fenics import convert_msh_to_xdmf, run_simulation, p_temperature_fenics
    input = datapath + "salt_dome_model.msh"
    convert_msh_to_xdmf(input)
    nodal_coordinates, temperature_nodes = run_simulation(input, topology_info=physical_tag_id, top_bc=10, bot_bc=40,
                                                          geometry=geometry, show=False)
    from pygeoiga.plot.solution_mpl import p_temperature
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax = p_temperature(nodal_coordinates[:,0],
                       nodal_coordinates[:,1],
                       temperature_nodes,
                       vmin=temperature_nodes.min(),
                       vmax=temperature_nodes.max(),
                       ax=ax,
                       point=False,
                       fill=True,
                       contour=False,
                       colorbar=True,
                       levels=100)
    ax = p_temperature(nodal_coordinates[:, 0],
                       nodal_coordinates[:, 1],
                       temperature_nodes,
                       vmin=temperature_nodes.min(),
                       vmax=temperature_nodes.max(),
                       ax=ax,
                       point=False,
                       fill=False,
                       contour=True,
                       colorbar=False,
                       levels=10)
    fig.show()
    print(len(temperature_nodes))
    #695 200
    #2381 100
    fig.savefig(datapath + "salt_dome_model_100.png")
    #p_temperature_fenics(nodal_coordinates, temperature_nodes, levels=50, show=False, colorbar=True, ax=ax,
     #                point=False, fill=True, contour=False)
    #fig.show()

def test_plot_local_solution_unconformity():
    from pygeoiga.nurb.cad import make_unconformity_model
    from pygeoiga.nurb.nrb_to_gmsh import convert_geometry_mp_to_gmsh
    geometry = make_unconformity_model(refine=False)
    mesh, script, physical_tag_id = convert_geometry_mp_to_gmsh(geometry, size=50,
                                                                save_geo=datapath + "unconformity_model.geo",
                                                                save_msh=datapath + "unconformity_model.msh")
    from pygeoiga.FE_solvers.run_fenics import convert_msh_to_xdmf, run_simulation
    input = datapath + "unconformity_model.msh"
    convert_msh_to_xdmf(input)
    nodal_coordinates, temperature_nodes = run_simulation(input, topology_info=physical_tag_id, top_bc=10, bot_bc=40,
                                                          geometry=geometry, show=False)
    from pygeoiga.plot.solution_mpl import p_temperature
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax = p_temperature(nodal_coordinates[:,0],
                       nodal_coordinates[:,1],
                       temperature_nodes,
                       vmin=temperature_nodes.min(),
                       vmax=temperature_nodes.max(),
                       ax=ax,
                       point=True,
                       fill=False)
    fig.show()