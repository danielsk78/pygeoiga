#%%
import matplotlib.pyplot as plt
import numpy as np
from pygeoiga import myPath
datapath = myPath+"/FE_solvers/data/"
#from fenics import *

#%%

def _convert_msh_to_xml(input_file_path_msh, output_file_path_msh):
    """
    DEPECRATED
    Args:
        input_file_path_msh:
        output_file_path_msh:

    Returns:

    """
    try:
        import subprocess
        p = subprocess.Popen(
            ["dolfin-convert"] + [input_file_path_msh] + [output_file_path_msh], stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
    except FileNotFoundError:
        print("Is dolfin installed?")
        raise

def convert_msh_to_xdmf(filepath):
    import meshio
    msh = meshio.read(filepath)
    for cell in msh.cells:
        if cell.type == "triangle":
            triangle_cells = cell.data
        elif  cell.type == "line":
            line_cells = cell.data

    for key in msh.cell_data_dict["gmsh:physical"].keys():
        if key == "line":
            line_data = msh.cell_data_dict["gmsh:physical"][key]
        elif key == "triangle":
            triangle_data = msh.cell_data_dict["gmsh:physical"][key]

    triangle_mesh = meshio.Mesh(points=msh.points,
                                cells={"triangle": triangle_cells},
                                cell_data={"subdomains":[triangle_data]})
    line_mesh =meshio.Mesh(points=msh.points,
                               cells={"line": line_cells},
                               cell_data={"boundary_conditions":[line_data]})

    meshio.write("%s_triangle.xdmf"%filepath.split('.')[0], triangle_mesh)
    meshio.write("%s_line.xdmf"%filepath.split('.')[0], line_mesh)
    print("successfully saved in:"+filepath.split('.')[0])

def run_simulation(filepath,
                   topology_info:int=None,
                   top_bc:int=None,
                   bot_bc:int=None,
                   left_bc:int=None,
                   right_bc:int = None,
                   geometry: dict = None,
                   kappa = 3, #only if geometry is None
                   show=True,
                   save_solution=False
                   ):

    from dolfin import (Mesh, XDMFFile,MeshValueCollection, cpp, FunctionSpace,
                        TrialFunction, TestFunction, DirichletBC, Constant,
                        Measure, inner, nabla_grad, Function, solve, plot, File)
    mesh = Mesh()
    with XDMFFile("%s_triangle.xdmf"%filepath.split('.')[0]) as infile:
        infile.read(mesh) # read the complete mesh

    mvc_subdo = MeshValueCollection("size_t", mesh, mesh.geometric_dimension() - 1)
    with XDMFFile("%s_triangle.xdmf"%filepath.split('.')[0]) as infile:
        infile.read(mvc_subdo, "subdomains") # read the diferent subdomians
    subdomains = cpp.mesh.MeshFunctionSizet(mesh, mvc_subdo)

    mvc = MeshValueCollection("size_t", mesh, mesh.geometric_dimension() - 2)
    with XDMFFile("%s_line.xdmf"%filepath.split('.')[0]) as infile:
        infile.read(mvc, "boundary_conditions") #read the boundary conditions
    boundary = cpp.mesh.MeshFunctionSizet(mesh, mvc)

    # Define function space and basis functions
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    # Boundary conditions
    bcs =[]
    for bc_id in topology_info.keys():
        if bc_id[-2:] == "bc":
            if bot_bc is not None and bc_id[:3] == "bot":
                bcs.append(DirichletBC(V, Constant(bot_bc), boundary, topology_info[bc_id]))
            elif left_bc is not None and bc_id[:4] == "left":
                bcs.append(DirichletBC(V, Constant(left_bc), boundary, topology_info[bc_id]))
            elif top_bc is not None and bc_id[:3] == "top":
                bcs.append(DirichletBC(V, Constant(top_bc), boundary, topology_info[bc_id]))
            elif right_bc is not None and bc_id[:5] == "right":
                bcs.append(DirichletBC(V, Constant(right_bc), boundary, topology_info[bc_id]))
            else:
                print(bc_id + " Not assigned as boundary condition ")
            #    raise NotImplementedError

    # Define new measures associated with the interior domains and
    # exterior boundaries
    dx = Measure("dx", subdomain_data=subdomains)
    ds = Measure("ds", subdomain_data=boundary)

    f = Constant(0)
    g = Constant(0)
    if geometry is not None: # run multipatch implementation (Multiple domains)
        a = []
        L = []
        for patch_id in geometry.keys():
            kappa = geometry[patch_id].get("kappa")
            a.append(inner(Constant(kappa) * nabla_grad(u), nabla_grad(v)) * dx(topology_info[patch_id]))
            L.append(f*v*dx(topology_info[patch_id]))
        a = sum(a)
        L = sum(L)
    else:
        a = inner(Constant(kappa) * nabla_grad(u), nabla_grad(v)) * dx
        L = f * v * dx

    ## Redefine u as a function in function space V for the solution
    u = Function(V)
    # Solve
    solve(a == L, u, bcs)
    u.rename('u', 'Temperature')
    # Save solution to file in VTK format
    print('  [+] Output to %s_solution.pvd' % filepath.split('.')[0])
    vtkfile = File('%s_solution.pvd' % filepath.split('.')[0])
    vtkfile << u

    if show:
        import matplotlib
        matplotlib.use("Qt5Agg")
        # Plot solution and gradient
        plot(u, title="Temperature")
        plt.show()

    dofs = V.tabulate_dof_coordinates().reshape(V.dim(), mesh.geometry().dim()) #coordinates of nodes
    vals = u.vector().get_local() #temperature at nodes

    if save_solution:
        from dolfin import HDF5File, MPI
        output_file = HDF5File(MPI.comm_world, filepath.split('.')[0] + "_solution_field.h5", "w")
        output_file.write(u, "solution")
        output_file.close()
    u.set_allow_extrapolation(True)
    return dofs, vals, mesh, u

def read_fenics_solution(filepath):
    from dolfin import (Mesh, XDMFFile, MeshValueCollection, cpp, FunctionSpace, Function, HDF5File, MPI)
    mesh = Mesh()
    with XDMFFile("%s_triangle.xdmf" % filepath.split('.')[0]) as infile:
        infile.read(mesh)  # read the complete mesh

    #mvc_subdo = MeshValueCollection("size_t", mesh, mesh.geometric_dimension() - 1)
    #with XDMFFile("%s_triangle.xdmf" % filepath.split('.')[0]) as infile:
    #    infile.read(mvc_subdo, "subdomains")  # read the diferent subdomians
    #subdomains = cpp.mesh.MeshFunctionSizet(mesh, mvc_subdo)

    #mvc = MeshValueCollection("size_t", mesh, mesh.geometric_dimension() - 2)
    #with XDMFFile("%s_line.xdmf" % filepath.split('.')[0]) as infile:
    #    infile.read(mvc, "boundary_conditions")  # read the boundary conditions
    #boundary = cpp.mesh.MeshFunctionSizet(mesh, mvc)

    # Define function space and basis functions
    V = FunctionSpace(mesh, "CG", 1)
    U = Function(V)
    input_file = HDF5File(MPI.comm_world, filepath.split('.')[0] + "_solution_field.h5", "r")
    input_file.read(U, "solution")
    input_file.close()

    dofs = V.tabulate_dof_coordinates().reshape(V.dim(), mesh.geometry().dim())  # coordinates of nodes
    U.set_allow_extrapolation(True)
    return U, mesh, dofs.shape[0]


def p_temperature_fenics(coordinates, temperature, typ="2d", ax=None, inter_points=100, **kwargs):
    from pygeoiga.plot.solution_mpl import p_temperature
    # if ax is None:
    #    if typ=="2d":
    #        fig, ax = plt.subplots()
    #        ax.set_aspect("equal")
    #    elif typ=="3d":
    #        fig = plt.figure()
    #        ax = fig.add_subplot(111, projection='3d')

    x_temp = coordinates[:,0]
    y_temp = coordinates[:,1]
    t_temp = temperature
    # Set up a regular grid of interpolation points
    xi = np.linspace(x_temp.min(), x_temp.max(), inter_points)
    yi = np.linspace(y_temp.min(), y_temp.max(), inter_points)
    xi, yi = np.meshgrid(xi, yi)

    import scipy.interpolate
    ti = scipy.interpolate.griddata((x_temp, y_temp), t_temp, (xi, yi), method='linear')

    p_temperature(x_pos=None, y_pos=None, temperature=ti,
                  extent=[x_temp.min(), x_temp.max(), y_temp.min(), y_temp.max()], **kwargs)

    # p_temperature(coordinates[:,0], coordinates[:,1])


