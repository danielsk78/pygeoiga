import pygmsh
import meshio
import numpy as np
from pygeoiga import myPath
import pyvista as pv
data_path = myPath+"/FE_solvers/data/"

#%%
def convert_single_NURB_to_gmsh(B, size=0.5, save_geo = data_path + "temp.geo", save_msh = data_path + "temp.msh",
                                mesh_file_type="msh2", binary=False):
    """
    We just need the sides of the surface. The filling will be made automatically all in 2d
    Args:
        B:

    Returns:

    """
    bot = B[:, 0]
    right = B[-1]
    top = np.flipud(B[:, -1])
    left = np.flipud(B[0])
    bot[..., -1] = np.zeros(bot[..., -1].shape) # make the weight to be 0 so t reads a 3 dimensional object
    right[..., -1] = np.zeros(right[..., -1].shape)
    top[..., -1] = np.zeros(top[..., -1].shape)
    left[..., -1] = np.zeros(left[..., -1].shape)
    geom = pygmsh.opencascade.Geometry()

    # add points
    p_bot = [geom.add_point(pos, size) for pos in bot] # exclude the last row for the weights
    p_right = [geom.add_point(pos, size) for pos in right[1:]] # exclude the first one because is already assigned
    p_right.insert(0, p_bot[-1])
    p_top = [geom.add_point(pos, size) for pos in top[1:]]
    p_top.insert(0, p_right[-1])
    p_left = [geom.add_point(pos, size) for pos in left[1:-1]]
    p_left.insert(0, p_top[-1])
    p_left.append(p_bot[0])  #to close the loop

    # add spline
    s_bot = geom.add_bspline(p_bot)
    s_right = geom.add_bspline(p_right)
    s_top = geom.add_bspline(p_top)
    s_left = geom.add_bspline(p_left)

    boundary = geom.add_line_loop([s_bot,s_right,s_top,s_left])
    pl = geom.add_plane_surface(boundary)
    geom.add_physical(pl, label="patch")

    geom.add_physical(s_bot, label="bot_bc")
    geom.add_physical(s_right, label="right_bc")
    geom.add_physical(s_top, label="top_bc")
    geom.add_physical(s_left, label="left_bc")

    physical_tag_id = {"bot_bc": 2, "right_bc": 3, "top_bc": 4, "left_bc": 5}
    #script = geom.get_code()
    #mesh = pygmsh.generate_mesh(geom, geo_filename=save_geo, msh_filename=save_msh,
    #                            mesh_file_type=mesh_file_type)  # To have it in ASCII format

    script = geom.get_code()
    mesh = pygmsh.generate_mesh(geom, mesh_file_type= mesh_file_type, dim=2)  # ,
    # geo_filename= save_geo,
    # msh_filename=save_msh,
    # extra_gmsh_arguments = ["-string"],#, "Mesh.SaveElementTagType=2;"],
    # mesh_file_type= mesh_file_type
    # mesh = pygmsh.generate_mesh(geom, geo_filename = "mesh.msh") #keep returning in binary so just run it by myself
    # )
    with open(save_geo, "w+") as f:
        f.write(script)
    if binary:
        args = [f"-{2}", save_geo, "-format", mesh_file_type, "-bin", "-o", save_msh]
    else:
        args = [f"-{2}", save_geo, "-format", mesh_file_type, "-o", save_msh]

    try:
        import subprocess
        p = subprocess.Popen(
            ["gmsh"] + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        p.communicate()
    except FileNotFoundError:
        print("Is gmsh installed?")
        raise
    return mesh, script, physical_tag_id

def _assign_orga(geometry, orga, patch_id):
    """
    Manages the contact faces so it doesn't repeat contact. This is of special use for the physical names.
    It will assign the repeating face to the next patch_id, and later read if already assign.
    Args:
        geometry: dictionary containing all the information of the geometry
        orga: internal manager for assigning repeated b-splines and points to the contacts
        patch_id: current patch id

    Returns:
        orga

    """
    patch_faces = geometry[patch_id].get("patch_faces")
    for ids in patch_faces.keys():
        contact = patch_faces[ids]
        con_face = geometry[contact].get("patch_faces")
        loc_face = [nface for nface in con_face.keys() if con_face[nface] == patch_id][0]
        if orga.get(contact) is None:  # if not existing then create a new one and assign the repeating point
            orga[contact] = {loc_face: orga[patch_id].get(ids)}
        if orga.get(contact) is not None:  # check if already created and then start assigning the points
            if orga[contact].get(loc_face) is None:  # check if the face is already assigned and continue
                orga[contact].update({loc_face: orga[patch_id].get(ids)})
    return orga

def convert_geometry_mp_to_gmsh(geometry, size=0.5, save_geo = data_path+"temp.geo", save_msh = data_path+"temp.msh",
                                mesh_file_type="msh2", binary=False):

    geom = pygmsh.built_in.Geometry()
    orga = {}
    physical_tag_id = {}
    n_patches = len(list(geometry.keys()))
    for count, patch_id in enumerate(geometry.keys()):
        # Points are always in 3D in gmsh. Be carefull
        B = geometry[patch_id].get("B")
        bot = B[:, 0]
        right = B[-1]
        top = np.flipud(B[:, -1])
        left = np.flipud(B[0])
        bot[..., -1] = np.zeros(bot[..., -1].shape)  # make the weight to be 0 so t reads a 3 dimensional object
        right[..., -1] = np.zeros(right[..., -1].shape)
        top[..., -1] = np.zeros(top[..., -1].shape)
        left[..., -1] = np.zeros(left[..., -1].shape)

        if orga.get(patch_id) is None: # for the first time we create the mesh
            p_bot = [geom.add_point(pos, size) for pos in bot]  # exclude the last row for the weights
            p_right = [geom.add_point(pos, size) for pos in
                       right[1:]]  # exclude the first one because is already assigned
            p_right.insert(0, p_bot[-1])
            p_top = [geom.add_point(pos, size) for pos in top[1:]]
            p_top.insert(0, p_right[-1])
            p_left = [geom.add_point(pos, size) for pos in left[1:-1]]
            p_left.insert(0, p_top[-1])
            p_left.append(p_bot[0])  # to close the loop

            # add spline
            s_bot = geom.add_bspline(p_bot)
            s_right = geom.add_bspline(p_right)
            s_top = geom.add_bspline(p_top)
            s_left = geom.add_bspline(p_left)

            boundary = geom.add_line_loop([s_bot, s_right, s_top, s_left])
            pl = geom.add_plane_surface(boundary)

            geom.add_physical(pl, label=patch_id)
            # save the points for the other patches because the mesh need to be constructed using the same points
            orga[patch_id] = {0: (p_bot, s_bot), 1: (p_right, s_right), 2: (p_top, s_top), 3: (p_left, s_left)}
            #orga = _assign_orga(geometry, orga, patch_id)

        else:
            is_bot = orga[patch_id].get(0)
            is_right = orga[patch_id].get(1)
            is_top = orga[patch_id].get(2)
            is_left = orga[patch_id].get(3)

            if is_bot is None:
                if is_left is not None and is_right is not None:
                    p_bot = [geom.add_point(pos, size) for pos in bot[1:-1]]
                    p_bot.insert(0, list(reversed(is_left[0]))[-1])
                    p_bot.append(list(reversed(is_right[0]))[0])
                elif is_left is not None:
                    p_bot = [geom.add_point(pos, size) for pos in bot[1:]]
                    p_bot.insert(0, list(reversed(is_left[0]))[-1]) # insert the last one found
                elif is_right is not None:
                    p_bot = [geom.add_point(pos, size) for pos in bot[:-1]]
                    p_bot.append(list(reversed(is_right[0]))[0])
                else:
                    p_bot = [geom.add_point(pos, size) for pos in bot]
                s_bot = geom.add_bspline(p_bot)
                orga[patch_id][0] = (p_bot, s_bot)
            else:
                p_bot = list(reversed(is_bot[0])) #because it was in the top of the previous one, therefore is inverted
                s_bot = -is_bot[1]
            if is_right is None:
                if is_top is not None:
                    p_right = [geom.add_point(pos, size) for pos in right[1:-1]]
                    p_right.insert(0, p_bot[-1])
                    p_right.append(list(reversed(is_top[0]))[0])
                else:
                    p_right = [geom.add_point(pos, size) for pos in right[1:]]
                    p_right.insert(0, p_bot[-1])
                s_right = geom.add_bspline(p_right)
                orga[patch_id][1] = (p_right, s_right)
            else:
                p_right = list(reversed(is_right[0]))
                s_right = -is_right[1]
            if is_top is None:
                if is_left is not None:
                    p_top = [geom.add_point(pos, size) for pos in top[1:-1]]
                    p_top.insert(0, p_right[-1])
                    p_top.append(list(reversed(is_left[0]))[0])
                else:
                    p_top = [geom.add_point(pos, size) for pos in top[1:]]
                    p_top.insert(0, p_right[-1])
                s_top = geom.add_bspline(p_top)
                orga[patch_id][2] = (p_top, s_top)
            else:
                p_top = list(reversed(is_top[0]))
                s_top = -is_top[1]#geom.add_bspline(p_top)
            if is_left is None:
                p_left = [geom.add_point(pos, size) for pos in left[1:-1]]
                p_left.insert(0, p_top[-1])
                p_left.append(p_bot[0])  # to close the loop
                s_left = geom.add_bspline(p_left)
                orga[patch_id][3] = (p_left, s_left)
            else:
                p_left = list(reversed(is_left[0]))
                s_left = -is_left[1]

            boundary = geom.add_line_loop([s_bot, s_right, s_top, s_left])
            pl = geom.add_plane_surface(boundary)
            geom.add_physical(pl, label=patch_id)
            #orga = _assign_orga(geometry, orga, patch_id)

        if n_patches > 1:
            orga = _assign_orga(geometry, orga, patch_id)
        count += 1  # to start from 1
        physical_tag_id.update({patch_id: count}) #get the number of the physical tag starting from 1
    ### Add physical to Boundary conditions
    bot_bc = []
    right_bc =[]
    top_bc = []
    left_bc = []

    for patch_id in geometry.keys():
        BC = geometry[patch_id].get("BC")
        if BC is not None:
            for key in BC.keys(): # corresponding to the face
                name = BC[key]
                spl_BC = orga[patch_id][key][1]  # to catch the spline in that part
                if name[:3] == "bot":
                    if spl_BC not in bot_bc:
                        bot_bc.append(spl_BC)
                elif name[:5] =="right":
                    if spl_BC not in right_bc:
                        right_bc.append(spl_BC)
                elif name[:3] =="top":
                    if spl_BC not in top_bc:
                        top_bc.append(spl_BC)
                elif name[:4] =="left":
                    if spl_BC not in left_bc:
                        left_bc.append(spl_BC)
                else:
                    print(name, "not known")
                    raise NotImplementedError
    if len(bot_bc) > 0:
        geom.add_physical(bot_bc, label="bot_bc")
        count += 1
        physical_tag_id.update({"bot_bc": count})
    if len(right_bc) > 0:
        geom.add_physical(right_bc, label="right_bc")
        count += 1
        physical_tag_id.update({"right_bc": count})
    if len(top_bc) > 0:
        geom.add_physical(top_bc, label="top_bc")
        count += 1
        physical_tag_id.update({"top_bc": count})
    if len(left_bc) > 0:
        geom.add_physical(left_bc, label="left_bc")
        count += 1
        physical_tag_id.update({"left_bc": count})

    script = geom.get_code()
    with open(save_geo, "w+") as f:
        f.write(script)
    mesh = pygmsh.generate_mesh(geom, mesh_file_type= mesh_file_type, dim=2)#,
                                #geo_filename= save_geo,
                                #msh_filename=save_msh,
                                #extra_gmsh_arguments = ["-string"],#, "Mesh.SaveElementTagType=2;"],
                               # mesh_file_type= mesh_file_type
        #mesh = pygmsh.generate_mesh(geom, geo_filename = "mesh.msh") #keep returning in binary so just run it by myself
                               # )

    if binary:
        args = [f"-{2}", save_geo, "-format", mesh_file_type, "-bin", "-o", save_msh]
    else:
        args = [f"-{2}", save_geo, "-format", mesh_file_type, "-o", save_msh]
    try:
        import subprocess
        p = subprocess.Popen(
            ["gmsh"] + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        p.communicate()
    except FileNotFoundError:
        print("Is gmsh installed?")
        raise

    return mesh, script, physical_tag_id

def write_mesh(mesh, filename="test.vtk"):
    meshio.write(data_path+filename, mesh)
    print("Mesh created in folder:" + data_path + filename)

def plot_mesh(mesh):
    mesh.write(data_path + "out.vtk")
    # read the data
    grid = pv.read(data_path+"out.vtk")
    # plot the data with an automatically created Plotter
    grid.plot(show_scalar_bar=True, show_axes=True)
