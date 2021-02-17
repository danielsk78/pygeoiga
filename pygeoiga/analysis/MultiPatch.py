from pygeoiga.analysis.common import IEN_element_topology
from pygeoiga.analysis.iga import map_solution_elements
import numpy as np

def patch_topology(geometry):
    """
    The idea is to have a numbering of the internal elements and control points with respect to
    the neighbour patches. This mean that it will assign a value to each control point (global degree of freedom)
    this will be important when using this value relate the control point to each patch.
    Args:
        list_B: will be a list containing all the control points.

    Returns:
        patch_topology

    """

    cp_all = [] # Save all the previous control points to identify wich ones are repeated
    for patch_id in geometry.keys():

        cp = geometry[patch_id].get("B")
        control_points = cp[..., :-1]
        ref_weights = cp[..., -1, np.newaxis]

        #number of control points in each direction
        n, m = control_points.shape[0], control_points.shape[1]

        P = [] # List of control point
        W = [] # list of weights
        p_t = [] #list for assigning the control point the global degree of freedom
        local_counter = 0
        for y in range(m):
            for x in range(n):
                val = control_points[x, y, :].tolist()
                P.append(val)
                W.append(ref_weights[x, y, :])
                p_t.append(local_counter)
                cp_all.append(val)
                local_counter += 1

        geometry[patch_id]["ldof"] = local_counter
        geometry[patch_id]["glob_num"] = np.ones(local_counter)*-1 #This is just so I can identify easily the dof that haven't been assign yet

        geometry[patch_id]["cp_dof"] = np.asarray(p_t)
        geometry[patch_id]["list_cp"] = np.asarray(P)
        geometry[patch_id]["list_weight"] = np.asarray(W)

        fill_information(geometry[patch_id])
        if geometry[patch_id].get("patch_faces") is not None:
            patches_connection(geometry[patch_id])

    # when they have connectivity, we need to substarct the repeated control points so they dont repeat. This will be basically dividing the number of control points off all
    #global_degrees_freedom = counter #int(counter - shared_dof/2)
    if geometry[list(geometry.keys())[0]].get("cp_repeated") is not None:
        global_degrees_freedom = global_numbering_dof(geometry)
    else:
        n, m = geometry[list(geometry.keys())[0]].get("n_basis")
        global_degrees_freedom = n*m
        geometry[list(geometry.keys())[0]]["glob_num"] = np.arange(0, local_counter)

    return geometry, global_degrees_freedom

def fill_information(current_patch):
    """

    Args:
        current_patch:

    Returns:

    """
    B = current_patch.get("B")
    U, V = current_patch.get("knots")
    degree_u = len(np.where(U == 0.)[0]) - 1
    degree_v = len(np.where(V == 0.)[0]) - 1
    current_patch["degree"] = (degree_u, degree_v)

    # n_xi and n_eta are the number of elements
    n_xi = B.shape[0] - degree_u
    n_eta = B.shape[1] - degree_v
    current_patch["n_element"] = [n_xi, n_eta]
    #nel = n_xi * n_eta  # total number of elements
    n = n_xi + degree_u  # Number of basis functions/ control points in xi direction
    m = n_eta + degree_v  # Number of basis functions/ control points in eta direction
    current_patch["n_basis"] = [n, m]
    ncp = n * m  # Total number of control points
    pDof = ncp  # patch degrees of freedom - Temperature
    current_patch["patch_DOF"] = pDof
    assert degree_u == degree_v #cannot manage different degrees
    current_patch["IEN"] = IEN_element_topology(n_xi, n_eta, degree_u)

def patches_connection(geometry_patch):
    """
    #n_patches = geometry.__len__()
    # They repeat the control points in the upper and lower part
    #for patch_id in geometry.keys():
    n, m = geometry[patch_id].get("n_basis")
    dof = geometry[patch_id].get("patch_topology")
    sides = geometry[patch_id].get("patch_faces")
    position_repeated_cp = []
    for face in sides:
        if face == 1: #loking down
            position_repeated_cp.append(dof[:n])
        elif face == 2: # loking right side
            position_repeated_cp.append(dof[n-1::m])
        elif face == 3: # loking up
            position_repeated_cp.append(dof[-n:])
        elif face == 4: # loking left side
            position_repeated_cp.append(dof[::m])
    geometry[patch_id]["cp_repeated"] = np.asarray(position_repeated_cp)
    """
    n, m = geometry_patch.get("n_basis")
    dof = geometry_patch.get("cp_dof")
    sides = geometry_patch.get("patch_faces").keys()
    position_repeated_cp = [[],[],[],[]]
    for face in sides:
        if face == 0:  # loking down
            #position_repeated_cp.append(dof[:n])
            position_repeated_cp[0]= dof[:n]
        if face == 1:  # loking right side
            #position_repeated_cp.append(dof[n - 1::m])
            position_repeated_cp[1] = dof[n - 1::n]
        if face == 2:  # loking up
            #position_repeated_cp.append(dof[-n:])
            position_repeated_cp[2] = dof[-n:]

        if face == 3:  # loking left side
            #position_repeated_cp.append(dof[::m])
            position_repeated_cp[3] = dof[::n]
    geometry_patch["cp_repeated"] = np.asarray(position_repeated_cp, dtype=object)

def global_numbering_dof(geometry):
    """

    Args:
        geometry:

    Returns:

    """
    # First are the dof that are inside the patch
    glob_dof = 0
    for patch_id in geometry.keys():
        local_numbering = geometry[patch_id].get("cp_dof")
        repeated_cp = geometry[patch_id].get("cp_repeated")
        repeated_cp = [item for sublist in repeated_cp for item in sublist]

        non_repeated_cp = np.setdiff1d(local_numbering, repeated_cp)
        geometry[patch_id]["glob_num"][np.ix_(non_repeated_cp)] = glob_dof + np.arange(len(non_repeated_cp))
        glob_dof += len(non_repeated_cp)
    # Now we will fill the dof that are in the interface of the patches
    keys = geometry.keys()

    for idx, patch_id in enumerate(keys):
        ldof = geometry[patch_id].get("ldof")
        repeated_cp = geometry[patch_id].get("cp_repeated")

        patch_faces = geometry[patch_id].get("patch_faces")
        for c, cp in enumerate(repeated_cp):
            if len(cp) > 1 and cp[0] < ldof:
                ###new
                # and check for glob_num if already assigned before hand in the contact face

                new_dofs = glob_dof + np.arange(len(cp))
                temp_new_dof = len(new_dofs)

                before, = np.where(geometry[patch_id]["glob_num"][np.ix_(cp)] != -1)
                if len(before) > 0:
                    already_assigned = geometry[patch_id]["glob_num"][np.ix_(cp)][before]
                    for e in before:
                        if e == 0:
                            new_dofs -= 1 # we need to move all elements to the left
                        temp_new_dof -= 1
                    new_dofs[before] = already_assigned

                # This way we enter the next side patch and replace the sides
                contact = patch_faces[c]
                con_cp = geometry[contact].get("cp_repeated")
                con_face = geometry[contact].get("patch_faces")
                loc_face = [nface for nface in con_face.keys() if con_face[nface] == patch_id ][0]

                geometry[patch_id]["glob_num"][np.ix_(cp)] = new_dofs
                geometry[patch_id]["cp_repeated"][c] = new_dofs
                geometry[contact]["glob_num"][np.ix_(con_cp[loc_face])] = new_dofs
                geometry[contact]["cp_repeated"][loc_face] = new_dofs

                glob_dof += temp_new_dof#len(new_dofs)
        geometry[patch_id]["glob_num"] = geometry[patch_id]["glob_num"].astype("int")
    return glob_dof


def form_k_IGA_mp(geometry, K_glob):
    """

    Args:
        geometry:
        k_glob: empty global stiffness matrix

    Returns:

    """
    from pygeoiga.analysis.iga import form_k_IGA
    # form the stiffness matrix
    for patch_id in geometry.keys():
        pDof = geometry[patch_id].get("patch_DOF")
        K = np.zeros((pDof, pDof))  # stiffness matrix size

        P = geometry[patch_id].get("list_cp")
        W = geometry[patch_id].get("list_weight")
        n_xi, n_eta = geometry[patch_id].get("n_element")
        U, V = geometry[patch_id].get("knots")
        degree = geometry[patch_id].get("degree")[0]

        # Element topology
        IEN = geometry[patch_id].get("IEN")#IEN_element_topology(n_xi, n_eta, degree)
        kappa = geometry[patch_id].get("kappa")
        K = form_k_IGA(K, IEN, P, kappa, n_xi, n_eta, degree, knots=[U,V])
        geometry[patch_id]["K"]= K
        patch_glob_num = geometry[patch_id].get("glob_num")
        K_glob[np.ix_(patch_glob_num, patch_glob_num)] = K_glob[np.ix_(patch_glob_num,
                                                                       patch_glob_num)] + K

    return K_glob


def boundary_condition_mp(geometry, D, T_t, T_b, T_l, T_r):
    """
     Construct the "Drichlet" boundary conditions constrained by the top and bottom temperatures
    and get the corresponding degrees of freedom from the mesh - Strong imposed drichlet BC
    Args:
        geometry: information of all patches
        T_t: Top temperature/ can be a function as well
        T_b: Bottom temperature/ can be a function as well
        T_l: Left temperature/ can be a function as well
        T_r right temperature/ can be a function as well
        D: empty displacement vector
    Returns:
        dictionary containing the indexes where the boundary conditions are applied, the top and bottom temperature
    """
    max_col = 0
    max_row = 0
    for patch_id in geometry.keys():
        position = geometry[patch_id].get("position")
        if position[0] > max_row:
            max_row = position[0]
        if position[1] > max_col:
            max_col = position[1]
    acD = np.array([])
    for patch_id in geometry.keys():
        position = geometry[patch_id].get("position")
        n, m = geometry[patch_id].get("n_basis")
        patch_dof = geometry[patch_id].get("glob_num")
        if T_b is not None:
            #now we need to check if the position of the patch is in the bottom
            if position[0] == 1: # row is the first one
                prDOF = patch_dof[:n]
                if callable(T_b):
                    # T_b = lambda x, m: T_t + 10 * np.sin(np.pi * x / m)  # something like this
                    D[prDOF] = T_b((prDOF), len(prDOF))
                else:
                    D[prDOF] = T_b
                acD = np.hstack([acD, prDOF])

        if T_t is not None:
            if position[0] == max_row:  # row is the last one
                prDOF = patch_dof[-n:]
                if callable(T_t):
                    # T_b = lambda x, m: T_t + 10 * np.sin(np.pi * x / m)  # something like this
                    D[prDOF] = T_t((prDOF), len(prDOF))
                else:
                    D[prDOF] = T_t
                acD = np.hstack([acD, prDOF])

        if T_l is not None:
            if position[1] == 1: #column is the first one
                prDOF = patch_dof[::n]
                if callable(T_l):
                    # T_b = lambda x, m: T_t + 10 * np.sin(np.pi * x / m)  # something like this
                    D[prDOF] = T_l((prDOF), len(prDOF))
                else:
                    D[prDOF] = T_l
                acD = np.hstack([acD, prDOF])

        if T_r is not None:
            if position[1] == max_col: #column is the last one
                prDOF = patch_dof[n - 1::n]
                if callable(T_r):
                    # T_b = lambda x, m: T_t + 10 * np.sin(np.pi * x / m)  # something like this
                    D[prDOF] = T_r((prDOF), len(prDOF))
                else:
                    D[prDOF] = T_r
                acD = np.hstack([acD, prDOF])

    acD = acD.astype('int')
    return {"prDOF": np.unique(acD)}, D

def bezier_extraction_mp(geometry):
    from pygeoiga.analysis.bezier_extraction import bezier_extraction_operator_bivariate, bezier_extraction_operator

    for patch_id in geometry.keys():
        knots = geometry[patch_id].get("knots")
        degree = geometry[patch_id].get("degree")
        n_xi, n_eta = geometry[patch_id].get("n_element")
        assert degree[0] == degree[1], "Degree of the geometry is not the same"
        degree = degree[0]

        C_xi = bezier_extraction_operator(knots[0], degree)
        C_eta = bezier_extraction_operator(knots[1], degree)
        C = bezier_extraction_operator_bivariate(C_xi, C_eta, n_xi, n_eta, degree)

        geometry[patch_id]["bezier_extraction"] = C

    return geometry

def form_k_bezier_mp(geometry, K_glob):
    """
    Form stiffness matrix based on bezier extraction for multiple patches
    Args:
        geometry:
        k_glob: empty global stiffness matrix

    Returns:

    """
    from pygeoiga.analysis.bezier_FE import form_k
    #K_glob = np.zeros((gDoF, gDoF))
    for patch_id in geometry.keys():
        pDof = geometry[patch_id].get("patch_DOF")  # Degrees of freedom per patch
        nx, ny = geometry[patch_id].get("n_element")  # number of elements in parametric space
        nel = nx * ny  # total number of elements
        K = np.zeros((pDof, pDof))  # Initialize empty patch stiffness matrix

        P = geometry[patch_id].get("list_cp")  # Get list with location of control points
        W = geometry[patch_id].get("list_weight")  # Get list of weights
        # Support only the same degree for both parametric directions
        degree = geometry[patch_id].get("degree")[0]

        # Bezier exatractor operator of the patch
        C = geometry[patch_id].get("bezier_extraction")

        kappa = geometry[patch_id].get("kappa")  # Get thermal conductivity of patch

        IEN = geometry[patch_id].get("IEN")  # Global connectivity array (Element topology)

        K = form_k(K, IEN, P, kappa, nel, degree, W, C)
        geometry[patch_id]["K"] = K
        patch_glob_num = geometry[patch_id].get("glob_num")
        K_glob[np.ix_(patch_glob_num,
                      patch_glob_num)] = K_glob[np.ix_(patch_glob_num,
                                                       patch_glob_num)] + K

    return K_glob


def map_MP_elements(geometry, D):
    """

    Args:
        geometry:
        D:
        gDoF: global degrees of freedom

    Returns:

    """
    from pygeoiga.analysis.bezier_FE import map_bezier_elements
    for patch_id in geometry.keys():
        degree, _ = geometry[patch_id].get("degree")
        P = geometry[patch_id].get("list_cp")
        nx, ny = geometry[patch_id].get("n_element")
        n, m = geometry[patch_id].get("n_basis")
        ncp = n*m
        W = geometry[patch_id].get("list_weight")
        knots = geometry[patch_id].get("knots")
        glob_num = geometry[patch_id].get("glob_num")
        D_patch = D[glob_num]
        IEN = geometry[patch_id].get("IEN")

        C = geometry[patch_id].get("bezier_extraction")
        #C = None
        if C is not None:
            x_temp, y_temp, t_temp = map_bezier_elements(D_patch, degree, P, n, m, ncp, nx*ny, IEN, W, C)
        else:
            x_temp, y_temp, t_temp = map_solution_elements(D_patch, degree, P, nx, ny, n, m, ncp, IEN, W, knots)
        geometry[patch_id]["x_sol"] = x_temp
        geometry[patch_id]["y_sol"] = y_temp
        geometry[patch_id]["t_sol"] = t_temp #TODO: What is happening here?

    return geometry

def point_solution_mp(x, y, geometry, **kwargs):
    """
    Get the value of the field (temperature) at the location (x, y) from multiple patches.
    Args:
        x: x coordinate
        y: y coordinate
        geometry: all patch information
        kwargs: tolerance:
                itera:
    Returns:
    """
    from pygeoiga.analysis.common import point_solution
    for patch_id in geometry.keys():
        B = geometry[patch_id].get("B")
        knots = geometry[patch_id].get("knots")
        temp = geometry[patch_id].get("t_sol").ravel()
        t = point_solution(x, y, temp, B, knots, **kwargs)
        if t is not None:
            break
    return t