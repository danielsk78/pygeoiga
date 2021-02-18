import numpy as np
import matplotlib.pyplot as plt
import pygeoiga as gn
fig_folder=gn.myPath+'/../manuscript_IGA_MasterThesis/manuscript/Thesis/figures/06_pyGeoIGA/'
kwargs_savefig=dict(transparent=True, box_inches='tight', pad_inches=0)
save_all=False

def test_create_geometry():

    def plot(geometry, file_name):
        from pygeoiga.plot.nrbplotting_mpl import p_surface, p_cpoints, p_knots

        fig, [ax2, ax] = plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
        #fig2, ax2 = plt.subplots(constrained_layout=True)

        for patch_id in geometry.keys():
            ax = p_knots(geometry[patch_id].get("knots"),
                           geometry[patch_id].get("B"),
                         ax=ax,
                           color='k',
                           dim=2,
                         point=False,
                         line=True)
            print(patch_id, geometry[patch_id].get("knots"))
            ax = p_surface(geometry[patch_id].get("knots"),
                           geometry[patch_id].get("B"),
                           color=geometry[patch_id].get("color"),
                           dim=2,
                           fill=True,
                           border=False,
                           ax=ax)

            ax2 = p_cpoints(geometry[patch_id].get("B"),
                           dim=2,
                           ax=ax2,
                           point=True,
                           marker="o",
                           color=geometry[patch_id].get("color"),
                            linestyle="-",
                           line=True)

        ax.legend(labels=list(geometry.keys()),
                  handles=ax.patches,
                  loc='upper left',
                  bbox_to_anchor=(1., .5),
                  borderaxespad=0)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_aspect("equal")
        ax.set_ylabel(r"$y$")
        ax.set_xlabel(r"$x$")

        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.set_aspect("equal")
        ax2.set_ylabel(r"$y$")
        ax2.set_xlabel(r"$x$")

        fig.show()

        save = False
        if save or save_all:
            fig.savefig(fig_folder + file_name, **kwargs_savefig)

    from pygeoiga.nurb.cad import make_3_layer_patches

    geometry = make_3_layer_patches()
    plot(geometry, "anticline.pdf")
    geometry2 = make_3_layer_patches(refine=True, knot_ins=[np.arange(0.25,1,0.25), np.arange(0.25,1,0.25)])
    plot(geometry2, "anticline_refined_2.pdf")
    geometry3 = make_3_layer_patches(refine=True, knot_ins=[np.arange(0.1,1,0.1), np.arange(0.1,1,0.1)])
    plot(geometry3, "anticline_refined_3.pdf")

def test_listings():
    import numpy as np
    from collections import OrderedDict

    def create_three_layer_model():
        # Lower layer control points
        bottom_c = np.array([[[0., 0., 1.], [0., 50., 1.], [0., 100., 1.]],
                             [[250., 0., 1.], [250., 180., 1.], [250., 360., 1.]],
                             [[500., 0., 1.], [500., 50., 1.], [500., 100., 1.]]])
        knot_b = ([0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1])  # knot vector (U, V)

        # Middle layer control points
        middle_c = np.array([[[0., 100., 1.], [0., 200., 1.], [0., 300., 1.]],
                             [[250., 360., 1.], [250., 380., 1.], [250., 400., 1.]],
                             [[500., 100., 1.], [500., 200., 1.], [500., 300., 1.]]])
        knot_m = ([0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1])  # knot vector (U, V)

        # Upper layer control points
        upper_c = np.array([[[0., 300., 1.], [0., 400., 1.], [0., 500., 1.]],
                            [[250., 400., 1.], [250., 450., 1.], [250., 500., 1.]],
                            [[500., 300., 1.], [500., 400., 1.], [500., 500., 1.]]])
        knot_u = ([0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1])  # knot vector (U, V)

        cpoints = [bottom_c, middle_c, upper_c]
        knots = [knot_b, knot_m, knot_u]

        geometry = OrderedDict({})
        name = ["granite", "mudstone", "sandstone"]  # type of litholgy

        for i, lith in enumerate(name):
            geometry[lith] = {"B": cpoints[i], "knots": knots[i]}

        return geometry

    def assign_properties(geometry: dict):

        color = ["red", "blue", "green"]  # Assign a fixed color to the NURBS. Useful for visualization
        kappa = [3.1, 0.9, 3]  # Thermal conductivity of the layer [W/mK]
        position = [(1, 1), (2, 1), (3, 1)]  # Position of the patch in a global grid (Row and colum)
        for i, patch_name in enumerate(geometry.keys()):
            geometry[patch_name]["kappa"] = kappa[i]
            geometry[patch_name]["color"] = color[i]
            geometry[patch_name]["position"] = position[i]

        # Topology of patches -BOUNDARIES - faces of the patch in contact
        # 0: down; 1: right; 2: up; 3: left
        # Granite is in contact to mudstone in the "up" face
        geometry["granite"]["patch_faces"] = {2: "mudstone"}
        # Mudstone is in contact to granite in the "down" face and to sandstone in the "up" face
        geometry["mudstone"]["patch_faces"] = {0: "granite", 2: "sandstone"}
        # Sandstone is in contact to mudstone in the "down" face
        geometry["sandstone"]["patch_faces"] = {0: "mudstone"}

        # Specify the face that have a boundary condition
        #geometry["granite"]["BC"] = {0: "bot_bc"}
        #geometry["sandstone"]["BC"] = {2: "top_bc"}

        return geometry

    def refinement(geometry: dict):
        # Knot insertion following the computational procedure of Piegl and Tille (1997)
        from pygeoiga.nurb.refinement import knot_insertion

        knot_ins_0 = np.arange(0.1, 1, 0.1)
        knot_ins_1 = np.arange(0.1, 1, 0.1)

        for count, patch_name in enumerate(geometry.keys()):
            B = geometry[patch_name].get("B")
            knots = geometry[patch_name].get("knots")

            B, knots = knot_insertion(B, degree=(2, 2), knots=knots,
                                      knots_ins=knot_ins_0, direction=0)  # U refinement
            B, knots = knot_insertion(B, degree=(2, 2), knots=knots,
                                      knots_ins=knot_ins_1, direction=1)  # V refinement

            geometry[patch_name]["B"] = B
            geometry[patch_name]["knots"] = knots

        return geometry

    def form_k_IGA(K_glb: np.ndarray,
                   IEN: np.ndarray,
                   P: list,
                   kappa: float,
                   nx: int,
                   ny: int,
                   degree: int,
                   knots: float):
        """
           Function to form the stiffness matrix
           Args:
               K_glb: empty stiffness matrix
               IEN: element topology = numbering of control points
               P: coordinates of NURBS control points x, y, z in single matrix (:,3)
               kappa: Thermal conductivity
               nx: number of elements in u direction
               ny: number of elments in v direction
               degree: polynomial order
               knots: knot vector (U, V)
           Returns:
               K -> Stiffness matrix
           """

        from pygeoiga.analysis.common import gauss_points
        from pygeoiga.analysis.iga import nurb_basis_IGA, jacobian_IGA
        # Gauss quadrature for performing numerical integration. Gauss points and weights
        G, W = gauss_points(degree)

        # Assign a value of thermal conductivity to each control point and calculate the element thermal conductivity
        #kappa_element = kappa_domain(kappa, IEN)

        # Knot vector from each parametric direction
        U = knots[0][degree:-degree]
        V = knots[1][degree:-degree]

        e = 0
        for i in range(ny):
            for j in range(nx):
                IEN_e = IEN[e]  # Element topology of current element
                eDOF = IEN_e  # Element degrees of freedom
                #kappa_e = kappa_element[e] #
                k = 0
                for g in range(len(G)):
                    # Obtain gauss points from reference element
                    xi = G[g, 0]  # Gauss point in xi direction
                    eta = G[g, 1]  # Gauss point in eta direction
                    w = W[g]  # weight of Gauss point

                    # Map Gauss points to parameter space
                    u = U[j] + (xi + 1) * (U[j + 1] - U[j]) / 2
                    v = V[i] + (eta + 1) * (V[i + 1] - V[i]) / 2

                    # Evaluate basis functions and derivatives
                    N_u, dN_u, N_v, dN_v = nurb_basis_IGA(u, v, knots=knots, degree=degree)

                    # Map of derivatives from parameter space to reference element
                    dN_xi = dN_u / 2
                    dN_eta = dN_v / 2

                    # Collect the basis functions and derivatives which support the element
                    N_u = N_u[j:j + degree + 1]
                    dN_xi = dN_xi[j:j + degree + 1]
                    N_v = N_v[i:i + degree + 1]
                    dN_eta = dN_eta[i:i + degree + 1]

                    # Calculate the Jacobian to map from the reference element to physical space
                    J, dxy = jacobian_IGA(N_u, dN_xi, N_v, dN_eta, P, IEN_e, degree)

                    # add contributions to local stiffness matrix
                    k = k + (dxy.T @ dxy) * np.linalg.det(J) * w * kappa#kappa_e
                # Add local stiffness matrix to global stiffness matrix
                K_glb[np.ix_(eDOF, eDOF)] = K_glb[np.ix_(eDOF, eDOF)] + k
                e += 1
        return K_glb

    geometry = create_three_layer_model()
    geometry = assign_properties(geometry)
    geometry = refinement(geometry)

    from pygeoiga.analysis.MultiPatch import patch_topology
    # Extract the information from each patch and create a global numbering of the multi-patch geometry.
    # The total amount of non-repeated control points will be global degrees of freedom
    geometry, gDoF = patch_topology(geometry)
    print(gDoF)

    def assemble_stiffness_matrix(geometry: dict, gDoF: int):
        """
        Args:
            geometry: NURBS multipatch
            gDoF: Global degrees of freedom
        Return:
            K_glob: Global stiffnes matrix
        """
        # Set empty the stiffness matrix according to the global degrees of freedom
        K_glob = np.zeros((gDoF, gDoF))

        # iterate over the patches
        for patch_id in geometry.keys():
            pDof = geometry[patch_id].get("patch_DOF")  # Degrees of freedom per patch
            K = np.zeros((pDof, pDof))  # Initialize empty patch stiffness matrix

            nx, ny = geometry[patch_id].get("n_element")  # number of elements in u and v parametric coordinates
            U, V = geometry[patch_id].get("knots")  # knot vectors
            degree = geometry[patch_id].get("degree")  # degree of NURBS patch
            # Currently support only the same degree for both parametric directions
            assert degree[0] == degree[1]
            degree = degree[0]

            P = geometry[patch_id].get("list_cp")  # Get list with location of control points
            IEN = geometry[patch_id].get("IEN")  # connectivity array (element topology)
            kappa = geometry[patch_id].get("kappa")  # Patch thermal conductivity

            # create patch stiffness matrix
            K = form_k_IGA(K, IEN, P, kappa, nx, ny, degree, knots=(U, V))

            # Assemble global stiffness matrix according to global indexing
            patch_glob_num = geometry[patch_id].get("glob_num")
            K_glob[np.ix_(patch_glob_num,
                          patch_glob_num)] = K_glob[np.ix_(patch_glob_num,
                                                           patch_glob_num)] + K

        return K_glob

    K_glob = assemble_stiffness_matrix(geometry, gDoF)
    fig, ax = plt.subplots(constrained_layout=True)
    ax.spy(K_glob)
    ax.set_xlabel("Global $K$ of Anticline multi-patch geometry")
    fig.show()
    save = False
    if save or save_all:
        fig.savefig(fig_folder+"stiffnes_matrix.pdf", **kwargs_savefig)

    from pygeoiga.analysis.MultiPatch import boundary_condition_mp
    a = np.zeros(gDoF)

    T_top = 10; T_bottom = 25; T_left = None; T_right = None
    # Also possible to pass a function as
    #T_bottom = lambda x, m: 10 * np.sin(np.pi * x / m)
    # with x the position of the node and m the total number of nodes
    bc, a = boundary_condition_mp(geometry, a, T_top, T_bottom, T_left, T_right)
    bc["gDOF"] = gDoF

    from scipy.sparse.linalg import cg

    # Empty Force vector
    F = np.zeros(gDoF)

    def solve(bc: dict, K: np.ndarray, F: np.ndarray, a: np.ndarray):
        prDOF = bc.get("prDOF")  # list of indexes for control points with boundary condition
        gDOF = bc.get("gDOF")  # Degrees of freedom for the system

        # Find the active control points
        acDOF = np.setxor1d(np.arange(0, gDOF), prDOF).astype('int')

        # Reduced stiffness matrix using only active control points
        Kfs = K[np.ix_(acDOF, prDOF)]
        f = F[acDOF]
        bf_n = f - Kfs @ a[prDOF]

        # Solve for the system of equations for displacement vector
        a[acDOF], _ = cg(A=K[np.ix_(acDOF, acDOF)], b=bf_n)

        # Calculate Force vector
        F = K @ a

        return a, F

    a, F = solve(bc, K_glob, F, a)

    from pygeoiga.analysis import map_IGA_elements

    def map_MP_elements(geometry, a):
        for patch_id in geometry.keys():
            degree, _ = geometry[patch_id].get("degree")  # Degree of NURBS patch
            P = geometry[patch_id].get("list_cp")  # List of control points associated to the patch
            nx, ny = geometry[patch_id].get("n_element")  # number of elements in u and v diection
            n, m = geometry[patch_id].get("n_basis")  # number of basis functions in u and v direction
            ncp = n * m  # Total amount of control points
            W = geometry[patch_id].get("list_weight")  # List of weights
            knots = geometry[patch_id].get("knots")  # knot vectors in u and v
            glob_num = geometry[patch_id].get("glob_num")  # global index numbering

            a_patch = a[glob_num]  # Extract from the displacement vector the values corresponding to the patch control points
            IEN = geometry[patch_id].get("IEN")  # Connectivity array (element topology)

            # Procedure to obtain the coordinate x and y of the temperature value t
            x_temp, y_temp, t_temp = map_IGA_elements(a_patch, degree, P, nx, ny, n, m, ncp, IEN, W, knots)

            geometry[patch_id]["x_sol"] = x_temp
            geometry[patch_id]["y_sol"] = y_temp
            geometry[patch_id]["t_sol"] = t_temp

        return geometry

    geometry = map_MP_elements(geometry, a)

    fig_sol, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(15,5), sharey=True)
    from pygeoiga.plot.nrbplotting_mpl import p_surface, p_cpoints, p_knots, np
    from pygeoiga.plot.solution_mpl import  p_temperature

    for patch_id in geometry.keys():
        ax1 = p_surface(geometry[patch_id].get("knots"),
                       geometry[patch_id].get("B"),
                       color=geometry[patch_id].get("color"),
                       dim=2,
                       fill=True,
                       border=False,
                       ax=ax1)

        x = geometry[patch_id].get("x_sol")
        y = geometry[patch_id].get("y_sol")
        t = geometry[patch_id].get("t_sol")

        ax2 = p_temperature(x, y, t, vmin=np.min(a), vmax=np.max(a), levels=100, show=False, colorbar=False,
                                 ax=ax2, point=True, fill=False, markersize=50)

        ax2 = p_knots(geometry[patch_id].get("knots"),
                     geometry[patch_id].get("B"),
                     ax=ax2,
                     color='k',
                     dim=2,
                     point=False,
                     line=True,
                      linestyle="--",
                      linewidth=0.2)
        #ax2 = p_cpoints(geometry[patch_id].get("B"), dim=2, ax=ax2, point=True, line=False)
        ax3 = p_temperature(x, y, t, vmin=np.min(a), vmax=np.max(a), levels=200, show=False, colorbar=True, ax=ax3,
                               point=False, fill=True, contour=False)

    ax1.legend(labels=list(geometry.keys()),
              handles=ax1.patches,
              loc='upper left',
            bbox_to_anchor=(0.05, .9),
             borderaxespad=0)

    for ax in ax1, ax2, ax3:
        ax.set_aspect("equal")
        ax.set_ylabel(r"$y$")
        ax.set_xlabel(r"$x$")
        ax.set_xlim(0,500)
        ax.set_ylim(0,500)

    for c in ax3.collections:
        c.set_edgecolor("face")

    plt.tight_layout()
    fig_sol.show()

    save = False
    if save or save_all:
        fig_sol.savefig(fig_folder + "solution_anticline.pdf", **kwargs_savefig)